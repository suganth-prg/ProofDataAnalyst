import json
import re
import math
import pandas as pd
import ollama

import analyzer


# ============================================================
# DYNAMIC PROOF-CARRYING DATA ANALYZER
# ============================================================

MODEL = "llama3.2:3b"


# ============================================================
# DATASET DISCOVERY
# ============================================================

def get_datasets():
    datasets = {}

    if hasattr(analyzer, "DATASETS"):
        for name, df in analyzer.DATASETS.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                datasets[name] = df

    # Fallback for current analyzer structure
    if not datasets:

        if hasattr(analyzer, "sales"):
            datasets["sales"] = analyzer.sales

        if hasattr(analyzer, "customers"):
            datasets["customers"] = analyzer.customers

        if hasattr(analyzer, "inventory"):
            datasets["inventory"] = analyzer.inventory

    return datasets


def describe_datasets():

    datasets = get_datasets()

    description = {}

    for name, df in datasets.items():

        description[name] = {
            "rows": len(df),
            "columns": list(df.columns),
            "types": {
                str(column): str(dtype)
                for column, dtype in df.dtypes.items()
            },
            "sample": df.head(3).fillna("").to_dict(
                orient="records"
            )
        }

    return description


# ============================================================
# LLM JSON CLEANER
# ============================================================

def clean_json(text):

    text = text.strip()

    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text.strip()


# ============================================================
# ASK LLM FOR ANALYSIS PLAN
# ============================================================

def create_plan(question):

    datasets = describe_datasets()

    prompt = f"""
You are the analysis planner for a proof-carrying data analyst.

The user asks a natural-language question about available datasets.

Your job is NOT to calculate the answer.

Your job is to determine exactly what computation is required.

AVAILABLE DATASETS:

{json.dumps(datasets, indent=2, default=str)}

USER QUESTION:

{question}

Return ONLY valid JSON.

Use this schema:

{{
    "status": "ok | ambiguous | unsupported",
    "dataset": "dataset name",
    "columns": [],
    "filters": [],
    "group_by": [],
    "operation": "sum | average | count | min | max | ranking | sort | percentage | ratio | difference | arithmetic | list | correlation | custom",
    "metric": "column name or expression",
    "secondary_metric": "column name or expression or null",
    "direction": "ascending | descending | null",
    "limit": null,
    "calculation": "plain description of required calculation",
    "reason": "short explanation"
}}

Rules:

1. Use the ACTUAL dataset and column names.
2. Do not invent columns.
3. Natural language can refer to a column indirectly.
4. "money earned", "revenue", "income", etc. may refer to a numeric sales/revenue column if supported by the dataset.
5. "how many" can mean count or sum depending on context.
6. "best", "worst", "top", or "bottom" is ambiguous if the metric is not clear.
7. Do not calculate the answer.
8. Do not invent missing data.
9. If the requested calculation cannot be determined from the datasets, use unsupported.
10. If multiple interpretations are possible and materially change the answer, use ambiguous.
11. The generated plan must be reproducible using Python/Pandas.
12. Never combine quantities with incompatible units.
13. If the question asks for a ranking, identify both the ranking metric and the entity column.
14. For group analysis, identify the grouping column.
15. For arithmetic, identify every required column.
16. For percentage questions, identify numerator and denominator.
17. For ratio questions, identify numerator and denominator.
18. For comparisons, identify both values being compared.

Examples:

Question:
How much did Rice make?

Possible interpretation:
sales dataset
Product = Rice
metric = Sales
operation = sum

Question:
Which product sold the most units?

Possible interpretation:
sales dataset
group/entity = Product
metric = Units
operation = ranking
direction = descending

Question:
What percentage of sales came from fruit?

Possible interpretation:
sales dataset
filter Category = Fruit
metric = Sales
operation = percentage

Question:
What is the average sales per unit?

Possible interpretation:
sales dataset
metric = Sales / Units
operation = arithmetic

Question:
Which category generated the most money?

Possible interpretation:
sales dataset
group_by = Category
metric = Sales
operation = ranking
direction = descending

Question:
What is the best product?

This is ambiguous because "best" has no defined metric.

Return JSON only.
"""

    try:

        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        raw = response["message"]["content"]

        cleaned = clean_json(raw)

        return json.loads(cleaned)

    except Exception as e:

        return {
            "status": "unsupported",
            "dataset": "",
            "columns": [],
            "filters": [],
            "group_by": [],
            "operation": "",
            "metric": "",
            "secondary_metric": None,
            "direction": None,
            "limit": None,
            "calculation": "",
            "reason": f"LLM planning error: {e}"
        }


# ============================================================
# PLAN VALIDATION
# ============================================================

def validate_plan(plan):

    datasets = get_datasets()

    status = plan.get("status")

    if status not in {
        "ok",
        "ambiguous",
        "unsupported"
    }:
        return False, "Invalid analysis status."

    if status != "ok":
        return True, plan.get(
            "reason",
            "The question cannot be answered reliably."
        )

    dataset_name = plan.get("dataset")

    if dataset_name not in datasets:
        return False, (
            f"Dataset '{dataset_name}' is not available."
        )

    df = datasets[dataset_name]

    available_columns = list(df.columns)

    # Check referenced columns
    for column in plan.get("columns", []):

        if column not in available_columns:

            return False, (
                f"Column '{column}' does not exist "
                f"in dataset '{dataset_name}'."
            )

    # Metric can be an expression, so only directly
    # validate plain column names.
    metric = plan.get("metric")

    if (
        metric
        and metric not in available_columns
        and not is_expression(metric)
    ):

        return False, (
            f"Metric '{metric}' does not exist "
            f"in dataset '{dataset_name}'."
        )

    secondary = plan.get("secondary_metric")

    if (
        secondary
        and secondary not in available_columns
        and not is_expression(secondary)
    ):

        return False, (
            f"Secondary metric '{secondary}' "
            f"does not exist."
        )

    for column in plan.get("group_by", []):

        if column not in available_columns:

            return False, (
                f"Grouping column '{column}' "
                f"does not exist."
            )

    return True, ""


# ============================================================
# EXPRESSION DETECTION
# ============================================================

def is_expression(value):

    if not isinstance(value, str):
        return False

    operators = [
        "+",
        "-",
        "*",
        "/",
        "(",
        ")"
    ]

    return any(
        operator in value
        for operator in operators
    )


# ============================================================
# SAFE FILTER GENERATION
# ============================================================

def apply_filters(df, filters):

    data = df.copy()

    for item in filters:

        if not isinstance(item, dict):
            continue

        column = item.get("column")
        operator = item.get("operator", "equals")
        value = item.get("value")

        if column not in data.columns:
            raise ValueError(
                f"Unknown filter column: {column}"
            )

        if operator == "equals":

            data = data[
                data[column].astype(str).str.casefold()
                ==
                str(value).casefold()
            ]

        elif operator == "not_equals":

            data = data[
                data[column].astype(str).str.casefold()
                !=
                str(value).casefold()
            ]

        elif operator == "greater":

            data = data[
                pd.to_numeric(
                    data[column],
                    errors="coerce"
                ) > float(value)
            ]

        elif operator == "less":

            data = data[
                pd.to_numeric(
                    data[column],
                    errors="coerce"
                ) < float(value)
            ]

        elif operator == "greater_equal":

            data = data[
                pd.to_numeric(
                    data[column],
                    errors="coerce"
                ) >= float(value)
            ]

        elif operator == "less_equal":

            data = data[
                pd.to_numeric(
                    data[column],
                    errors="coerce"
                ) <= float(value)
            ]

    return data


# ============================================================
# EXPRESSION EVALUATION
# ============================================================

def evaluate_expression(data, expression):

    expression = expression.strip()

    # Direct column
    if expression in data.columns:
        return pd.to_numeric(
            data[expression],
            errors="coerce"
        )

    # Only allow expressions containing actual columns
    # and arithmetic operators.
    allowed_names = {
        column: pd.to_numeric(
            data[column],
            errors="coerce"
        )
        for column in data.columns
    }

    # Replace column names safely.
    expression_python = expression

    # Longest names first
    for column in sorted(
        data.columns,
        key=len,
        reverse=True
    ):

        expression_python = expression_python.replace(
            column,
            f"__COL_{list(data.columns).index(column)}"
        )

    local_values = {}

    for index, column in enumerate(data.columns):

        local_values[
            f"__COL_{index}"
        ] = allowed_names[column]

    # Restrict expression
    if not re.fullmatch(
        r"[\w\s\+\-\*\/\(\)\.\_]+",
        expression_python
    ):

        raise ValueError(
            "Unsafe arithmetic expression."
        )

    return eval(
        expression_python,
        {
            "__builtins__": {}
        },
        local_values
    )


# ============================================================
# GENERATE PANDAS CODE
# ============================================================

def generate_code(plan):

    dataset = plan["dataset"]
    operation = plan["operation"]

    metric = plan.get("metric")
    secondary = plan.get("secondary_metric")

    group_by = plan.get("group_by", [])

    direction = plan.get(
        "direction"
    )

    limit = plan.get("limit")

    lines = []

    lines.append(
        f'data = {dataset}.copy()'
    )

    # --------------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # --------------------------------------------------------

    lines.append(
        "data = data.drop_duplicates()"
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    for item in plan.get("filters", []):

        column = item.get("column")
        operator = item.get(
            "operator",
            "equals"
        )
        value = item.get("value")

        if operator == "equals":

            lines.append(
                f'data = data['
                f'data["{column}"]'
                f'.astype(str).str.casefold() '
                f'== "{str(value).casefold()}"'
                f']'
            )

        elif operator == "greater":

            lines.append(
                f'data = data['
                f'data["{column}"] > {float(value)}'
                f']'
            )

        elif operator == "less":

            lines.append(
                f'data = data['
                f'data["{column}"] < {float(value)}'
                f']'
            )

    # --------------------------------------------------------
    # METRIC PREPARATION
    # --------------------------------------------------------

    if metric and metric in analyzer.DATASETS[dataset].columns:

        lines.append(
            f'data = data.dropna(subset=["{metric}"])'
        )

    # --------------------------------------------------------
    # GROUP BY
    # --------------------------------------------------------

    if operation in {
        "sum",
        "average",
        "min",
        "max",
        "ranking"
    } and group_by:

        group_text = ", ".join(
            f'"{column}"'
            for column in group_by
        )

        if metric in analyzer.DATASETS[dataset].columns:

            if operation == "sum":

                lines.append(
                    f'result = data.groupby('
                    f'[{group_text}])'
                    f'["{metric}"].sum()'
                )

            elif operation == "average":

                lines.append(
                    f'result = data.groupby('
                    f'[{group_text}])'
                    f'["{metric}"].mean()'
                )

            elif operation == "min":

                lines.append(
                    f'result = data.groupby('
                    f'[{group_text}])'
                    f'["{metric}"].min()'
                )

            elif operation == "max":

                lines.append(
                    f'result = data.groupby('
                    f'[{group_text}])'
                    f'["{metric}"].max()'
                )

            elif operation == "ranking":

                lines.append(
                    f'result = data.groupby('
                    f'[{group_text}])'
                    f'["{metric}"].sum()'
                    f'.sort_values('
                    f'ascending='
                    f'{direction != "descending"}'
                    f')'
                )

                if limit:
                    lines.append(
                        f'result = result.head({int(limit)})'
                    )

        else:

            lines.append(
                f'values = evaluate_expression(data, '
                f'{metric!r})'
            )

            lines.append(
                f'data["__metric__"] = values'
            )

            lines.append(
                f'result = data.groupby('
                f'[{group_text}])'
                f'["__metric__"].sum()'
            )

    # --------------------------------------------------------
    # SIMPLE OPERATIONS
    # --------------------------------------------------------

    elif operation in {
        "sum",
        "average",
        "min",
        "max"
    }:

        if metric in analyzer.DATASETS[dataset].columns:

            if operation == "sum":
                lines.append(
                    f'result = data["{metric}"].sum()'
                )

            elif operation == "average":
                lines.append(
                    f'result = data["{metric}"].mean()'
                )

            elif operation == "min":
                lines.append(
                    f'result = data["{metric}"].min()'
                )

            elif operation == "max":
                lines.append(
                    f'result = data["{metric}"].max()'
                )

        else:

            lines.append(
                f'data["__metric__"] = '
                f'evaluate_expression(data, {metric!r})'
            )

            if operation == "sum":
                lines.append(
                    'result = data["__metric__"].sum()'
                )

            elif operation == "average":
                lines.append(
                    'result = data["__metric__"].mean()'
                )

            elif operation == "min":
                lines.append(
                    'result = data["__metric__"].min()'
                )

            elif operation == "max":
                lines.append(
                    'result = data["__metric__"].max()'
                )

    # --------------------------------------------------------
    # COUNT
    # --------------------------------------------------------

    elif operation == "count":

        lines.append(
            "result = len(data)"
        )

    # --------------------------------------------------------
    # LIST
    # --------------------------------------------------------

    elif operation == "list":

        column = metric

        if column not in data.columns:
            raise ValueError(
                f"Cannot list unknown column: {column}"
            )

        lines.append(
            f'result = data["{column}"]'
            '.dropna()'
            '.drop_duplicates()'
            '.tolist()'
        )

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    elif operation == "ranking":

        if (
            group_by
            and metric in data.columns
        ):

            group = group_by[0]

            lines.append(
                f'result = data.groupby("{group}")'
                f'["{metric}"].sum()'
                f'.sort_values('
                f'ascending='
                f'{direction != "descending"}'
                f')'
            )

            if limit:
                lines.append(
                    f'result = result.head({int(limit)})'
                )

        else:

            raise ValueError(
                "Ranking requires a grouping/entity column."
            )

    # --------------------------------------------------------
    # PERCENTAGE
    # --------------------------------------------------------

    elif operation == "percentage":

        if metric not in data.columns:
            raise ValueError(
                "Percentage metric unavailable."
            )

        lines.append(
            f'total = data["{metric}"].sum()'
        )

        lines.append(
            'result = '
            f'data["{metric}"].sum() / total * 100'
        )

    # --------------------------------------------------------
    # RATIO
    # --------------------------------------------------------

    elif operation == "ratio":

        if (
            metric not in data.columns
            or secondary not in data.columns
        ):

            raise ValueError(
                "Ratio requires two numeric columns."
            )

        lines.append(
            f'data = data.dropna('
            f'subset=["{metric}", "{secondary}"])'
        )

        lines.append(
            f'result = '
            f'data["{metric}"].sum() / '
            f'data["{secondary}"].sum()'
        )

    # --------------------------------------------------------
    # DIFFERENCE
    # --------------------------------------------------------

    elif operation == "difference":

        if (
            metric not in data.columns
            or secondary not in data.columns
        ):

            raise ValueError(
                "Difference requires two numeric columns."
            )

        lines.append(
            f'result = '
            f'data["{metric}"].sum() - '
            f'data["{secondary}"].sum()'
        )

    # --------------------------------------------------------
    # ARITHMETIC
    # --------------------------------------------------------

    elif operation == "arithmetic":

        if not metric:
            raise ValueError(
                "Arithmetic metric is missing."
            )

        lines.append(
            f'data["__metric__"] = '
            f'evaluate_expression(data, {metric!r})'
        )

        lines.append(
            'result = data["__metric__"]'
        )

    else:

        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    lines.append("result")

    return "\n".join(lines)


# ============================================================
# EXECUTE GENERATED CODE
# ============================================================

def execute_code(code):

    datasets = get_datasets()

    namespace = {
        **datasets,
        "pd": pd,
        "math": math,
        "evaluate_expression": evaluate_expression
    }

    try:

        exec(
            code,
            {
                "__builtins__": {}
            },
            namespace
        )

        return namespace.get(
            "result"
        ), None

    except Exception as e:

        return None, str(e)


# ============================================================
# RESULT NORMALIZATION
# ============================================================

def normalize_result(result):

    if isinstance(result, pd.Series):

        return {
            str(key): normalize_result(value)
            for key, value in result.to_dict().items()
        }

    if isinstance(result, pd.DataFrame):

        return result.to_dict(
            orient="records"
        )

    if isinstance(result, (pd.Timestamp,)):

        return str(result)

    if hasattr(result, "item"):

        try:
            return result.item()
        except Exception:
            pass

    if isinstance(result, float):

        if math.isnan(result):
            return None

    return result


# ============================================================
# INDEPENDENT VERIFICATION
# ============================================================

def verify_result(plan, result):

    datasets = get_datasets()

    dataset_name = plan["dataset"]

    if dataset_name not in datasets:
        return False

    data = datasets[
        dataset_name
    ].copy()

    # Same fundamental data-cleaning policy
    data = data.drop_duplicates()

    # Apply filters independently
    try:
        data = apply_filters(
            data,
            plan.get("filters", [])
        )
    except Exception:
        return False

    operation = plan.get(
        "operation"
    )

    metric = plan.get(
        "metric"
    )

    group_by = plan.get(
        "group_by",
        []
    )

    try:

        # --------------------------------------------
        # COUNT
        # --------------------------------------------

        if operation == "count":

            expected = len(data)

        # --------------------------------------------
        # SIMPLE COLUMN OPERATIONS
        # --------------------------------------------

        elif (
            metric in data.columns
            and not group_by
        ):

            numeric = pd.to_numeric(
                data[metric],
                errors="coerce"
            ).dropna()

            if operation == "sum":
                expected = numeric.sum()

            elif operation == "average":
                expected = numeric.mean()

            elif operation == "min":
                expected = numeric.min()

            elif operation == "max":
                expected = numeric.max()

            else:
                return False

        # --------------------------------------------
        # GROUPED OPERATIONS
        # --------------------------------------------

        elif (
            metric in data.columns
            and group_by
        ):

            data = data.dropna(
                subset=[metric]
            )

            expected = (
                data
                .groupby(group_by)[metric]
                .sum()
            )

            if operation == "ranking":

                ascending = (
                    plan.get("direction")
                    != "descending"
                )

                expected = expected.sort_values(
                    ascending=ascending
                )

                limit = plan.get("limit")

                if limit:
                    expected = expected.head(
                        int(limit)
                    )

        else:

            # Complex expressions are verified
            # independently through a fresh
            # expression evaluation.
            if not metric:
                return False

            values = evaluate_expression(
                data,
                metric
            )

            values = pd.to_numeric(
                values,
                errors="coerce"
            ).dropna()

            if operation == "sum":
                expected = values.sum()

            elif operation == "average":
                expected = values.mean()

            elif operation == "min":
                expected = values.min()

            elif operation == "max":
                expected = values.max()

            elif operation == "arithmetic":

                expected = values

            else:
                return False

        return compare_results(
            result,
            expected
        )

    except Exception:
        return False


# ============================================================
# RESULT COMPARISON
# ============================================================

def compare_results(actual, expected):

    actual = normalize_result(
        actual
    )

    expected = normalize_result(
        expected
    )

    if isinstance(
        actual,
        dict
    ) and isinstance(
        expected,
        dict
    ):

        if actual.keys() != expected.keys():
            return False

        for key in actual:

            a = actual[key]
            e = expected[key]

            if isinstance(a, (int, float)) and \
               isinstance(e, (int, float)):

                if not math.isclose(
                    float(a),
                    float(e),
                    rel_tol=1e-9,
                    abs_tol=1e-9
                ):
                    return False

            elif a != e:
                return False

        return True

    if isinstance(
        actual,
        (int, float)
    ) and isinstance(
        expected,
        (int, float)
    ):

        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=1e-9,
            abs_tol=1e-9
        )

    return actual == expected


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze(question):

    print("\n" + "=" * 65)
    print("USER QUESTION")
    print("=" * 65)
    print(question)

    # --------------------------------------------------------
    # STEP 1: LLM PLAN
    # --------------------------------------------------------

    plan = create_plan(
        question
    )

    print("\n" + "=" * 65)
    print("DYNAMIC ANALYSIS PLAN")
    print("=" * 65)

    print(
        json.dumps(
            plan,
            indent=2
        )
    )

    # --------------------------------------------------------
    # STEP 2: REFUSAL
    # --------------------------------------------------------

    if plan.get("status") != "ok":

        print("\n" + "=" * 65)
        print("RESULT: REFUSED")
        print("=" * 65)

        print(
            plan.get(
                "reason",
                "The question cannot be answered reliably."
            )
        )

        return

    # --------------------------------------------------------
    # STEP 3: VALIDATE PLAN
    # --------------------------------------------------------

    valid, reason = validate_plan(
        plan
    )

    if not valid:

        print("\n" + "=" * 65)
        print("RESULT: REFUSED")
        print("=" * 65)

        print(reason)

        return

    # --------------------------------------------------------
    # STEP 4: GENERATE CODE
    # --------------------------------------------------------

    try:

        code = generate_code(
            plan
        )

    except Exception as e:

        print("\nRESULT: REFUSED")
        print(
            f"Could not create safe analysis: {e}"
        )

        return

    print("\n" + "=" * 65)
    print("GENERATED PYTHON CODE")
    print("=" * 65)

    print(code)

    # --------------------------------------------------------
    # STEP 5: EXECUTE
    # --------------------------------------------------------

    result, error = execute_code(
        code
    )

    if error:

        print("\n" + "=" * 65)
        print("RESULT: EXECUTION FAILED")
        print("=" * 65)

        print(error)

        return

    result = normalize_result(
        result
    )

    # --------------------------------------------------------
    # STEP 6: INDEPENDENT VERIFICATION
    # --------------------------------------------------------

    verified = verify_result(
        plan,
        result
    )

    print("\n" + "=" * 65)

    if verified:

        print("VERIFICATION: PASSED")

    else:

        print("VERIFICATION: FAILED")

    print("=" * 65)

    if not verified:

        print(
            "\nThe generated result could not be "
            "independently verified."
        )

        return

    # --------------------------------------------------------
    # STEP 7: FINAL ANSWER
    # --------------------------------------------------------

    print("\nANSWER:")
    print(result)

    print("\nEVIDENCE:")
    print(
        f"Dataset       : {plan.get('dataset')}"
    )

    print(
        f"Operation     : {plan.get('operation')}"
    )

    print(
        f"Metric        : {plan.get('metric')}"
    )

    print(
        f"Grouping      : {plan.get('group_by')}"
    )

    print(
        f"Filters       : {plan.get('filters')}"
    )

    print(
        "Generated code: Executed successfully"
    )

    print(
        "Independent verification: PASSED"
    )


# ============================================================
# TEST MODE
# ============================================================

def test_mode():

    questions = [

        "How much money did Rice make?",

        "Which product sold the most?",

        "Which product sold the most units?",

        "What is the average sales of Apple?",

        "How many units were sold in total?",

        "Which category generated the most money?",

        "What is the difference between total sales and total units?",

        "What is the average amount earned per unit?",

        "What is the best product?",

        "What is the total sales?"
    ]

    for question in questions:

        analyze(question)

        print("\n")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("       DYNAMIC PROOF-CARRYING DATA ANALYST")
    print("=" * 65)

    print("\nAvailable datasets:")

    for name, df in get_datasets().items():

        print(
            f"  {name}: "
            f"{len(df)} rows × "
            f"{len(df.columns)} columns"
        )

    print("\nCommands:")
    print("  test  -> run demonstration questions")
    print("  exit  -> quit")

    while True:

        question = input(
            "\nAsk a data question: "
        ).strip()

        if not question:
            continue

        if question.lower() == "exit":
            break

        if question.lower() == "test":
            test_mode()
            continue

        analyze(question)


if __name__ == "__main__":
    main()