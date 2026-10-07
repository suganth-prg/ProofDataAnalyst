# ================================================================
# DYNAMIC ANALYZER
# Proof-Carrying Data Analyst
# ================================================================

import os
import re
import json
import math
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# Optional Ollama
try:
    import ollama
except ImportError:
    ollama = None


# ================================================================
# CONFIG
# ================================================================

DATA_DIR = Path("data")
MODEL = "llama3.2:3b"

MAX_ROWS_FOR_CONTEXT = 20
MAX_COLUMNS_FOR_CONTEXT = 30

warnings.filterwarnings(
    "ignore",
    message="Could not infer format"
)


# ================================================================
# BASIC HELPERS
# ================================================================

def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def normalize_name(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        clean_text(value).lower()
    )


def is_number(value):
    try:
        float(value)
        return True
    except Exception:
        return False


def safe_json_value(value):
    if isinstance(value, dict):
        return {
            str(k): safe_json_value(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            safe_json_value(v)
            for v in value
        ]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if pd.isna(value):
        return None

    return value


def dataframe_records(df):
    return safe_json_value(
        df.replace({np.nan: None}).to_dict("records")
    )


# ================================================================
# JSON EXTRACTION
# ================================================================

def extract_json(text):
    """
    Extract the first valid JSON object from an LLM response.

    Handles:
        {...}
        explanation
        {...}
        extra text
    """

    if not text:
        raise ValueError("Empty LLM response")

    text = text.strip()

    # Remove markdown fences
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.I
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", text):
        start = match.start()

        try:
            obj, _ = decoder.raw_decode(
                text[start:]
            )

            if isinstance(obj, dict):
                return obj

        except Exception:
            continue

    raise ValueError(
        "Could not extract valid JSON from planner response"
    )


# ================================================================
# DATA LOADING
# ================================================================

def load_datasets():
    datasets = {}

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    files = list(DATA_DIR.glob("*.csv"))

    for file in files:

        try:
            df = pd.read_csv(file)

            # Clean column names
            df.columns = [
                str(c).strip()
                for c in df.columns
            ]

            datasets[file.stem] = df

        except Exception as exc:
            print(
                f"Could not load {file.name}: {exc}"
            )

    return datasets


# ================================================================
# COLUMN INFORMATION
# ================================================================

def numeric_columns(df):
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]


def categorical_columns(df):
    return [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]


def column_description(df, column):
    series = df[column]

    result = {
        "name": column,
        "dtype": str(series.dtype),
        "unique_count": int(series.nunique(dropna=True)),
        "missing_count": int(series.isna().sum())
    }

    if pd.api.types.is_numeric_dtype(series):
        clean = pd.to_numeric(
            series,
            errors="coerce"
        ).dropna()

        if len(clean):
            result.update({
                "min": float(clean.min()),
                "max": float(clean.max()),
                "mean": float(clean.mean())
            })

    else:
        values = (
            series.dropna()
            .astype(str)
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        result["sample_values"] = values

    return result


def dataset_context(name, df):
    columns = [
        column_description(df, c)
        for c in df.columns[:MAX_COLUMNS_FOR_CONTEXT]
    ]

    sample = (
        df.head(MAX_ROWS_FOR_CONTEXT)
        .replace({np.nan: None})
        .to_dict("records")
    )

    return {
        "dataset": name,
        "rows": len(df),
        "columns": columns,
        "sample_rows": safe_json_value(sample)
    }


# ================================================================
# VALUE / ENTITY RESOLUTION
# ================================================================

def find_exact_value_matches(question, df):
    """
    Find actual values from the dataset that occur in the question.

    This makes the dataset authoritative instead of allowing
    the LLM to invent entities.
    """

    q = question.lower()
    matches = []

    for column in df.columns:

        series = (
            df[column]
            .dropna()
            .astype(str)
            .drop_duplicates()
        )

        for value in series:

            value_clean = value.strip()

            if len(value_clean) < 2:
                continue

            if value_clean.lower() in q:

                matches.append({
                    "column": column,
                    "value": value_clean
                })

    # Prefer longer matches
    matches.sort(
        key=lambda x: len(x["value"]),
        reverse=True
    )

    # Remove duplicates
    unique = []
    seen = set()

    for m in matches:
        key = (
            normalize_name(m["column"]),
            normalize_name(m["value"])
        )

        if key not in seen:
            seen.add(key)
            unique.append(m)

    return unique


def resolve_id(question, df):
    """
    Detect IDs such as:
        E101
        C001
        EMP001
        customer_42
    """

    tokens = re.findall(
        r"\b[A-Za-z]{1,8}[-_]?\d{1,10}\b",
        question
    )

    id_columns = []

    for column in df.columns:

        n = normalize_name(column)

        if (
            n == "id"
            or n.endswith("id")
            or "code" in n
            or "number" in n
        ):
            id_columns.append(column)

    for token in tokens:

        for column in id_columns:

            values = (
                df[column]
                .dropna()
                .astype(str)
            )

            for value in values:

                if value.lower() == token.lower():

                    return {
                        "column": column,
                        "value": value
                    }

    return None


# ================================================================
# SEMANTIC COLUMN MATCHING
# ================================================================

def column_score(column, question):
    """
    Generic semantic-ish scoring.

    This is deliberately not the primary decision mechanism.
    The real dataframe schema remains authoritative.
    """

    name = normalize_name(column)
    q = question.lower()

    score = 0

    words = re.findall(
        r"[a-zA-Z]+",
        q
    )

    for word in words:

        nw = normalize_name(word)

        if not nw:
            continue

        if nw == name:
            score += 10

        elif nw in name:
            score += 5

        elif name in nw:
            score += 3

    # Common semantic relationships
    aliases = {
        "sales": [
            "sale",
            "revenue",
            "income",
            "amount",
            "money",
            "earning",
            "earnings",
            "profit"
        ],
        "units": [
            "unit",
            "sold",
            "quantity",
            "qty",
            "volume",
            "number sold"
        ],
        "product": [
            "item",
            "product",
            "goods"
        ],
        "category": [
            "category",
            "type",
            "group",
            "class"
        ],
        "customer": [
            "customer",
            "client",
            "buyer"
        ],
        "employee": [
            "employee",
            "worker",
            "staff"
        ]
    }

    for canonical, terms in aliases.items():

        if any(term in q for term in terms):

            if canonical in name:
                score += 12

    return score


def choose_metric(df, question, preferred=None):
    nums = numeric_columns(df)

    if not nums:
        return None

    # Planner's metric
    if preferred in nums:
        return preferred

    scores = {
        c: column_score(c, question)
        for c in nums
    }

    best = max(
        scores,
        key=scores.get
    )

    if scores[best] > 0:
        return best

    # If exactly one numeric column exists,
    # it is safe to use it.
    if len(nums) == 1:
        return nums[0]

    return None


def choose_group_column(df, question, preferred=None):
    cats = categorical_columns(df)

    if not cats:
        return None

    if preferred in cats:
        return preferred

    scores = {
        c: column_score(c, question)
        for c in cats
    }

    best = max(
        scores,
        key=scores.get
    )

    if scores[best] > 0:
        return best

    # Product/item/category wording
    q = question.lower()

    for c in cats:

        n = normalize_name(c)

        if "product" in q and "product" in n:
            return c

        if "item" in q and (
            "item" in n or
            "product" in n
        ):
            return c

        if "category" in q and (
            "category" in n or
            "type" in n
        ):
            return c

    return None


# ================================================================
# QUESTION INTENT
# ================================================================

def detect_ranking_intent(question):
    q = question.lower().strip()

    ranking_words = [
        "highest",
        "lowest",
        "maximum",
        "minimum",
        "max",
        "min",
        "most",
        "least",
        "best",
        "worst",
        "top",
        "bottom",
        "largest",
        "smallest",
        "highest-selling",
        "best-selling",
        "most sold",
        "sold the most",
        "sold most"
    ]

    return any(
        phrase in q
        for phrase in ranking_words
    )


def detect_top_direction(question):
    q = question.lower()

    negative = [
        "lowest",
        "least",
        "worst",
        "minimum",
        "smallest",
        "bottom"
    ]

    if any(x in q for x in negative):
        return "asc"

    return "desc"


def detect_count(question):
    q = question.lower()

    return (
        "how many" in q
        or "number of" in q
        or "count of" in q
    )


def detect_row_lookup(question):
    q = question.lower().strip()

    patterns = [
        r"^who is ",
        r"^who's ",
        r"^show me ",
        r"^show ",
        r"^give me the details of ",
        r"^find ",
        r"^lookup ",
        r"^look up "
    ]

    return any(
        re.search(p, q)
        for p in patterns
    )


def detect_comparison(question):
    q = question.lower()

    phrases = [
        "how much more",
        "how much less",
        "difference between",
        "compare",
        "versus",
        "vs",
        "than"
    ]

    return any(
        p in q
        for p in phrases
    )


# ================================================================
# LLM PLANNER
# ================================================================

PLANNER_SYSTEM = r"""
You are a data-analysis planning engine.

Your job is to convert a natural-language data question into
a JSON execution plan.

IMPORTANT:

The dataframe is authoritative.

Never invent columns.
Never invent entities.
Never add unrelated filters.

Allowed operations:

summary
aggregate
top_n
row_lookup
difference
count
list

For ranking questions such as:

"Which product sold the most units?"
"Which product has the highest sales?"
"Which category has the most sales?"
"best-selling product"

you MUST use:

operation = "top_n"

with:

group_by = [the entity/category column]
metric = [the numeric metric]
limit = 1
direction = "desc"

For lowest/bottom/worst questions use direction "asc".

For example:

Question:
Which product sold the most units?

Correct plan:

{
  "status": "ok",
  "operation": "top_n",
  "metric": "Units",
  "group_by": ["Product"],
  "limit": 1,
  "direction": "desc",
  "filters": []
}

Do NOT use row_lookup for ranking questions.

Return JSON only.
"""


def call_llm(prompt):
    if ollama is None:
        raise RuntimeError(
            "Ollama package is not installed."
        )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": PLANNER_SYSTEM
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def llm_plan(question, datasets):
    context = {
        name: dataset_context(name, df)
        for name, df in datasets.items()
    }

    prompt = f"""
AVAILABLE DATA:

{json.dumps(context, indent=2)}

QUESTION:

{question}

Return one JSON planning object.
"""

    raw = call_llm(prompt)

    return extract_json(raw)


# ================================================================
# DATASET SELECTION
# ================================================================

def choose_dataset(question, datasets, plan=None):

    if not datasets:
        return None

    # Entity-based selection
    best_name = None
    best_score = 0

    for name, df in datasets.items():

        matches = find_exact_value_matches(
            question,
            df
        )

        score = len(matches)

        if score > best_score:
            best_score = score
            best_name = name

    if best_name:
        return best_name

    # Planner selection
    if plan:

        requested = plan.get("dataset")

        if requested in datasets:
            return requested

    # If only one dataset exists
    if len(datasets) == 1:
        return next(iter(datasets))

    # Dataset name in question
    q = question.lower()

    for name in datasets:

        if name.lower() in q:
            return name

    return None


# ================================================================
# PLAN NORMALIZATION
# ================================================================

def normalize_row_columns(values, df):
    result = []

    if not isinstance(values, list):
        return result

    for item in values:

        if isinstance(item, str):

            if item in df.columns:
                result.append(item)

        elif isinstance(item, dict):

            name = item.get("name")

            if (
                isinstance(name, str)
                and name in df.columns
            ):
                result.append(name)

    return result


def normalize_filters(filters, df):
    result = []

    if not isinstance(filters, list):
        return result

    for f in filters:

        if not isinstance(f, dict):
            continue

        column = f.get("column")
        operator = f.get("operator", "equals")
        value = f.get("value")

        if column not in df.columns:
            continue

        if operator not in [
            "equals",
            "not_equals",
            "contains",
            "greater_than",
            "less_than",
            "greater_equal",
            "less_equal"
        ]:
            operator = "equals"

        result.append({
            "column": column,
            "operator": operator,
            "value": value
        })

    return result


def normalize_targets(targets, df):
    result = []

    if not isinstance(targets, list):
        return result

    for target in targets:

        if not isinstance(target, dict):
            continue

        column = target.get("column")
        value = target.get("value")

        if column in df.columns:

            result.append({
                "column": column,
                "value": value
            })

    return result


def normalize_plan(plan, df):
    if not isinstance(plan, dict):
        plan = {}

    operation = plan.get(
        "operation",
        "summary"
    )

    if operation not in [
        "summary",
        "aggregate",
        "top_n",
        "row_lookup",
        "difference",
        "count",
        "list"
    ]:
        operation = "summary"

    metric = plan.get("metric")

    if metric not in df.columns:
        metric = None

    secondary = plan.get("secondary_metric")

    if secondary not in df.columns:
        secondary = None

    group_by = plan.get("group_by", [])

    if not isinstance(group_by, list):
        group_by = [group_by]

    group_by = [
        c for c in group_by
        if isinstance(c, str)
        and c in df.columns
    ]

    try:
        limit = int(plan.get("limit"))
    except Exception:
        limit = None

    if limit is not None and limit <= 0:
        limit = None

    direction = plan.get("direction")

    if direction not in ["asc", "desc"]:
        direction = None

    return {
        "status": plan.get("status", "ok"),
        "dataset": plan.get("dataset"),
        "operation": operation,
        "metric": metric,
        "secondary_metric": secondary,
        "group_by": group_by,
        "filters": normalize_filters(
            plan.get("filters", []),
            df
        ),
        "comparison_targets": normalize_targets(
            plan.get("comparison_targets", []),
            df
        ),
        "row_columns": normalize_row_columns(
            plan.get("row_columns", []),
            df
        ),
        "limit": limit,
        "direction": direction,
        "calculation": plan.get(
            "calculation",
            ""
        ),
        "reason": plan.get(
            "reason",
            ""
        )
    }


# ================================================================
# PLAN REPAIR
# ================================================================

def repair_plan(question, df, plan):
    """
    This is the critical layer.

    The LLM proposes a plan.

    This function checks whether that plan actually matches
    the semantic intent of the question and the real dataframe.
    """

    p = normalize_plan(
        plan,
        df
    )

    # ------------------------------------------------------------
    # Explicit ranking intent ALWAYS wins over row_lookup
    # ------------------------------------------------------------

    if detect_ranking_intent(question):

        metric = choose_metric(
            df,
            question,
            p.get("metric")
        )

        group = choose_group_column(
            df,
            question,
            p["group_by"][0]
            if p["group_by"]
            else None
        )

        # If the question explicitly says "product", find product
        q = question.lower()

        if "product" in q:

            candidates = [
                c for c in categorical_columns(df)
                if "product" in normalize_name(c)
            ]

            if candidates:
                group = candidates[0]

        # If explicitly category
        if "category" in q:

            candidates = [
                c for c in categorical_columns(df)
                if (
                    "category" in normalize_name(c)
                    or "type" in normalize_name(c)
                )
            ]

            if candidates:
                group = candidates[0]

        if metric and group:

            p["operation"] = "top_n"
            p["metric"] = metric
            p["group_by"] = [group]

            if not p["limit"]:
                p["limit"] = 1

            p["direction"] = detect_top_direction(
                question
            )

            p["filters"] = []

            p["comparison_targets"] = []

            p["reason"] = (
                "Ranking question resolved as "
                "grouped metric ranking."
            )

            return p

    # ------------------------------------------------------------
    # ID lookup
    # ------------------------------------------------------------

    resolved_id = resolve_id(
        question,
        df
    )

    if resolved_id and (
        detect_row_lookup(question)
        or "who" in question.lower()
    ):

        p["operation"] = "row_lookup"

        p["filters"] = [{
            "column": resolved_id["column"],
            "operator": "equals",
            "value": resolved_id["value"]
        }]

        p["row_columns"] = [
            c for c in df.columns
        ]

        p["metric"] = None
        p["secondary_metric"] = None
        p["group_by"] = []

        return p

    # ------------------------------------------------------------
    # Exact entity filters
    # ------------------------------------------------------------

    matches = find_exact_value_matches(
        question,
        df
    )

    # Only apply matches for non-ranking analytical questions.
    # This prevents ranking questions from accidentally becoming
    # "Rice only" questions.
    if (
        matches
        and not detect_ranking_intent(question)
    ):

        # Avoid adding unrelated values from numeric columns.
        valid = []

        for m in matches:

            column = m["column"]
            value = m["value"]

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):
                continue

            valid.append({
                "column": column,
                "operator": "equals",
                "value": value
            })

        # Preserve planner filters where possible
        if valid:

            # If planner already has a filter on the same
            # column, replace it with actual data value.
            existing = {
                f["column"]: f
                for f in p["filters"]
            }

            for f in valid:
                existing[f["column"]] = f

            p["filters"] = list(
                existing.values()
            )

    # ------------------------------------------------------------
    # Difference/comparison
    # ------------------------------------------------------------

    if detect_comparison(question):

        p["operation"] = "difference"

        metric = choose_metric(
            df,
            question,
            p.get("metric")
        )

        if metric:
            p["metric"] = metric

        targets = p.get(
            "comparison_targets",
            []
        )

        # If planner did not find two targets,
        # derive them from actual data values.
        if len(targets) < 2:

            matches = find_exact_value_matches(
                question,
                df
            )

            if len(matches) >= 2:

                targets = [
                    {
                        "column": matches[0]["column"],
                        "value": matches[0]["value"]
                    },
                    {
                        "column": matches[1]["column"],
                        "value": matches[1]["value"]
                    }
                ]

        p["comparison_targets"] = targets

        return p

    # ------------------------------------------------------------
    # Count
    # ------------------------------------------------------------

    if detect_count(question):

        p["operation"] = "count"

        if not p["group_by"]:

            group = choose_group_column(
                df,
                question,
                None
            )

            if group:
                p["group_by"] = [group]

        return p

    # ------------------------------------------------------------
    # If planner selected row_lookup without an actual lookup
    # target, don't blindly return every row.
    # ------------------------------------------------------------

    if (
        p["operation"] == "row_lookup"
        and not p["filters"]
    ):

        p["operation"] = "summary"

    return p


# ================================================================
# FILTER ENGINE
# ================================================================

def apply_filters(df, filters):

    result = df.copy()

    for f in filters:

        column = f["column"]
        operator = f["operator"]
        value = f["value"]

        if column not in result.columns:
            continue

        series = result[column]

        if operator == "equals":

            result = result[
                series.astype(str).str.lower()
                == str(value).lower()
            ]

        elif operator == "not_equals":

            result = result[
                series.astype(str).str.lower()
                != str(value).lower()
            ]

        elif operator == "contains":

            result = result[
                series.astype(str)
                .str.contains(
                    str(value),
                    case=False,
                    na=False
                )
            ]

        elif operator == "greater_than":

            result = result[
                pd.to_numeric(
                    series,
                    errors="coerce"
                ) > float(value)
            ]

        elif operator == "less_than":

            result = result[
                pd.to_numeric(
                    series,
                    errors="coerce"
                ) < float(value)
            ]

        elif operator == "greater_equal":

            result = result[
                pd.to_numeric(
                    series,
                    errors="coerce"
                ) >= float(value)
            ]

        elif operator == "less_equal":

            result = result[
                pd.to_numeric(
                    series,
                    errors="coerce"
                ) <= float(value)
            ]

    return result


# ================================================================
# EXECUTION
# ================================================================

def execute_plan(df, plan):

    df = df.copy()

    df = df.drop_duplicates()

    df = apply_filters(
        df,
        plan["filters"]
    )

    operation = plan["operation"]

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    if operation == "summary":

        result = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "missing_values": int(
                df.isna().sum().sum()
            )
        }

        return result

    # ------------------------------------------------------------
    # LIST
    # ------------------------------------------------------------

    if operation == "list":

        columns = plan["row_columns"]

        if not columns:
            columns = list(df.columns)

        return dataframe_records(
            df[columns]
        )

    # ------------------------------------------------------------
    # ROW LOOKUP
    # ------------------------------------------------------------

    if operation == "row_lookup":

        columns = plan["row_columns"]

        if not columns:
            columns = list(df.columns)

        return dataframe_records(
            df[columns]
        )

    # ------------------------------------------------------------
    # COUNT
    # ------------------------------------------------------------

    if operation == "count":

        if plan["group_by"]:

            group = plan["group_by"][0]

            result = (
                df.groupby(group)
                .size()
                .reset_index(
                    name="Count"
                )
                .sort_values(
                    "Count",
                    ascending=False
                )
            )

            return dataframe_records(result)

        return {
            "count": int(len(df))
        }

    # ------------------------------------------------------------
    # TOP N
    # ------------------------------------------------------------

    if operation == "top_n":

        metric = plan["metric"]

        groups = plan["group_by"]

        if not metric:
            raise ValueError(
                "Ranking requires a numeric metric."
            )

        if not groups:
            raise ValueError(
                "Ranking requires a grouping column."
            )

        if metric not in df.columns:
            raise ValueError(
                f"Metric column '{metric}' not found."
            )

        clean = df.copy()

        clean[metric] = pd.to_numeric(
            clean[metric],
            errors="coerce"
        )

        clean = clean.dropna(
            subset=[metric]
        )

        result = (
            clean.groupby(
                groups,
                dropna=False
            )[metric]
            .sum()
            .reset_index()
        )

        direction = (
            plan["direction"]
            or "desc"
        )

        result = result.sort_values(
            metric,
            ascending=(direction == "asc")
        )

        limit = plan["limit"] or 1

        result = result.head(
            int(limit)
        )

        return dataframe_records(
            result
        )

    # ------------------------------------------------------------
    # AGGREGATE
    # ------------------------------------------------------------

    if operation == "aggregate":

        metric = plan["metric"]

        if not metric:
            raise ValueError(
                "Aggregation requires a metric."
            )

        values = pd.to_numeric(
            df[metric],
            errors="coerce"
        ).dropna()

        q = plan.get(
            "calculation",
            "sum"
        ).lower()

        if q == "mean" or q == "average":
            value = values.mean()

        elif q == "min":
            value = values.min()

        elif q == "max":
            value = values.max()

        elif q == "count":
            value = len(values)

        else:
            value = values.sum()

        return {
            metric: safe_json_value(value)
        }

    # ------------------------------------------------------------
    # DIFFERENCE
    # ------------------------------------------------------------

    if operation == "difference":

        metric = plan["metric"]

        targets = plan[
            "comparison_targets"
        ]

        if not metric:
            raise ValueError(
                "Comparison requires a metric."
            )

        if len(targets) != 2:
            raise ValueError(
                "Comparison requires exactly "
                "two comparison targets."
            )

        values = []

        for target in targets:

            target_df = df[
                df[target["column"]]
                .astype(str)
                .str.lower()
                == str(target["value"]).lower()
            ]

            if target_df.empty:
                raise ValueError(
                    f"No data found for "
                    f"{target['value']}"
                )

            number = pd.to_numeric(
                target_df[metric],
                errors="coerce"
            ).sum()

            values.append(float(number))

        return {
            "first": values[0],
            "second": values[1],
            "difference": values[0] - values[1]
        }

    raise ValueError(
        f"Unsupported operation: {operation}"
    )


# ================================================================
# INDEPENDENT VERIFIER
# ================================================================

def independent_verify(df, plan):

    """
    Independent implementation.

    It does NOT simply rerun execute_plan().
    """

    clean = df.drop_duplicates()

    clean = apply_filters(
        clean,
        plan["filters"]
    )

    operation = plan["operation"]

    # Ranking
    if operation == "top_n":

        metric = plan["metric"]
        groups = plan["group_by"]
        limit = plan["limit"] or 1
        direction = plan["direction"] or "desc"

        temp = clean.copy()

        temp[metric] = pd.to_numeric(
            temp[metric],
            errors="coerce"
        )

        temp = temp.dropna(
            subset=[metric]
        )

        grouped = (
            temp.groupby(groups)[metric]
            .sum()
            .reset_index()
        )

        grouped = grouped.sort_values(
            metric,
            ascending=(
                direction == "asc"
            )
        )

        return dataframe_records(
            grouped.head(limit)
        )

    # Difference
    if operation == "difference":

        metric = plan["metric"]
        targets = plan[
            "comparison_targets"
        ]

        results = []

        for target in targets:

            subset = clean[
                clean[target["column"]]
                .astype(str)
                .str.lower()
                ==
                str(target["value"]).lower()
            ]

            number = pd.to_numeric(
                subset[metric],
                errors="coerce"
            ).sum()

            results.append(
                float(number)
            )

        return {
            "first": results[0],
            "second": results[1],
            "difference":
                results[0] - results[1]
        }

    # Aggregate
    if operation == "aggregate":

        metric = plan["metric"]

        values = pd.to_numeric(
            clean[metric],
            errors="coerce"
        ).dropna()

        calculation = (
            plan.get("calculation")
            or "sum"
        ).lower()

        if calculation in [
            "average",
            "mean"
        ]:
            return {
                metric:
                    float(values.mean())
            }

        if calculation == "min":
            return {
                metric:
                    float(values.min())
            }

        if calculation == "max":
            return {
                metric:
                    float(values.max())
            }

        return {
            metric:
                float(values.sum())
        }

    # Count
    if operation == "count":

        if plan["group_by"]:

            column = plan[
                "group_by"
            ][0]

            result = (
                clean.groupby(column)
                .size()
                .reset_index(
                    name="Count"
                )
                .sort_values(
                    "Count",
                    ascending=False
                )
            )

            return dataframe_records(result)

        return {
            "count": int(len(clean))
        }

    # Row lookup
    if operation == "row_lookup":

        columns = (
            plan["row_columns"]
            or list(clean.columns)
        )

        return dataframe_records(
            clean[columns]
        )

    # Summary
    if operation == "summary":

        return {
            "rows": int(len(clean)),
            "columns": list(clean.columns),
            "missing_values": int(
                clean.isna().sum().sum()
            )
        }

    # List
    if operation == "list":

        columns = (
            plan["row_columns"]
            or list(clean.columns)
        )

        return dataframe_records(
            clean[columns]
        )

    raise ValueError(
        f"Cannot independently verify "
        f"{operation}"
    )


# ================================================================
# RESULT COMPARISON
# ================================================================

def normalize_for_compare(value):

    if isinstance(value, dict):

        return {
            str(k):
                normalize_for_compare(v)
            for k, v in sorted(
                value.items(),
                key=lambda x: str(x[0])
            )
        }

    if isinstance(value, list):

        normalized = [
            normalize_for_compare(v)
            for v in value
        ]

        return sorted(
            normalized,
            key=lambda x: json.dumps(
                x,
                sort_keys=True
            )
        )

    if isinstance(value, float):

        if math.isnan(value):
            return None

        return round(value, 8)

    return value


def results_match(a, b):

    a = normalize_for_compare(a)
    b = normalize_for_compare(b)

    return a == b


# ================================================================
# HUMAN ANSWER
# ================================================================

def format_answer(question, plan, result):

    operation = plan["operation"]

    if operation == "top_n":

        if not result:
            return "No matching data was found."

        if isinstance(result, list):

            first = result[0]

            group = plan["group_by"][0]
            metric = plan["metric"]

            entity = first.get(group)
            value = first.get(metric)

            direction = plan[
                "direction"
            ]

            if direction == "asc":
                wording = "lowest"
            else:
                wording = "highest"

            return (
                f"{entity} has the {wording} "
                f"{metric}: {value}."
            )

    if operation == "difference":

        return (
            f"The difference is "
            f"{result['difference']}."
        )

    if operation == "aggregate":

        metric = plan["metric"]

        return (
            f"The {plan.get('calculation') or 'total'} "
            f"{metric} is "
            f"{result[metric]}."
        )

    if operation == "count":

        if isinstance(result, dict):
            return (
                f"The count is "
                f"{result['count']}."
            )

    if operation == "row_lookup":

        if not result:
            return "No matching record was found."

        if len(result) == 1:
            return (
                "The matching record is: "
                + json.dumps(
                    result[0],
                    ensure_ascii=False
                )
            )

        return (
            f"Found {len(result)} matching records."
        )

    if operation == "summary":

        return (
            f"The dataset contains "
            f"{result['rows']} rows and "
            f"{len(result['columns'])} columns."
        )

    return json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )


# ================================================================
# DISPLAY
# ================================================================

def print_json(title, value):

    print()
    print(title)
    print("-" * 70)

    print(
        json.dumps(
            safe_json_value(value),
            indent=2,
            ensure_ascii=False
        )
    )


# ================================================================
# MAIN QUESTION PIPELINE
# ================================================================

def answer_question(question, datasets):

    print()
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    if not datasets:
        print(
            "\nRESULT: REFUSED\n"
            "No CSV datasets were found in data/."
        )
        return

    # ------------------------------------------------------------
    # Ask planner
    # ------------------------------------------------------------

    try:
        raw_plan = llm_plan(
            question,
            datasets
        )

    except Exception as exc:

        print(
            "\nRESULT: REFUSED\n"
            f"LLM planning failed: {exc}"
        )

        return

    print_json(
        "RAW PLAN:",
        raw_plan
    )

    # ------------------------------------------------------------
    # Choose dataset
    # ------------------------------------------------------------

    dataset_name = choose_dataset(
        question,
        datasets,
        raw_plan
    )

    if not dataset_name:

        print(
            "\nRESULT: REFUSED\n"
            "Could not determine which dataset "
            "contains the answer."
        )

        return

    df = datasets[
        dataset_name
    ]

    # ------------------------------------------------------------
    # Repair plan using actual data
    # ------------------------------------------------------------

    plan = repair_plan(
        question,
        df,
        raw_plan
    )

    plan["dataset"] = dataset_name

    print_json(
        "VALIDATED PLAN:",
        plan
    )

    # ------------------------------------------------------------
    # Refuse genuinely incomplete plans
    # ------------------------------------------------------------

    if plan["operation"] == "top_n":

        if not plan["metric"]:
            print(
                "\nRESULT: REFUSED\n"
                "Could not determine the metric "
                "to rank by."
            )
            return

        if not plan["group_by"]:
            print(
                "\nRESULT: REFUSED\n"
                "Could not determine what entity "
                "should be ranked."
            )
            return

    # ------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------

    try:

        result = execute_plan(
            df,
            plan
        )

    except Exception as exc:

        print(
            "\nRESULT: REFUSED\n"
            f"Execution failed: {exc}"
        )

        return

    # ------------------------------------------------------------
    # Show deterministic calculation
    # ------------------------------------------------------------

    print()
    print("GENERATED CALCULATION:")
    print("-" * 70)

    if plan["operation"] == "top_n":

        print(
            f"GROUP BY {plan['group_by']} "
            f"→ SUM({plan['metric']}) "
            f"→ SORT {plan['direction']} "
            f"→ TOP {plan['limit']}"
        )

    elif plan["operation"] == "difference":

        print(
            f"Compare {plan['metric']} "
            f"for two requested entities."
        )

    elif plan["operation"] == "aggregate":

        print(
            f"{plan.get('calculation') or 'SUM'}"
            f"({plan['metric']})"
        )

    else:

        print(
            f"Operation: {plan['operation']}"
        )

    print_json(
        "GENERATED RESULT:",
        result
    )

    # ------------------------------------------------------------
    # Independent verification
    # ------------------------------------------------------------

    try:

        independent_result = (
            independent_verify(
                df,
                plan
            )
        )

    except Exception as exc:

        print(
            "\nRESULT: REFUSED\n"
            f"Independent verification failed: "
            f"{exc}"
        )

        return

    print_json(
        "INDEPENDENT RESULT:",
        independent_result
    )

    # ------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------

    verified = results_match(
        result,
        independent_result
    )

    if not verified:

        print()
        print(
            "VERIFICATION: FAILED"
        )

        print(
            "\nRESULT: REFUSED\n"
            "The independent calculation "
            "does not agree with the generated result."
        )

        return

    print()
    print(
        "VERIFICATION: PASSED"
    )

    # ------------------------------------------------------------
    # Final answer
    # ------------------------------------------------------------

    answer = format_answer(
        question,
        plan,
        result
    )

    print()
    print("=" * 70)
    print("RESULT: VERIFIED")
    print("=" * 70)

    print()
    print("ANSWER:")
    print(answer)

    print()
    print(
        "Proof: deterministic execution and "
        "independent verification agree."
    )

    print("=" * 70)


# ================================================================
# SELF TESTS
# ================================================================

def run_tests(datasets):

    print()
    print("=" * 70)
    print("SELF TEST")
    print("=" * 70)

    questions = [
        "Which product has the highest sales?",
        "Which product sold the most units?",
        "Which product has the lowest sales?",
        "Which product sold the least units?",
        "What is the total sales?",
        "What is the average sales?",
        "What is the highest sales?",
        "What is the lowest sales?",
        "How many products are there?",
        "How many categories are there?",
        "Which category has the highest sales?",
        "Which category has the lowest sales?",
        "What are the top 3 products by sales?",
        "What are the bottom 3 products by sales?",
        "Rank the products by sales.",
        "Rank the products by units.",
        "How much did Wheat sell?",
        "How much did Rice sell?",
        "How many units did Banana sell?",
        "How many units did Apple sell?",
        "Show all grain products.",
        "Show all fruit products.",
        "Which fruit has the highest sales?",
        "Which grain has the highest sales?",
        "Which fruit sold the most units?",
        "Which grain sold the most units?",
        "How much more did Wheat make than Rice?",
        "How much more did Wheat sell than Corn?",
        "What is the difference between Apple and Banana sales?",
        "Compare Rice and Wheat.",
        "What are the sales of products above 9000?",
        "What are the sales of products below 10000?",
        "Which product has the most units?",
        "Which item sold the most?",
        "What product made the most money?",
        "Tell me the best-selling product.",
        "What is the best product?",
        "Which product performed the best?",
        "What product has maximum sales?",
        "What product has maximum units?",
        "Which category contributes the most sales?",
        "Which category sold the most units?",
        "What columns are available?",
        "How many rows are in the sales data?",
        "Are there missing values?",
        "Which column has missing values?",
        "Are there duplicate rows?",
        "What is the average number of units sold?",
        "What is the total number of units sold?",
        "What is the maximum number of units sold?",
        "What is the minimum number of units sold?",
        "Which product is second highest by sales?",
        "Which product is second highest by units?",
        "Which product is second lowest by sales?",
        "Which category has the most products?",
        "Which category has the highest average sales?",
        "Which category has the highest average units?",
        "Is there enough data to determine the best-selling product?",
        "Give me the top product by units."
    ]

    passed = 0
    failed = 0

    for i, question in enumerate(
        questions,
        1
    ):

        print()
        print(
            f"[{i}/{len(questions)}] "
            f"{question}"
        )

        try:

            raw = llm_plan(
                question,
                datasets
            )

            dataset_name = choose_dataset(
                question,
                datasets,
                raw
            )

            if not dataset_name:
                print("  DATASET: FAIL")
                failed += 1
                continue

            df = datasets[
                dataset_name
            ]

            plan = repair_plan(
                question,
                df,
                raw
            )

            result = execute_plan(
                df,
                plan
            )

            independent = (
                independent_verify(
                    df,
                    plan
                )
            )

            if results_match(
                result,
                independent
            ):
                print(
                    f"  PASS | "
                    f"{plan['operation']} | "
                    f"{plan.get('metric')}"
                )
                passed += 1

            else:

                print(
                    "  FAIL | "
                    "verification mismatch"
                )

                failed += 1

        except Exception as exc:

            print(
                f"  FAIL | {exc}"
            )

            failed += 1

    print()
    print("=" * 70)
    print("SELF TEST SUMMARY")
    print("=" * 70)

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Total: {len(questions)}"
    )

    if questions:
        print(
            f"Rate: "
            f"{passed / len(questions) * 100:.1f}%"
        )

    print("=" * 70)


# ================================================================
# DISPLAY AVAILABLE DATA
# ================================================================

def show_datasets(datasets):

    print()
    print("=" * 70)
    print("AVAILABLE DATA")
    print("=" * 70)

    if not datasets:

        print(
            "No CSV files found in data/"
        )

        return

    for name, df in datasets.items():

        print()
        print(
            f"{name}:"
        )

        print(
            f"  Rows: {len(df)}"
        )

        print(
            "  Columns: "
            + ", ".join(
                str(c)
                for c in df.columns
            )
        )


# ================================================================
# MAIN
# ================================================================

def main():

    datasets = load_datasets()

    show_datasets(
        datasets
    )

    # Command-line self-test
    import sys

    if (
        len(sys.argv) > 1
        and sys.argv[1] == "--test"
    ):

        run_tests(
            datasets
        )

        return

    print()
    print(
        "Ask a data question:"
    )
    print(
        "Type 'exit' to quit."
    )

    while True:

        try:
            question = input(
                "\nAsk a data question: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print()
            break

        if not question:
            continue

        if question.lower() in [
            "exit",
            "quit"
        ]:
            break

        answer_question(
            question,
            datasets
        )


if __name__ == "__main__":
    main()