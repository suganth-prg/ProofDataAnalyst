import json
import re
import ollama


class LLMParser:

    def __init__(self):
        self.model = "llama3.2:3b"

    def _clean_json(self, text):
        """Clean common formatting mistakes from the local LLM."""

        text = text.strip()

        # Remove markdown code fences
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```\s*", "", text)

        # Find the JSON object if the model added extra text
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:
            text = text[start:end + 1]

        return text.strip()

    def _validate(self, result, question):
        """Validate and normalize the LLM output."""

        allowed_intents = {
            "aggregate",
            "ranking",
            "count",
            "list",
            "ambiguous",
            "unknown"
        }

        allowed_datasets = {
            "sales",
            "customers",
            "inventory",
            "unknown"
        }

        allowed_metrics = {
            "Sales",
            "Units",
            "Customers",
            "Region",
            "unknown"
        }

        allowed_operations = {
            "sum",
            "average",
            "count",
            "maximum",
            "minimum",
            "list",
            "unknown"
        }

        # Make sure all fields exist
        result.setdefault("intent", "unknown")
        result.setdefault("dataset", "unknown")
        result.setdefault("metric", "unknown")
        result.setdefault("operation", "unknown")
        result.setdefault("product", None)
        result.setdefault("ambiguous", False)
        result.setdefault("reason", "")

        # Normalize values
        if result["intent"] not in allowed_intents:
            result["intent"] = "unknown"

        if result["dataset"] not in allowed_datasets:
            result["dataset"] = "unknown"

        if result["metric"] not in allowed_metrics:
            result["metric"] = "unknown"

        if result["operation"] not in allowed_operations:
            result["operation"] = "unknown"

        # Convert string "null" to actual None
        if str(result["product"]).lower() in {
            "null",
            "none",
            "unknown",
            ""
        }:
            result["product"] = None

        # ------------------------------------------
        # Deterministic corrections
        # ------------------------------------------

        q = question.lower()

        # Ranking questions
        if (
            "which product" in q
            and any(word in q for word in [
                "most",
                "highest",
                "top",
                "best"
            ])
        ):
            result["intent"] = "ranking"
            result["dataset"] = "sales"

            if "unit" in q:
                result["metric"] = "Units"
            elif "sales" in q or "money" in q:
                result["metric"] = "Sales"

            result["operation"] = "maximum"

        # "how many units of X"
        elif "how many units" in q:
            result["intent"] = "aggregate"
            result["dataset"] = "sales"
            result["metric"] = "Units"
            result["operation"] = "sum"

        # Sales aggregation
        elif any(word in q for word in [
            "total sales",
            "sales total",
            "money",
            "revenue",
            "made"
        ]):
            result["intent"] = "aggregate"
            result["dataset"] = "sales"
            result["metric"] = "Sales"
            result["operation"] = "sum"

        # Average sales
        elif "average" in q and "sales" in q:
            result["intent"] = "aggregate"
            result["dataset"] = "sales"
            result["metric"] = "Sales"
            result["operation"] = "average"

        # Best product without a metric
        if (
            "best product" in q
            and "sales" not in q
            and "unit" not in q
        ):
            result["intent"] = "ambiguous"
            result["ambiguous"] = True
            result["metric"] = "unknown"
            result["operation"] = "unknown"
            result["product"] = None
            result["reason"] = (
                "The question is ambiguous. "
                "Please specify the metric."
            )

        return result

    def parse_question(self, question):

        prompt = f"""
You are a question-understanding component.

Your ONLY job is to convert the user's question into JSON.
Do NOT calculate any answer.

Return ONLY valid JSON.
No markdown.
No explanation outside JSON.

Schema:

{{
    "intent": "aggregate | ranking | count | list | ambiguous | unknown",
    "dataset": "sales | customers | inventory | unknown",
    "metric": "Sales | Units | Customers | Region | unknown",
    "operation": "sum | average | count | maximum | minimum | list | unknown",
    "product": "product name or null",
    "ambiguous": true,
    "reason": "short explanation"
}}

Examples:

Question:
How much money did Rice make?

JSON:
{{
    "intent": "aggregate",
    "dataset": "sales",
    "metric": "Sales",
    "operation": "sum",
    "product": "Rice",
    "ambiguous": false,
    "reason": "The question asks for total sales of Rice."
}}

Question:
How many units of Rice were sold?

JSON:
{{
    "intent": "aggregate",
    "dataset": "sales",
    "metric": "Units",
    "operation": "sum",
    "product": "Rice",
    "ambiguous": false,
    "reason": "The question asks for total units of Rice."
}}

Question:
Which product sold the most?

JSON:
{{
    "intent": "ranking",
    "dataset": "sales",
    "metric": "Sales",
    "operation": "maximum",
    "product": null,
    "ambiguous": false,
    "reason": "The question asks which product has the highest sales."
}}

Question:
What is the best product?

JSON:
{{
    "intent": "ambiguous",
    "dataset": "sales",
    "metric": "unknown",
    "operation": "unknown",
    "product": null,
    "ambiguous": true,
    "reason": "The metric for best is not specified."
}}

User question:
{question}
"""

        # Try twice in case the small model produces malformed JSON
        for attempt in range(2):

            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            raw = response["message"]["content"]
            cleaned = self._clean_json(raw)

            try:
                result = json.loads(cleaned)

                return self._validate(result, question)

            except json.JSONDecodeError:

                if attempt == 1:
                    return {
                        "intent": "unknown",
                        "dataset": "unknown",
                        "metric": "unknown",
                        "operation": "unknown",
                        "product": None,
                        "ambiguous": True,
                        "reason": "LLM returned invalid JSON."
                    }

        return None


def test_parser():

    parser = LLMParser()

    questions = [
        "How much money did Rice make?",
        "Which product sold the most?",
        "How many units of Rice were sold?",
        "What is the average sales of Apple?",
        "What is the best product?"
    ]

    print("\n")
    print("=" * 60)
    print("        HARDENED LLM PARSER TEST")
    print("=" * 60)

    for question in questions:

        print("\nQuestion:", question)

        try:

            result = parser.parse_question(question)

            print(json.dumps(
                result,
                indent=4
            ))

        except Exception as e:

            print("ERROR:", e)


if __name__ == "__main__":
    test_parser()