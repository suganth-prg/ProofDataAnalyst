# ============================================================
# LLM + PROOF-CARRYING DATA ANALYST
# ============================================================

import json

import analyzer
from llm_parser import LLMParser


class LLMAnalyzer:

    def __init__(self):
        self.parser = LLMParser()

    # --------------------------------------------------------
    # Parse the natural-language question
    # --------------------------------------------------------

    def understand_question(self, question):
        return self.parser.parse_question(question)

    # --------------------------------------------------------
    # Convert LLM output into a readable explanation
    # --------------------------------------------------------

    def show_intent(self, intent):
        print("\n" + "=" * 60)
        print("LLM QUESTION UNDERSTANDING")
        print("=" * 60)

        print("Intent    :", intent.get("intent"))
        print("Dataset   :", intent.get("dataset"))
        print("Metric    :", intent.get("metric"))
        print("Operation :", intent.get("operation"))
        print("Product   :", intent.get("product"))
        print("Ambiguous :", intent.get("ambiguous"))
        print("Reason    :", intent.get("reason"))

    # --------------------------------------------------------
    # Run the trusted proof engine
    # --------------------------------------------------------

    def analyze(self, question):

        # STEP 1:
        # LLM understands the natural-language question.
        intent = self.understand_question(question)

        self.show_intent(intent)

        # ----------------------------------------------------
        # STEP 2:
        # If the LLM detects ambiguity, DO NOT calculate.
        # ----------------------------------------------------

        if intent.get("ambiguous", False):

            return {
                "success": False,
                "answer": None,
                "code": "",
                "evidence": "",
                "warning": intent.get(
                    "reason",
                    "The question is ambiguous."
                ),
                "intent": intent
            }

        # ----------------------------------------------------
        # STEP 3:
        # Unknown intent is rejected.
        # ----------------------------------------------------

        if intent.get("intent") in {
            "unknown",
            None
        }:

            return {
                "success": False,
                "answer": None,
                "code": "",
                "evidence": "",
                "warning": (
                    "The question could not be understood "
                    "reliably."
                ),
                "intent": intent
            }

        # ----------------------------------------------------
        # STEP 4:
        # IMPORTANT:
        #
        # The LLM does NOT calculate anything.
        #
        # analyzer.py remains the authority.
        # ----------------------------------------------------

        result = analyzer.generate_analysis(question)

        # Attach the LLM interpretation for evidence.
        result["intent"] = intent

        return result

    # --------------------------------------------------------
    # Complete proof pipeline
    # --------------------------------------------------------

    def run(self, question):

        print("\n")
        print("=" * 60)
        print("       PROOF-CARRYING DATA ANALYST")
        print("=" * 60)

        print("\nUser Question:")
        print(question)

        # ----------------------------------------------------
        # LLM UNDERSTANDING
        # ----------------------------------------------------

        result = self.analyze(question)

        # ----------------------------------------------------
        # REFUSAL / AMBIGUITY
        # ----------------------------------------------------

        if not result["success"]:

            print("\n" + "=" * 60)
            print("RESULT: REFUSED / NEEDS CLARIFICATION")
            print("=" * 60)

            print("\nReason:")
            print(result["warning"])

            return result

        # ----------------------------------------------------
        # EXECUTE GENERATED CODE
        # ----------------------------------------------------

        print("\n")
        print("=" * 60)
        print("EXECUTING GENERATED PYTHON")
        print("=" * 60)

        executed, execution_result = (
            analyzer.execute_generated_code(
                result["code"]
            )
        )

        if not executed:

            print("\nVERIFICATION FAILED")
            print("\nCode execution error:")
            print(execution_result)

            return {
                "success": False,
                "answer": None,
                "code": result["code"],
                "evidence": "",
                "warning": (
                    "Generated code could not be executed."
                ),
                "intent": result["intent"]
            }

        # ----------------------------------------------------
        # INDEPENDENT VERIFICATION
        # ----------------------------------------------------

        print("\n")
        print("=" * 60)
        print("INDEPENDENT VERIFICATION")
        print("=" * 60)

        verified = analyzer.verify(
            question,
            result
        )

        # Generated code must reproduce the answer.
        code_matches = (
            execution_result == result["answer"]
        )

        # ----------------------------------------------------
        # FINAL TRUST DECISION
        # ----------------------------------------------------

        if not verified or not code_matches:

            print("\nVERIFICATION FAILED")
            print("\nThe system will NOT trust this answer.")

            return {
                "success": False,
                "answer": None,
                "code": result["code"],
                "evidence": "",
                "warning": (
                    "The generated result failed "
                    "independent verification."
                ),
                "intent": result["intent"]
            }

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        print("\n")
        print("=" * 60)
        print("VERIFICATION: PASSED")
        print("=" * 60)

        print("\nAnswer:")
        print(result["answer"])

        print("\nRunnable Python Code:")
        print("-" * 60)
        print(result["code"])

        print("\nEvidence:")
        print("-" * 60)
        print(
            analyzer.generate_evidence(
                question,
                result
            )
        )

        if result.get("warning"):
            print("\nNotes:")
            print(result["warning"])

        return {
            "success": True,
            "answer": result["answer"],
            "code": result["code"],
            "evidence": analyzer.generate_evidence(
                question,
                result
            ),
            "warning": result.get("warning", ""),
            "intent": result["intent"]
        }


# ============================================================
# INTERACTIVE MODE
# ============================================================

def run():

    system = LLMAnalyzer()

    print("\n")
    print("=" * 60)
    print("       LLM PROOF-CARRYING DATA ANALYST")
    print("=" * 60)

    print("\nCommands:")
    print("  test  - run sample questions")
    print("  exit  - quit")

    while True:

        question = input(
            "\nAsk a data question: "
        ).strip()

        if not question:
            continue

        if question.lower() == "exit":
            print("Goodbye.")
            break

        if question.lower() == "test":

            questions = [
                "How much money did Rice make?",
                "Which product sold the most?",
                "How many units of Rice were sold?",
                "What is the average sales of Apple?",
                "What is the best product?",
                "What is the total sales?",
                "What is the total sales ignoring missing values?",
                "How many customers are there?"
            ]

            for q in questions:

                print("\n\n")
                system.run(q)

            continue

        system.run(question)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run()