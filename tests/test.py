import json
import os
from datetime import datetime

# ---------------------------------------------------
# CHANGE THESE IMPORTS BASED ON YOUR PROJECT FILES
# ---------------------------------------------------
# Example:
# from generate_response import generate_response
# from rag_pipeline import generate_rag_response

# Replace these two imports with your actual functions
from generate_response import generate_llm_response, generate_rag_response


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
TEST_FILE = "test_queries.json"
OUTPUT_FILE = "test_results.json"


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def load_test_cases(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("test_queries.json must contain a list of test cases.")

    return data


def safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, str(e)


def print_separator():
    print("\n" + "=" * 100 + "\n")


# ---------------------------------------------------
# MAIN TEST LOGIC
# ---------------------------------------------------
def run_tests():
    test_cases = load_test_cases(TEST_FILE)
    all_results = []

    print(f"Loaded {len(test_cases)} test cases from {TEST_FILE}\n")

    for idx, test_case in enumerate(test_cases, start=1):
        query = test_case.get("query", "").strip()
        expected = test_case.get("expected", "")

        if not query:
            print(f"Skipping test case #{idx} because query is empty.")
            continue

        print_separator()
        print(f"TEST CASE #{idx}")
        print(f"USER QUERY:\n{query}\n")

        # -----------------------------
        # Normal LLM Output
        # -----------------------------
        llm_output, llm_error = safe_call(generate_llm_response, query)

        # -----------------------------
        # RAG Output
        # -----------------------------
        rag_output, rag_error = safe_call(generate_rag_response, query)

        print("-" * 100)
        print("NORMAL LLM OUTPUT:\n")
        if llm_error:
            print(f"[ERROR] {llm_error}")
        else:
            print(llm_output)

        print("\n" + "-" * 100)
        print("RAG OUTPUT:\n")
        if rag_error:
            print(f"[ERROR] {rag_error}")
        else:
            print(rag_output)

        if expected:
            print("\n" + "-" * 100)
            print("EXPECTED / REFERENCE:\n")
            print(expected)

        result = {
            "test_id": idx,
            "query": query,
            "expected": expected,
            "llm_output": llm_output,
            "llm_error": llm_error,
            "rag_output": rag_output,
            "rag_error": rag_error,
        }

        all_results.append(result)

    # Save results
    final_payload = {
        "generated_at": datetime.now().isoformat(),
        "total_cases": len(all_results),
        "results": all_results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    print_separator()
    print(f"Testing completed. Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_tests()