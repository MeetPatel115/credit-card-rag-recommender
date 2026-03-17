import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


# ============================================================
# PATH SETUP
# ============================================================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent

# Allow imports from project root and scripts/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "scripts"))


# ============================================================
# IMPORT YOUR RAG FUNCTION
# ============================================================
try:
    from scripts.genrate_response import generate_response
except Exception:
    try:
        from genrate_response import generate_response
    except Exception as e:
        raise ImportError(
            "Could not import 'generate_response' from scripts/genrate_response.py.\n"
            "Make sure your project looks like this:\n"
            "credit-card-rag/\n"
            "  scripts/genrate_response.py\n"
            "  tests/test_ollama_vs_rag.py\n\n"
            f"Original import error: {e}"
        )


# ============================================================
# CONFIG
# ============================================================
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
RAG_MODEL = "gemma3:1b"
BASE_MODEL = "gemma3:1b"
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = RESULTS_DIR / "rag_vs_base_results.csv"
JSON_OUTPUT = RESULTS_DIR / "rag_vs_base_results.json"

DEFAULT_USER_PROFILE: Dict[str, Any] = {
    "income": 70000,
    "household_income": None,
    "max_annual_fee": 180,
    "preferred_rewards": "cashback",
    "target_categories": ["grocery", "dining", "gas"],
    "issuer": None,
    "network": None,
}

TEST_CASES: List[Dict[str, Any]] = [
    {
        "query": "Which credit card is best for groceries and dining with a low annual fee in Canada?",
        "expected_keywords": ["grocery", "dining", "annual fee"],
        "user_profile": {
            "income": 70000,
            "household_income": None,
            "max_annual_fee": 120,
            "preferred_rewards": "cashback",
            "target_categories": ["grocery", "dining"],
            "issuer": None,
            "network": None,
        },
    },
    {
        "query": "Suggest a credit card for travel rewards and airport lounge access.",
        "expected_keywords": ["travel", "lounge", "rewards"],
        "user_profile": {
            "income": 100000,
            "household_income": 150000,
            "max_annual_fee": 599,
            "preferred_rewards": "travel",
            "target_categories": ["travel"],
            "issuer": None,
            "network": None,
        },
    },
    {
        "query": "I spend a lot on gas, groceries, and recurring bills. Which card should I choose?",
        "expected_keywords": ["gas", "grocery", "recurring", "card"],
        "user_profile": {
            "income": 80000,
            "household_income": None,
            "max_annual_fee": 180,
            "preferred_rewards": "cashback",
            "target_categories": ["gas", "grocery", "recurring bills"],
            "issuer": None,
            "network": None,
        },
    },
    {
        "query": "What is a good Canadian card with no foreign transaction fee?",
        "expected_keywords": ["foreign", "transaction", "fee"],
        "user_profile": {
            "income": 90000,
            "household_income": None,
            "max_annual_fee": 200,
            "preferred_rewards": "travel",
            "target_categories": ["travel"],
            "issuer": None,
            "network": None,
        },
    },
    {
        "query": "Recommend a card for dining and entertainment rewards.",
        "expected_keywords": ["dining", "entertainment", "rewards"],
        "user_profile": {
            "income": 65000,
            "household_income": None,
            "max_annual_fee": 150,
            "preferred_rewards": "points",
            "target_categories": ["dining", "entertainment"],
            "issuer": None,
            "network": None,
        },
    },
]


# ============================================================
# HELPERS
# ============================================================
def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def keyword_hit_score(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 0.0

    answer_norm = normalize_text(answer)
    hits = 0
    for kw in expected_keywords:
        if normalize_text(kw) in answer_norm:
            hits += 1
    return round(hits / len(expected_keywords), 2)


def safe_get_final_response(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("final_response", "")).strip()
    return str(result).strip()


def ask_base_ollama(prompt: str, model: str = BASE_MODEL) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()
    except Exception as e:
        return f"ERROR: {e}"


def ask_rag(query: str, user_profile: Dict[str, Any], model: str = RAG_MODEL) -> Dict[str, Any]:
    try:
        result = generate_response(
            user_query=query,
            user_profile=user_profile,
            top_n=3,
            save_outputs=False,
            provider="ollama",
            model=model,
        )
        return result
    except Exception as e:
        return {
            "final_response": f"ERROR: {e}",
            "llm_error": str(e),
            "recommended_cards": None,
        }


# ============================================================
# MAIN TEST RUNNER
# ============================================================
def run_comparison_tests() -> None:
    results: List[Dict[str, Any]] = []
    rag_wins = 0
    base_wins = 0
    ties = 0

    print("=" * 100)
    print("RAG vs Normal Ollama Evaluation")
    print("=" * 100)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"RAG model    : {RAG_MODEL}")
    print(f"Base model   : {BASE_MODEL}")
    print(f"Results dir  : {RESULTS_DIR}")
    print()

    for idx, case in enumerate(TEST_CASES, start=1):
        query = case.get("query", "").strip()
        expected_keywords = case.get("expected_keywords", [])
        user_profile = case.get("user_profile") or DEFAULT_USER_PROFILE.copy()

        print("-" * 100)
        print(f"Test {idx}")
        print(f"Query: {query}")
        print(f"Expected keywords: {expected_keywords}")

        # RAG
        rag_start = time.time()
        rag_result = ask_rag(query, user_profile=user_profile, model=RAG_MODEL)
        rag_time = round(time.time() - rag_start, 2)
        rag_response = safe_get_final_response(rag_result)
        rag_score = keyword_hit_score(rag_response, expected_keywords)

        # BASE OLLAMA
        base_start = time.time()
        base_response = ask_base_ollama(query, model=BASE_MODEL)
        base_time = round(time.time() - base_start, 2)
        base_score = keyword_hit_score(base_response, expected_keywords)

        if rag_score > base_score:
            winner = "RAG"
            rag_wins += 1
        elif base_score > rag_score:
            winner = "BASE"
            base_wins += 1
        else:
            winner = "TIE"
            ties += 1

        recommended_cards = rag_result.get("recommended_cards")
        recommended_card_names: List[str] = []
        if recommended_cards is not None:
            try:
                if "card_name" in recommended_cards.columns:
                    recommended_card_names = recommended_cards["card_name"].astype(str).tolist()
            except Exception:
                pass

        row = {
            "test_id": idx,
            "query": query,
            "expected_keywords": ", ".join(expected_keywords),
            "rag_score": rag_score,
            "base_score": base_score,
            "winner": winner,
            "rag_time_sec": rag_time,
            "base_time_sec": base_time,
            "rag_llm_error": rag_result.get("llm_error"),
            "rag_recommended_cards": " | ".join(recommended_card_names),
            "rag_response": rag_response,
            "base_response": base_response,
        }
        results.append(row)

        print(f"RAG score           : {rag_score}")
        print(f"Base score          : {base_score}")
        print(f"Winner              : {winner}")
        print(f"RAG time            : {rag_time}s")
        print(f"Base time           : {base_time}s")
        if recommended_card_names:
            print(f"RAG recommended     : {recommended_card_names}")
        if rag_result.get("llm_error"):
            print(f"RAG llm_error       : {rag_result.get('llm_error')}")
        print()
        print("RAG response preview:")
        print(rag_response[:500] + ("..." if len(rag_response) > 500 else ""))
        print()
        print("Base response preview:")
        print(base_response[:500] + ("..." if len(base_response) > 500 else ""))
        print()

    # Save CSV
    csv_fields = [
        "test_id",
        "query",
        "expected_keywords",
        "rag_score",
        "base_score",
        "winner",
        "rag_time_sec",
        "base_time_sec",
        "rag_llm_error",
        "rag_recommended_cards",
        "rag_response",
        "base_response",
    ]

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(results)

    # Save JSON
    summary = {
        "rag_model": RAG_MODEL,
        "base_model": BASE_MODEL,
        "total_tests": len(TEST_CASES),
        "rag_wins": rag_wins,
        "base_wins": base_wins,
        "ties": ties,
        "csv_output": str(CSV_OUTPUT),
        "json_output": str(JSON_OUTPUT),
        "results": results,
    }

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"Total tests : {len(TEST_CASES)}")
    print(f"RAG wins    : {rag_wins}")
    print(f"Base wins   : {base_wins}")
    print(f"Ties        : {ties}")
    print(f"CSV saved   : {CSV_OUTPUT}")
    print(f"JSON saved  : {JSON_OUTPUT}")


if __name__ == "__main__":
    run_comparison_tests()
