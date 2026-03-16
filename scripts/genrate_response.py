from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from explain_recomendation import explain_recommendation


OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_fallback_response(result: Dict[str, Any], user_query: str) -> str:
    recommended_cards = result.get("recommended_cards")

    if recommended_cards is None or recommended_cards.empty:
        return (
            "I could not find any cards that match your current filters. "
            "Try increasing your maximum annual fee, changing your preferred rewards type, "
            "or removing issuer or network restrictions."
        )

    top_cards = recommended_cards.head(3).copy()

    lines: List[str] = []
    lines.append(f"Query: {user_query}")
    lines.append("")
    lines.append("Here are the top recommended cards based on your preferences:")
    lines.append("")

    for idx, (_, row) in enumerate(top_cards.iterrows(), start=1):
        card_name = str(row.get("card_name", "")).strip()
        issuer = str(row.get("issuer", "")).strip()
        annual_fee = row.get("annual_fee_value")
        rewards_type = str(row.get("rewards_type", "")).strip()
        best_categories = str(row.get("best_categories", "")).strip()
        reason = str(row.get("reason", "")).strip()

        fee_text = "annual fee not available"
        try:
            if annual_fee is not None:
                annual_fee = float(annual_fee)
                fee_text = "no annual fee" if annual_fee == 0 else f"${annual_fee:.2f} annual fee"
        except Exception:
            pass

        lines.append(f"{idx}. {card_name} ({issuer})")
        lines.append(f"   - Rewards type: {rewards_type or 'not available'}")
        lines.append(f"   - Best categories: {best_categories or 'not available'}")
        lines.append(f"   - Fee: {fee_text}")
        lines.append(f"   - Why it fits: {reason or 'good overall match for your filters'}")
        lines.append("")

    best = top_cards.iloc[0]
    lines.append(
        f"Best overall choice: {best.get('card_name', 'This card')} "
        f"because it ranks highest for your current preferences."
    )
    lines.append("")
    lines.append(
        "This summary is based on your structured recommendation results and retrieved card context."
    )

    return "\n".join(lines)


def build_llm_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful Canadian credit card recommendation assistant. "
                "Use only the provided context. "
                "Do not invent benefits, fees, welcome bonuses, or eligibility rules. "
                "Explain why each recommended card fits the user's needs, "
                "mention annual fee tradeoffs clearly, "
                "and identify the best overall option."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def call_ollama_openai_compat(
    messages: List[Dict[str, str]],
    model: str = "gemma3:1b",
    base_url: str = "http://localhost:11434/v1/",
) -> str:
    client = OpenAI(
        api_key="ollama",
        base_url=base_url,
    )

    response = client.responses.create(
        model=model,
        input=messages,
    )

    return response.output_text.strip()


def generate_response(
    user_query: str,
    user_profile: Dict[str, Any],
    top_n: int = 3,
    save_outputs: bool = True,
    provider: str = "ollama",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    explain_result = explain_recommendation(
        user_query=user_query,
        user_profile=user_profile,
        top_n=top_n,
    )

    prompt = explain_result.get("prompt", "")
    fallback_response = build_fallback_response(explain_result, user_query)
    llm_messages = build_llm_messages(prompt)

    final_response = fallback_response
    llm_error = None

    try:
        if provider == "ollama":
            final_response = call_ollama_openai_compat(
                llm_messages,
                model=model or "gemma3:1b",
            )
        else:
            llm_error = f"Unknown provider: {provider}. Using fallback response."
            final_response = fallback_response
    except Exception as e:
        llm_error = str(e)
        final_response = fallback_response

    output = {
        "user_query": user_query,
        "user_profile": user_profile,
        "recommended_cards": explain_result.get("recommended_cards"),
        "card_chunks_map": explain_result.get("card_chunks_map"),
        "prompt": prompt,
        "llm_messages": llm_messages,
        "fallback_response": fallback_response,
        "final_response": final_response,
        "provider": provider,
        "model": model or "gemma3:1b",
        "llm_error": llm_error,
    }

    if save_outputs:
        prompt_path = OUTPUT_DIR / "latest_recommendation_prompt.txt"
        response_path = OUTPUT_DIR / "latest_final_response.txt"

        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        with open(response_path, "w", encoding="utf-8") as f:
            f.write(final_response)

        output["prompt_path"] = str(prompt_path)
        output["response_path"] = str(response_path)

    return output


def print_result_summary(result: Dict[str, Any]) -> None:
    recommended_cards = result.get("recommended_cards")

    print("\n=== RECOMMENDED CARDS ===")
    if recommended_cards is None or recommended_cards.empty:
        print("No cards matched.")
    else:
        cols = [
            "card_name",
            "issuer",
            "annual_fee_value",
            "rewards_type",
            "best_categories",
            "total_score",
            "reason",
        ]
        existing = [c for c in cols if c in recommended_cards.columns]
        print(recommended_cards[existing].to_string(index=False))

    print("\n=== FINAL LLM RESPONSE ===\n")
    print(result.get("final_response", ""))

    if result.get("llm_error"):
        print("\n=== LLM ERROR / FALLBACK USED ===")
        print(result["llm_error"])

    if result.get("prompt_path"):
        print("\nPrompt saved to:", result["prompt_path"])
    if result.get("response_path"):
        print("Response saved to:", result["response_path"])


def main() -> None:
    user_query = "best credit card for groceries and dining with low annual fee"

    user_profile = {
        "income": 70000,
        "household_income": None,
        "max_annual_fee": 180,
        "preferred_rewards": "membership_rewards",
        "target_categories": ["grocery", "dining"],
        "issuer": None,
        "network": None,
    }

    result = generate_response(
        user_query=user_query,
        user_profile=user_profile,
        top_n=3,
        save_outputs=True,
        provider="ollama",
        model="gemma3:1b",
    )

    print_result_summary(result)


if __name__ == "__main__":
    main()