from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

from recommender_card import recommend_cards  # see note below


CSV_PATH = Path(r"C:\Users\91951\OneDrive\Desktop\pythonProject\leetcode\Ai-ML-Projects\credit-card-rag\data\processed\combined_cards_enriched.csv")
DB_DIR = r"C:\Users\91951\OneDrive\Desktop\pythonProject\leetcode\Ai-ML-Projects\credit-card-rag\data\vectordb\chroma_db"
COLLECTION_NAME = "credit_card_chunks"

GOOD_SECTIONS = {
    "overview",
    "rewards",
    "welcome_bonus",
    "benefits",
    "eligibility",
    "insurance",
}


def load_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def get_supporting_chunks_for_card(
    collection,
    card_name: str,
    user_query: str,
    n_results: int = 8,
) -> List[Dict[str, Any]]:
    query_text = f"{user_query}. Card name: {card_name}"

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(docs)

    selected = []
    seen_sections = set()

    for doc, meta, distance in zip(docs, metas, distances):
        if not meta:
            continue

        meta_card_name = str(meta.get("card_name", "")).strip().lower()
        if meta_card_name != card_name.strip().lower():
            continue

        section = str(meta.get("section", "")).strip().lower()
        if section not in GOOD_SECTIONS:
            continue

        if section in seen_sections:
            continue

        seen_sections.add(section)

        selected.append({
            "card_name": meta.get("card_name"),
            "issuer": meta.get("issuer"),
            "section": section,
            "distance": distance,
            "document": doc,
        })

    return selected


def build_card_summary_block(card_row: pd.Series) -> str:
    return "\n".join([
        f"Card Name: {card_row.get('card_name', '')}",
        f"Issuer: {card_row.get('issuer', '')}",
        f"Network: {card_row.get('network', '')}",
        f"Annual Fee: {card_row.get('annual_fee_value', '')}",
        f"Rewards Type: {card_row.get('rewards_type', '')}",
        f"Best Categories: {card_row.get('best_categories', '')}",
        f"Welcome Bonus: {card_row.get('welcome_bonus_summary', '')}",
        f"Eligibility: {card_row.get('eligibility_summary', '')}",
        f"Reason from recommender: {card_row.get('reason', '')}",
        f"Recommendation Score: {card_row.get('total_score', '')}",
    ])


def build_explanation_prompt(
    user_query: str,
    user_profile: Dict[str, Any],
    recommended_cards_df: pd.DataFrame,
    card_chunks_map: Dict[str, List[Dict[str, Any]]],
) -> str:
    lines = []

    lines.append("You are a credit card recommendation assistant.")
    lines.append("Use the user profile, recommended cards, and supporting card context below to explain the best options.")
    lines.append("Prioritize accuracy and do not invent benefits that are not in the context.")
    lines.append("")
    lines.append("USER QUERY:")
    lines.append(user_query)
    lines.append("")
    lines.append("USER PROFILE:")
    for k, v in user_profile.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("TOP RECOMMENDED CARDS:")

    for _, row in recommended_cards_df.iterrows():
        card_name = row.get("card_name", "")
        lines.append("")
        lines.append("=" * 80)
        lines.append(build_card_summary_block(row))
        lines.append("Supporting Context:")

        chunks = card_chunks_map.get(card_name, [])
        if not chunks:
            lines.append("- No supporting chunks found.")
        else:
            for idx, chunk in enumerate(chunks, start=1):
                lines.append(f"[Chunk {idx} | Section: {chunk['section']}]")
                lines.append(chunk["document"][:1200])
                lines.append("")

    lines.append("=" * 80)
    lines.append("")
    lines.append("INSTRUCTIONS FOR FINAL ANSWER:")
    lines.append("1. Recommend the top 3 cards for this user.")
    lines.append("2. Explain why each card matches the user's needs.")
    lines.append("3. Mention annual fee tradeoffs clearly.")
    lines.append("4. Mention reward type fit clearly.")
    lines.append("5. If one card is best overall, say why.")
    lines.append("6. If a card is strong but expensive, say that explicitly.")
    lines.append("7. Keep the answer practical and user-friendly.")
    lines.append("8. Do not mention cards that are not in the recommended list unless necessary.")

    return "\n".join(lines)


def explain_recommendation(
    user_query: str,
    user_profile: Dict[str, Any],
    top_n: int = 3,
) -> Dict[str, Any]:
    recommended_cards = recommend_cards(
        user_profile=user_profile,
        top_n=top_n,
    )

    if recommended_cards.empty:
        return {
            "recommended_cards": recommended_cards,
            "card_chunks_map": {},
            "prompt": "No cards matched the user's filters.",
        }

    collection = load_collection()

    card_chunks_map = {}
    for _, row in recommended_cards.iterrows():
        card_name = row["card_name"]
        chunks = get_supporting_chunks_for_card(
            collection=collection,
            card_name=card_name,
            user_query=user_query,
            n_results=10,
        )
        card_chunks_map[card_name] = chunks

    prompt = build_explanation_prompt(
        user_query=user_query,
        user_profile=user_profile,
        recommended_cards_df=recommended_cards,
        card_chunks_map=card_chunks_map,
    )

    return {
        "recommended_cards": recommended_cards,
        "card_chunks_map": card_chunks_map,
        "prompt": prompt,
    }


def main():
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

    result = explain_recommendation(
        user_query=user_query,
        user_profile=user_profile,
        top_n=3,
    )

    print("\n=== TOP RECOMMENDED CARDS ===")
    if result["recommended_cards"].empty:
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
        existing = [c for c in cols if c in result["recommended_cards"].columns]
        print(result["recommended_cards"][existing].to_string(index=False))

    print("\n=== PROMPT FOR LLM ===\n")
    print(result["prompt"][:8000])  # preview only


if __name__ == "__main__":
    main()