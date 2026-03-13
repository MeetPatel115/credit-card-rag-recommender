from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

CSV_PATH = Path(r"C:\Users\91951\OneDrive\Desktop\pythonProject\leetcode\Ai-ML-Projects\credit-card-rag\data\processed\combined_cards_enriched.csv")
DB_DIR = r"C:\Users\91951\OneDrive\Desktop\pythonProject\leetcode\Ai-ML-Projects\credit-card-rag\data\vectordb\chroma_db"
COLLECTION_NAME = "credit_card_chunks"

# keep only useful sections for recommendation
GOOD_SECTIONS = {
    "overview",
    "rewards",
    "welcome_bonus",
    "benefits",
    "eligibility",
    "insurance",
}

# optional section weights
SECTION_WEIGHTS = {
    "overview": 1.0,
    "rewards": 1.2,
    "welcome_bonus": 0.8,
    "benefits": 0.9,
    "eligibility": 0.7,
    "insurance": 0.6,
}


def load_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    return collection


def load_cards():
    df = pd.read_csv(CSV_PATH)

    # normalize
    for col in ["card_name", "issuer", "network", "rewards_type", "best_categories"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if "annual_fee_value" in df.columns:
        df["annual_fee_value"] = pd.to_numeric(df["annual_fee_value"], errors="coerce")

    if "income_requirement_personal" in df.columns:
        df["income_requirement_personal"] = pd.to_numeric(df["income_requirement_personal"], errors="coerce")

    if "income_requirement_household" in df.columns:
        df["income_requirement_household"] = pd.to_numeric(df["income_requirement_household"], errors="coerce")

    return df


def filter_cards(df, max_annual_fee=None, issuer=None, network=None, income=None):
    result = df.copy()

    if max_annual_fee is not None:
        result = result[result["annual_fee_value"].fillna(999999) <= max_annual_fee]

    if issuer:
        result = result[result["issuer"].str.lower() == issuer.lower()]

    if network:
        result = result[result["network"].str.lower() == network.lower()]

    if income is not None and "income_requirement_personal" in result.columns:
        # keep cards where requirement is missing OR user qualifies
        result = result[
            result["income_requirement_personal"].isna() |
            (result["income_requirement_personal"] <= income)
        ]

    return result


def parse_query_preferences(query: str):
    q = query.lower()

    prefs = {
        "target_categories": [],
        "wants_low_fee": False,
        "rewards_type": None,
    }

    category_keywords = {
        "grocery": ["grocery", "groceries", "supermarket"],
        "dining": ["dining", "restaurant", "restaurants", "food"],
        "travel": ["travel", "flight", "hotel", "trip"],
        "gas": ["gas", "fuel"],
        "transit": ["transit", "commute", "transport"],
        "streaming": ["streaming", "subscription"],
        "drugstore": ["drugstore", "pharmacy"],
        "cashback": ["cash back", "cashback"],
    }

    for label, keywords in category_keywords.items():
        if any(k in q for k in keywords):
            prefs["target_categories"].append(label)

    if "low annual fee" in q or "low fee" in q or "no annual fee" in q:
        prefs["wants_low_fee"] = True

    if "cash back" in q or "cashback" in q:
        prefs["rewards_type"] = "cashback"
    elif "aeroplan" in q:
        prefs["rewards_type"] = "aeroplan"
    elif "avion" in q:
        prefs["rewards_type"] = "avion"
    elif "scene+" in q or "scene plus" in q:
        prefs["rewards_type"] = "scene+"
    elif "aventura" in q:
        prefs["rewards_type"] = "aventura"
    elif "membership rewards" in q:
        prefs["rewards_type"] = "membership_rewards"

    return prefs


def retrieve_chunks(collection, query, n_results=30):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    return results


def dedupe_and_score_chunks(results):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(docs)

    card_best = {}

    for doc, meta, distance in zip(docs, metas, distances):
        if not meta:
            continue

        section = str(meta.get("section", "")).lower()
        if section not in GOOD_SECTIONS:
            continue

        card_id = meta.get("card_id")
        if not card_id:
            continue

        # lower distance is better in chroma; convert to score
        base_score = 1.0
        if distance is not None:
            base_score = 1 / (1 + float(distance))

        weighted_score = base_score * SECTION_WEIGHTS.get(section, 1.0)

        current = card_best.get(card_id)
        if current is None or weighted_score > current["retrieval_score"]:
            card_best[card_id] = {
                "card_id": card_id,
                "card_name": meta.get("card_name"),
                "issuer": meta.get("issuer"),
                "network": meta.get("network"),
                "section": section,
                "retrieval_score": weighted_score,
                "document": doc,
                "metadata": meta,
            }

    deduped = pd.DataFrame(card_best.values())
    if not deduped.empty:
        deduped = deduped.sort_values("retrieval_score", ascending=False)

    return deduped


def compute_structured_score(df, query_prefs, max_annual_fee=None):
    scored = df.copy()
    scored["structured_score"] = 0.0

    target_categories = query_prefs.get("target_categories", [])
    rewards_type = query_prefs.get("rewards_type")
    wants_low_fee = query_prefs.get("wants_low_fee", False)

    # category match score
    for cat in target_categories:
        scored["structured_score"] += scored["best_categories"].str.contains(cat, case=False, na=False).astype(float) * 2.0

    # rewards type match
    if rewards_type:
        scored["structured_score"] += (
            scored["rewards_type"].str.lower() == rewards_type.lower()
        ).astype(float) * 2.0

    # fee preference
    if wants_low_fee:
        fee_series = scored["annual_fee_value"].fillna(999999)
        scored["structured_score"] += (fee_series <= 0).astype(float) * 3.0
        scored["structured_score"] += ((fee_series > 0) & (fee_series <= 120)).astype(float) * 1.5
        scored["structured_score"] += ((fee_series > 120) & (fee_series <= 180)).astype(float) * 0.5

    # soft fee bonus if user provided max fee
    if max_annual_fee is not None:
        fee_series = scored["annual_fee_value"].fillna(max_annual_fee + 999)
        fee_bonus = ((max_annual_fee - fee_series).clip(lower=0) / max(max_annual_fee, 1)) * 1.5
        scored["structured_score"] += fee_bonus.fillna(0)

    return scored


def combine_retrieval_and_structured(scored_cards, retrieved_cards):
    if retrieved_cards.empty:
        scored_cards["retrieval_score"] = 0.0
        scored_cards["final_score"] = scored_cards["structured_score"]
        return scored_cards.sort_values("final_score", ascending=False)

    merged = scored_cards.merge(
        retrieved_cards[["card_id", "retrieval_score", "section"]],
        on="card_id",
        how="left"
    )

    merged["retrieval_score"] = merged["retrieval_score"].fillna(0.0)
    merged["final_score"] = merged["structured_score"] + (merged["retrieval_score"] * 5.0)

    return merged.sort_values(
        ["final_score", "retrieval_score", "annual_fee_value"],
        ascending=[False, False, True]
    )


def recommend(query: str, user_preferences: dict):
    df = load_cards()
    collection = load_collection()

    filtered = filter_cards(
        df,
        max_annual_fee=user_preferences.get("max_annual_fee"),
        issuer=user_preferences.get("issuer"),
        network=user_preferences.get("network"),
        income=user_preferences.get("income"),
    )

    query_prefs = parse_query_preferences(query)
    scored = compute_structured_score(
        filtered,
        query_prefs=query_prefs,
        max_annual_fee=user_preferences.get("max_annual_fee"),
    )

    raw_results = retrieve_chunks(collection, query, n_results=30)
    deduped_retrieval = dedupe_and_score_chunks(raw_results)

    final_ranked = combine_retrieval_and_structured(scored, deduped_retrieval)

    top_cards = final_ranked.head(10)

    print("\n=== TOP FINAL RANKED CARDS ===")
    cols = [
        "card_id",
        "card_name",
        "issuer",
        "network",
        "annual_fee_value",
        "rewards_type",
        "best_categories",
        "structured_score",
        "retrieval_score",
        "final_score",
    ]
    existing_cols = [c for c in cols if c in top_cards.columns]
    print(top_cards[existing_cols].to_string(index=False))

    print("\n=== TOP UNIQUE RETRIEVED CARDS ===")
    if deduped_retrieval.empty:
        print("No retrieved cards after section filtering.")
    else:
        for i, row in enumerate(deduped_retrieval.head(10).itertuples(index=False), start=1):
            print(f"\n--- Retrieved {i} ---")
            print("card_id:", row.card_id)
            print("card_name:", row.card_name)
            print("issuer:", row.issuer)
            print("network:", row.network)
            print("section:", row.section)
            print("retrieval_score:", round(row.retrieval_score, 4))
            print("document preview:")
            print(str(row.document)[:500], "...")


if __name__ == "__main__":
    query = "best credit card for groceries and dining with low annual fee"
    user_preferences = {
        "max_annual_fee": 180,
        "income": 70000,
        # "issuer": "American Express",
        # "network": "Visa",
    }
    recommend(query, user_preferences)