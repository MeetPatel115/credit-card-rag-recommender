from pathlib import Path
import json
import pandas as pd

CSV_PATH = Path("data/processed/combined_cards_enriched.csv")
CHUNKS_IN = Path("data/processed/all_card_snippets.jsonl")
CHUNKS_OUT = Path("data/processed/all_card_snippets_enriched.jsonl")


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_val(value, default="Not available"):
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def row_to_card_doc(row: dict, section_text: str, section_name: str) -> str:
    parts = [
        f"Card Name: {safe_val(row.get('card_name'))}",
        f"Issuer: {safe_val(row.get('issuer'))}",
        f"Country: {safe_val(row.get('country'))}",
        f"Network: {safe_val(row.get('network'))}",
        f"Card Type: {safe_val(row.get('card_type'))}",
        f"Annual Fee: {safe_val(row.get('annual_fee_value', row.get('annual_fee')))}",
        f"Monthly Fee: {safe_val(row.get('monthly_fee_value', row.get('monthly_fee')))}",
        f"Rewards Type: {safe_val(row.get('rewards_type'))}",
        f"Best Categories: {safe_val(row.get('best_categories'))}",
        f"Welcome Bonus: {safe_val(row.get('welcome_bonus_summary'))}",
        f"Eligibility: {safe_val(row.get('eligibility_summary'))}",
        f"Personal Income Requirement: {safe_val(row.get('income_requirement_personal'))}",
        f"Household Income Requirement: {safe_val(row.get('income_requirement_household'))}",
        f"FX Fee: {safe_val(row.get('fx_fee'))}",
        f"Lounge Access: {safe_val(row.get('lounge_access'))}",
        f"Insurance Summary: {safe_val(row.get('insurance_summary'))}",
        f"Section: {safe_val(section_name)}",
        f"Section Text: {safe_val(section_text)}",
        f"Apply Link: {safe_val(row.get('link'))}",
    ]
    return "\n".join(parts)


def main():
    cards_df = pd.read_csv(CSV_PATH)

    # normalize card_id
    cards_df["card_id"] = cards_df["card_id"].astype(str).str.strip()
    card_map = cards_df.set_index("card_id").to_dict(orient="index")

    total = 0
    missing = 0

    with open(CHUNKS_OUT, "w", encoding="utf-8") as out:
        for chunk in load_jsonl(CHUNKS_IN):
            card_id = str(chunk.get("card_id", "")).strip()
            row = card_map.get(card_id)

            if row is None:
                missing += 1
                continue

            enriched = {
                "id": chunk.get("chunk_id"),
                "chunk_id": chunk.get("chunk_id"),
                "card_id": card_id,
                "card_name": row.get("card_name"),
                "issuer": row.get("issuer"),
                "country": row.get("country"),
                "network": row.get("network"),
                "card_type": row.get("card_type"),
                "annual_fee_value": row.get("annual_fee_value"),
                "monthly_fee_value": row.get("monthly_fee_value"),
                "rewards_type": row.get("rewards_type"),
                "best_categories": row.get("best_categories"),
                "income_requirement_personal": row.get("income_requirement_personal"),
                "income_requirement_household": row.get("income_requirement_household"),
                "section": chunk.get("section"),
                "raw_text": chunk.get("text"),
                "document": row_to_card_doc(
                    row=row,
                    section_text=chunk.get("text", ""),
                    section_name=chunk.get("section", ""),
                ),
                "metadata": {
                    "card_id": card_id,
                    "card_name": safe_val(row.get("card_name"), ""),
                    "issuer": safe_val(row.get("issuer"), ""),
                    "network": safe_val(row.get("network"), ""),
                    "card_type": safe_val(row.get("card_type"), ""),
                    "annual_fee_value": (
                        float(row["annual_fee_value"])
                        if pd.notna(row.get("annual_fee_value"))
                        else -1.0
                    ),
                    "rewards_type": safe_val(row.get("rewards_type"), ""),
                    "section": safe_val(chunk.get("section"), ""),
                    "best_categories": safe_val(row.get("best_categories"), ""),
                },
            }

            out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            total += 1

    print(f"Saved: {CHUNKS_OUT}")
    print(f"Enriched chunks: {total}")
    print(f"Skipped due to missing card_id in CSV: {missing}")


if __name__ == "__main__":
    main()