from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMNS = [
    "card_id",
    "card_name",
    "issuer",
    "country",
    "network",
    "card_type",
    "link",
    "monthly_fee",
    "annual_fee",
    "welcome_bonus_summary",
    "best_for",
    "one_liner",
    "eligibility_summary",
]

# map possible alternate column names to standard names
COLUMN_ALIASES = {
    "issuer": ["issuer", "bank_name", "bank"],
    "card_name": ["card_name", "name", "title"],
    "country": ["country"],
    "network": ["network"],
    "card_type": ["card_type", "type"],
    "link": ["link", "url", "apply_url"],
    "monthly_fee": ["monthly_fee"],
    "annual_fee": ["annual_fee"],
    "welcome_bonus_summary": ["welcome_bonus_summary", "welcome_bonus", "welcome_offer"],
    "best_for": ["best_for"],
    "one_liner": ["one_liner", "summary"],
    "eligibility_summary": ["eligibility_summary", "eligibility"],
    "card_id": ["card_id", "id"],
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}

    lower_to_original = {c.lower().strip(): c for c in df.columns}

    for standard_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_key = alias.lower().strip()
            if alias_key in lower_to_original:
                rename_map[lower_to_original[alias_key]] = standard_col
                break

    df = df.rename(columns=rename_map)

    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[TARGET_COLUMNS]


def infer_issuer_from_filename(filename: str) -> str | None:
    name = filename.lower()
    if "amex" in name:
        return "American Express"
    if "cibc" in name:
        return "CIBC"
    if "scotia" in name:
        return "Scotiabank"
    if "rbc" in name:
        return "RBC"
    if "td" in name:
        return "TD"
    if "bmo" in name:
        return "BMO"
    return None


def main():
    csv_files = sorted(RAW_DIR.glob("cards_min_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No files found like data/raw/cards_min_*.csv")

    all_dfs = []

    for file in csv_files:
        print(f"Reading {file.name}")
        df = pd.read_csv(file)
        df = standardize_columns(df)

        if df["issuer"].isna().all():
            inferred = infer_issuer_from_filename(file.name)
            if inferred:
                df["issuer"] = inferred

        all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)

    merged = merged.drop_duplicates(subset=["card_id"], keep="first")
    merged = merged.drop_duplicates(subset=["issuer", "card_name"], keep="first")

    out_path = OUT_DIR / "combined_cards.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(merged)}")
    print(f"Columns: {list(merged.columns)}")


if __name__ == "__main__":
    main()