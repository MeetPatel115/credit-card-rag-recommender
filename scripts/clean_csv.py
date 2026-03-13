from pathlib import Path
import re
import pandas as pd

IN_PATH = Path("data/processed/combined_cards.csv")
OUT_PATH = Path("data/processed/combined_cards_enriched.csv")


def clean_text(value):
    if pd.isna(value) or value is None:
        return None
    value = str(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value if value else None


def parse_money(value):
    if pd.isna(value) or value is None:
        return None
    text = str(value).replace(",", "").strip().lower()

    if text in {"none", "nan", "", "not available", "n/a"}:
        return None

    if "no annual fee" in text or "no fee" in text:
        return 0.0

    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group()) if match else None


def parse_income_from_text(text):
    if not text or pd.isna(text):
        return None, None

    t = str(text).replace(",", "")
    nums = re.findall(r"\$?\s*(\d{2,6})", t)
    nums = [int(x) for x in nums]

    personal = None
    household = None

    lower = t.lower()

    if "personal income" in lower and nums:
        personal = nums[0]
    if "household income" in lower and len(nums) >= 2:
        household = nums[1]

    # fallback if two numbers exist
    if personal is None and household is None and len(nums) >= 2:
        personal, household = nums[0], nums[1]
    elif personal is None and len(nums) == 1:
        personal = nums[0]

    return personal, household

def classify_rewards_type(row):
    card_name = str(row.get("card_name") or "").lower()
    issuer = str(row.get("issuer") or "").lower()
    welcome = str(row.get("welcome_bonus_summary") or "").lower()
    best_for = str(row.get("best_for") or "").lower()
    one_liner = str(row.get("one_liner") or "").lower()

    text = " ".join([card_name, issuer, welcome, best_for, one_liner])

    # specific programs first
    if "aeroplan" in card_name or "aeroplan" in text:
        return "aeroplan"

    if "avion" in card_name or "avion" in text:
        return "avion"

    if "scene+" in text or "scene plus" in text:
        return "scene+"

    if "aventura" in card_name or "aventura" in text:
        return "aventura"

    if "bonvoy" in text or "marriott" in text:
        return "hotel_points"

    if "membership rewards" in text or "mr points" in text:
        return "membership_rewards"

    # cashback should come later, not first
    if "cash back" in card_name or "cashback" in card_name:
        return "cashback"

    if "cash back" in welcome or "cashback" in welcome:
        return "cashback"

    if "cash back" in one_liner or "cashback" in one_liner:
        return "cashback"

    # american express fallback
    if "american express" in issuer or issuer == "amex":
        return "points"

    # generic points fallback
    if "points" in text:
        return "points"

    return "unknown"
def extract_best_categories(text):
    if not text or pd.isna(text):
        return ["general"]

    t = str(text).lower()

    mapping = {
        "grocery": ["grocery", "groceries", "supermarket"],
        "dining": ["dining", "restaurant", "food", "eats", "drinks"],
        "travel": ["travel", "trip", "flight", "hotel", "airfare"],
        "gas": ["gas", "fuel"],
        "transit": ["transit", "commute", "transport"],
        "streaming": ["streaming", "subscription"],
        "drugstore": ["drugstore", "pharmacy"],
        "student": ["student"],
        "business": ["business"],
        "cashback": ["cash back", "cashback"],
        "daily_spend": ["daily spend", "everyday", "every day"],
    }

    labels = []
    for label, keywords in mapping.items():
        if any(k in t for k in keywords):
            labels.append(label)

    return labels if labels else ["general"]

def main():
    df = pd.read_csv(IN_PATH)

    for col in df.columns:
        df[col] = df[col].apply(clean_text)

    # numeric fee columns
    df["monthly_fee_value"] = df["monthly_fee"].apply(parse_money)
    df["annual_fee_value"] = df["annual_fee"].apply(parse_money)

    # derive annual fee if only monthly fee exists
    missing_annual = df["annual_fee_value"].isna() & df["monthly_fee_value"].notna()
    df.loc[missing_annual, "annual_fee_value"] = df.loc[missing_annual, "monthly_fee_value"] * 12

    # income extraction from eligibility text
    income_pairs = df["eligibility_summary"].apply(parse_income_from_text)
    df["income_requirement_personal"] = income_pairs.apply(lambda x: x[0])
    df["income_requirement_household"] = income_pairs.apply(lambda x: x[1])

    # simple enrichment
    df["rewards_type"] = df.apply(classify_rewards_type, axis=1)
    df["best_categories_list"] = df["best_for"].apply(extract_best_categories)
    df["best_categories"] = df["best_categories_list"].apply(lambda x: ",".join(x))

    # placeholder columns for future structured extraction
    extra_cols = [
        "fx_fee",
        "lounge_access",
        "insurance_summary",
        "grocery_rate",
        "dining_rate",
        "gas_rate",
        "travel_rate",
        "transit_rate",
        "drugstore_rate",
        "streaming_rate",
        "base_rate",
    ]
    for col in extra_cols:
        if col not in df.columns:
            df[col] = None

    # reorder
    preferred_order = [
        "card_id",
        "card_name",
        "issuer",
        "country",
        "network",
        "card_type",
        "link",
        "monthly_fee",
        "monthly_fee_value",
        "annual_fee",
        "annual_fee_value",
        "welcome_bonus_summary",
        "best_for",
        "best_categories",
        "best_categories_list",
        "one_liner",
        "eligibility_summary",
        "income_requirement_personal",
        "income_requirement_household",
        "rewards_type",
        "fx_fee",
        "lounge_access",
        "insurance_summary",
        "grocery_rate",
        "dining_rate",
        "gas_rate",
        "travel_rate",
        "transit_rate",
        "drugstore_rate",
        "streaming_rate",
        "base_rate",
    ]
    existing = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    df = df[existing + remaining]
    df = df.drop(columns=["best_categories_list"], errors="ignore")
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved: {OUT_PATH}")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()