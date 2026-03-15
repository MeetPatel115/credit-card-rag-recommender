from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

CSV_PATH = Path(r"C:\Users\91951\OneDrive\Desktop\pythonProject\leetcode\Ai-ML-Projects\credit-card-rag\data\processed\combined_cards_enriched.csv")


def load_cards(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    text_cols = [
        "card_id",
        "card_name",
        "issuer",
        "country",
        "network",
        "card_type",
        "link",
        "welcome_bonus_summary",
        "best_for",
        "best_categories",
        "one_liner",
        "eligibility_summary",
        "rewards_type",
        "fx_fee",
        "lounge_access",
        "insurance_summary",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    numeric_cols = [
        "annual_fee_value",
        "monthly_fee_value",
        "income_requirement_personal",
        "income_requirement_household",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def parse_categories(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip().lower()
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def is_income_eligible(
    row: pd.Series,
    income: Optional[float] = None,
    household_income: Optional[float] = None,
) -> bool:
    if income is None and household_income is None:
        return True

    personal_req = row.get("income_requirement_personal")
    household_req = row.get("income_requirement_household")

    personal_ok = pd.isna(personal_req) or income is None or income >= personal_req
    household_ok = pd.isna(household_req) or household_income is None or household_income >= household_req

    # If both requirements exist, qualifying for either is usually enough.
    if pd.notna(personal_req) and pd.notna(household_req):
        return (
            (income is not None and income >= personal_req)
            or (household_income is not None and household_income >= household_req)
        )

    if pd.notna(personal_req):
        return income is not None and income >= personal_req

    if pd.notna(household_req):
        return household_income is not None and household_income >= household_req

    return True


def filter_cards(
    df: pd.DataFrame,
    user_profile: Dict[str, Any],
) -> pd.DataFrame:
    result = df.copy()

    max_annual_fee = user_profile.get("max_annual_fee")
    issuer = user_profile.get("issuer")
    network = user_profile.get("network")
    income = user_profile.get("income")
    household_income = user_profile.get("household_income")

    if max_annual_fee is not None:
        result = result[result["annual_fee_value"].fillna(999999) <= max_annual_fee]

    if issuer:
        result = result[result["issuer"].str.lower() == str(issuer).lower()]

    if network:
        result = result[result["network"].str.lower() == str(network).lower()]

    result = result[
        result.apply(
            lambda row: is_income_eligible(
                row,
                income=income,
                household_income=household_income,
            ),
            axis=1,
        )
    ]

    return result.copy()


def compute_card_score(
    row: pd.Series,
    user_profile: Dict[str, Any],
) -> Dict[str, float]:
    score = 0.0

    target_categories = [str(x).lower() for x in user_profile.get("target_categories", [])]
    preferred_rewards = str(user_profile.get("preferred_rewards") or "").lower().strip()
    max_annual_fee = user_profile.get("max_annual_fee")

    card_categories = parse_categories(row.get("best_categories"))
    rewards_type = str(row.get("rewards_type") or "").lower().strip()
    annual_fee = row.get("annual_fee_value")

    category_score = 0.0
    for cat in target_categories:
        if cat in card_categories:
            category_score += 3.0

    reward_score = 0.0
    if preferred_rewards and rewards_type == preferred_rewards:
        reward_score = 2.5

    fee_score = 0.0
    fee_penalty = 0.0
    if pd.notna(annual_fee):
        if annual_fee == 0:
            fee_score += 2.5
        elif annual_fee <= 120:
            fee_score += 1.5
        elif annual_fee <= 180:
            fee_score += 0.5

        if max_annual_fee is not None and annual_fee > max_annual_fee:
            fee_penalty += 5.0

    score += category_score
    score += reward_score
    score += fee_score
    score -= fee_penalty

    return {
        "category_score": category_score,
        "reward_score": reward_score,
        "fee_score": fee_score,
        "fee_penalty": fee_penalty,
        "total_score": score,
    }


def build_reason(row: pd.Series, user_profile: Dict[str, Any]) -> str:
    reasons: List[str] = []

    target_categories = [str(x).lower() for x in user_profile.get("target_categories", [])]
    preferred_rewards = str(user_profile.get("preferred_rewards") or "").lower().strip()
    card_categories = parse_categories(row.get("best_categories"))
    rewards_type = str(row.get("rewards_type") or "").lower().strip()
    annual_fee = row.get("annual_fee_value")

    matched_categories = [cat for cat in target_categories if cat in card_categories]
    if matched_categories:
        reasons.append(f"matches your target categories: {', '.join(matched_categories)}")

    if preferred_rewards and rewards_type == preferred_rewards:
        reasons.append(f"matches your preferred rewards type: {rewards_type}")

    if pd.notna(annual_fee):
        if annual_fee == 0:
            reasons.append("has no annual fee")
        else:
            reasons.append(f"annual fee is ${annual_fee:.2f}")

    if row.get("welcome_bonus_summary"):
        reasons.append("includes a welcome bonus")

    if not reasons:
        reasons.append("fits your filters reasonably well")

    return "; ".join(reasons)


def recommend_cards(
    user_profile: Dict[str, Any],
    csv_path: Path = CSV_PATH,
    top_n: int = 5,
) -> pd.DataFrame:
    df = load_cards(csv_path)
    filtered = filter_cards(df, user_profile=user_profile)

    if filtered.empty:
        return filtered

    score_rows: List[Dict[str, Any]] = []
    for _, row in filtered.iterrows():
        scores = compute_card_score(row, user_profile)
        score_rows.append(scores)

    score_df = pd.DataFrame(score_rows, index=filtered.index)
    ranked = pd.concat([filtered, score_df], axis=1)

    ranked["reason"] = ranked.apply(lambda row: build_reason(row, user_profile), axis=1)

    ranked = ranked.sort_values(
        by=["total_score", "annual_fee_value", "card_name"],
        ascending=[False, True, True],
    )

    return ranked.head(top_n).copy()


def print_recommendations(df: pd.DataFrame) -> None:
    if df.empty:
        print("No cards matched the current filters.")
        return

    display_cols = [
        "card_id",
        "card_name",
        "issuer",
        "network",
        "annual_fee_value",
        "rewards_type",
        "best_categories",
        "total_score",
        "reason",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    print(df[existing_cols].to_string(index=False))


def main() -> None:
    user_profile = {
        "income": 70000,
        "household_income": None,
        "max_annual_fee": 180,
        "preferred_rewards": "membership_rewards",
        "target_categories": ["grocery", "dining"],
        "issuer": None,
        "network": None,
    }

    recommendations = recommend_cards(user_profile=user_profile, top_n=10)
    print_recommendations(recommendations)


if __name__ == "__main__":
    main()