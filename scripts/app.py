import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Credit Card Recommendation Advisor",
    page_icon="💳",
    layout="wide",
)


def load_generate_response():
    candidate_paths = [
        Path(__file__).resolve().parent / "genrate_response.py",
        Path("scripts/genrate_response.py").resolve(),
    ]

    for path in candidate_paths:
        if path.exists():
            module_name = f"loaded_{path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                if hasattr(module, "generate_response"):
                    return module.generate_response

    raise ImportError("Could not find generate_response() in scripts/genrate_response.py")


@st.cache_resource
def get_generate_response_func():
    return load_generate_response()


def build_user_profile(
    income: int,
    household_income: int,
    max_annual_fee: int,
    preferred_rewards: str,
    target_categories: list[str],
    issuer: str,
    network: str,
) -> Dict[str, Any]:
    return {
        "income": income if income > 0 else None,
        "household_income": household_income if household_income > 0 else None,
        "max_annual_fee": max_annual_fee if max_annual_fee >= 0 else None,
        "preferred_rewards": None if preferred_rewards == "Any" else preferred_rewards,
        "target_categories": target_categories,
        "issuer": None if issuer == "Any" else issuer,
        "network": None if network == "Any" else network,
    }


def format_currency(value: Any) -> str:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "Not available"
        value = float(value)
        if value == 0:
            return "$0"
        return f"${value:,.2f}"
    except Exception:
        return str(value)


def render_recommended_cards(df: pd.DataFrame):
    st.subheader("Top Recommended Cards")

    if df is None or df.empty:
        st.warning("No cards matched the current filters.")
        return

    for _, row in df.iterrows():
        with st.container(border=True):
            left, right = st.columns([3, 1])

            with left:
                st.markdown(f"### {row.get('card_name', 'Unknown Card')}")
                st.write(f"**Issuer:** {row.get('issuer', 'Not available')}")
                st.write(f"**Network:** {row.get('network', 'Not available')}")
                st.write(f"**Rewards Type:** {row.get('rewards_type', 'Not available')}")
                st.write(f"**Best Categories:** {row.get('best_categories', 'Not available')}")
                st.write(f"**Why it fits:** {row.get('reason', 'No explanation available')}")

            with right:
                st.metric("Annual Fee", format_currency(row.get("annual_fee_value")))
                try:
                    st.metric("Score", f"{float(row.get('total_score', 0)):.2f}")
                except Exception:
                    st.metric("Score", str(row.get("total_score", "N/A")))


def render_chunks(card_chunks_map: Dict[str, list]):
    st.subheader("Supporting Retrieved Context")

    if not card_chunks_map:
        st.info("No supporting chunks were found.")
        return

    for card_name, chunks in card_chunks_map.items():
        with st.expander(f"{card_name} — Retrieved Chunks", expanded=False):
            if not chunks:
                st.write("No chunks found for this card.")
                continue

            for i, chunk in enumerate(chunks, start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(f"Section: {chunk.get('section', 'Unknown')}")
                st.text_area(
                    label=f"{card_name} chunk {i}",
                    value=str(chunk.get("document", ""))[:2500],
                    height=180,
                    key=f"{card_name}_{i}",
                    label_visibility="collapsed",
                )


def main():
    st.title("💳 Credit Card Recommendation Advisor")
    st.write(
        "This app uses structured card ranking, retrieved card context, and a local Ollama model "
        "to generate the final explanation."
    )

    with st.sidebar:
        st.header("User Preferences")

        user_query = st.text_area(
            "Query",
            value="best credit card for groceries and dining with low annual fee",
            height=100,
        )

        income = st.number_input("Personal Income", min_value=0, value=70000, step=1000)
        household_income = st.number_input("Household Income", min_value=0, value=0, step=1000)
        max_annual_fee = st.number_input("Max Annual Fee", min_value=0, value=180, step=10)

        preferred_rewards = st.selectbox(
            "Preferred Rewards Type",
            options=[
                "Any",
                "cashback",
                "membership_rewards",
                "points",
                "aeroplan",
                "avion",
                "scene+",
                "aventura",
                "hotel_points",
            ],
            index=1,
        )

        target_categories = st.multiselect(
            "Target Categories",
            options=[
                "grocery",
                "dining",
                "travel",
                "gas",
                "transit",
                "streaming",
                "drugstore",
                "student",
                "business",
                "cashback",
                "daily_spend",
            ],
            default=["grocery", "dining"],
        )

        issuer = st.selectbox(
            "Issuer Filter",
            options=[
                "Any",
                "American Express",
                "Scotiabank",
                "RBC",
                "TD",
                "CIBC",
                "BMO",
            ],
            index=0,
        )

        network = st.selectbox(
            "Network Filter",
            options=[
                "Any",
                "Visa",
                "Mastercard",
                "American Express",
            ],
            index=0,
        )

        top_n = st.slider("Number of Recommendations", min_value=1, max_value=10, value=3)
        ollama_model = st.text_input("Ollama Model", value="gemma3:1b")

        show_chunks = st.checkbox("Show Retrieved Chunks", value=False)
        show_prompt = st.checkbox("Show LLM Prompt", value=False)
        show_fallback = st.checkbox("Show Fallback Response", value=False)

        run_button = st.button("Recommend Cards", use_container_width=True)

    if not run_button:
        st.info("Set your preferences in the sidebar and click **Recommend Cards**.")
        return

    try:
        generate_response = get_generate_response_func()

        user_profile = build_user_profile(
            income=income,
            household_income=household_income,
            max_annual_fee=max_annual_fee,
            preferred_rewards=preferred_rewards,
            target_categories=target_categories,
            issuer=issuer,
            network=network,
        )

        with st.spinner("Generating recommendations with Ollama..."):
            result = generate_response(
                user_query=user_query,
                user_profile=user_profile,
                top_n=top_n,
                save_outputs=False,
                provider="ollama",
                model=ollama_model,
            )

        recommended_cards = result.get("recommended_cards")
        final_response = result.get("final_response", "")
        fallback_response = result.get("fallback_response", "")
        prompt = result.get("prompt", "")
        card_chunks_map = result.get("card_chunks_map", {})
        llm_error = result.get("llm_error")

        tab1, tab2 = st.tabs(["Recommendations", "Final Explanation"])

        with tab1:
            render_recommended_cards(recommended_cards)

        with tab2:
            st.subheader("Generated Recommendation")
            st.write(final_response if final_response else fallback_response)

            if llm_error:
                st.warning(f"Ollama issue detected. Fallback may have been used.\n\n{llm_error}")

        if show_fallback:
            st.subheader("Fallback Response")
            st.write(fallback_response)

        if show_chunks:
            render_chunks(card_chunks_map)

        if show_prompt:
            st.subheader("LLM Prompt Preview")
            st.text_area(
                "Prompt",
                value=prompt[:12000],
                height=400,
                key="prompt_preview",
            )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()