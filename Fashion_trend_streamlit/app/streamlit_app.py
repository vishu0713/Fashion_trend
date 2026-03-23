import sys
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.keyword_detector import detect_keyword
from src.trend_analysis import analyze_keyword_trend
from src.time_series_forecast import forecast_keyword
from src.chatbot_response import generate_fashion_response

st.set_page_config(
    page_title="FashionTrend AI",
    page_icon="🖤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #060b16 0%, #0b1020 100%);
        color: #f5f5f5;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero {
        background: linear-gradient(135deg, #111111 0%, #1f1f1f 35%, #3a2f2f 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 28px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.15);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #e7e2de;
        max-width: 760px;
    }

    .prediction-card {
        background: linear-gradient(135deg, #ff4d6d 0%, #ff758f 45%, #ff8fab 100%);
        color: white;
        border-radius: 24px;
        padding: 1.6rem 1.5rem;
        box-shadow: 0 16px 35px rgba(255, 77, 109, 0.24);
        margin-top: 0.6rem;
        margin-bottom: 1rem;
    }

    .prediction-small {
        font-size: 0.95rem;
        opacity: 0.92;
        margin-bottom: 0.35rem;
    }

    .prediction-main {
        font-size: 2rem;
        font-weight: 900;
        margin-bottom: 0.3rem;
    }

    .card {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border: 1px solid #1f2937;
        border-radius: 22px;
        padding: 1.2rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
        color: #f9fafb;
    }

    .card-title {
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
        color: #ffffff;
        letter-spacing: -0.3px;
    }

    .section-subtitle {
        font-size: 1rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 0.6rem;
    }

    .ai-text {
        font-size: 1.05rem;
        line-height: 1.8;
        color: #f3f4f6;
    }

    .badge-hot {
        display: inline-block;
        background: #111111;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .badge-rise {
        display: inline-block;
        background: #0f766e;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .badge-cool {
        display: inline-block;
        background: #475569;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .shop-pill {
        display: inline-block;
        padding: 0.7rem 1rem;
        border-radius: 999px;
        background: #111111;
        color: white !important;
        text-decoration: none !important;
        margin-right: 0.6rem;
        margin-bottom: 0.6rem;
        font-weight: 600;
    }

    .helper-text {
        color: #cbd5e1;
        font-size: 0.98rem;
        line-height: 1.8;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid #1f2937;
        padding: 14px;
        border-radius: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helper functions
# -----------------------------
def get_style_label(direction: str, confidence: float) -> str:
    if direction == "Rising" and confidence >= 0.7:
        return "Hot Trend"
    elif direction == "Rising":
        return "Emerging"
    elif direction == "Stable":
        return "Stable Pick"
    else:
        return "Cooling Down"


def get_badge_html(label: str) -> str:
    if label in ["Hot Trend", "Emerging"]:
        return f'<span class="badge-rise">{label}</span>'
    elif label == "Stable Pick":
        return f'<span class="badge-hot">{label}</span>'
    return f'<span class="badge-cool">{label}</span>'


def make_store_links(keyword: str):
    query = keyword.replace("_", " ").replace(" ", "+")
    return {
        "Zara": f"https://www.zara.com/ca/en/search?searchTerm={query}",
        "H&M": f"https://www2.hm.com/en_ca/search-results.html?q={query}",
        "SHEIN": f"https://ca.shein.com/search?keyword={query}",
    }


def get_related_items(keyword: str):
    related_map = {
        "cargo_pants": ["oversized hoodie", "chunky sneakers", "baggy jeans"],
        "baggy_jeans": ["graphic hoodie", "chunky sneakers", "bomber jacket"],
        "oversized_hoodie": ["baggy jeans", "cargo pants", "white sneakers"],
        "puffer_jacket": ["combat boots", "baggy jeans", "beanie"],
        "linen_shirt": ["wide leg jeans", "mini bag", "white sneakers"],
        "leather_jacket": ["combat boots", "mini skirt", "shoulder bag"],
        "mini_skirt": ["corset top", "shoulder bag", "platform sneakers"],
        "chunky_sneakers": ["cargo pants", "baggy jeans", "oversized hoodie"]
    }
    return related_map.get(keyword, ["baggy jeans", "oversized hoodie", "chunky sneakers"])


def plot_recent_trend(chart_df, keyword):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(chart_df["date"], chart_df["trend"], marker="o", linewidth=2.5)
    ax.scatter(chart_df["date"].iloc[-1], chart_df["trend"].iloc[-1], s=90)
    ax.set_title(f"Recent Trend: {keyword.replace('_', ' ').title()}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Trend Score")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_forecast(history, forecast_df, keyword):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(history["date"], history["trend"], marker="o", linewidth=2.5, label="History")
    ax.plot(
        forecast_df["date"],
        forecast_df["forecast"],
        marker="o",
        linestyle="--",
        linewidth=2.5,
        label="Forecast"
    )
    ax.set_title(f"Future Forecast: {keyword.replace('_', ' ').title()}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Trend Score")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


# -----------------------------
# Hero section
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">FashionTrend AI</div>
    <div class="hero-subtitle">
        Discover whether a fashion item is rising, stable, or cooling down.
        Get AI-powered trend analysis, short-term forecasting, and direct shopping links — all in one stylish experience.
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([3, 1])
with left:
    user_query = st.text_input(
        "Ask about a fashion item",
        placeholder="Example: Are cargo pants trending right now?"
    )
with right:
    forecast_steps = st.selectbox("Forecast horizon", [3, 6, 9, 12], index=0)

# -----------------------------
# Default intro
# -----------------------------
if not user_query:
    st.markdown("""
    <div class="card">
        <div class="card-title">How to Use</div>
        <div class="helper-text">
            Try asking things like:<br><br>
            • Are cargo pants trending right now?<br>
            • Should I buy a puffer jacket?<br>
            • Tell me about baggy jeans<br>
            • What is the trend for oversized hoodie?
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Main flow
# -----------------------------
if user_query:
    keyword = detect_keyword(user_query)

    if not keyword:
        st.error("I couldn’t detect a supported fashion keyword from your query.")
        st.info("Try a query with items like cargo pants, baggy jeans, oversized hoodie, or puffer jacket.")
    else:
        with st.spinner("Analyzing fashion trend..."):
            trend_result = analyze_keyword_trend(keyword)
            forecast_result = forecast_keyword(keyword, steps=forecast_steps)

        if trend_result["success"] and forecast_result["success"]:
            style_label = get_style_label(
                trend_result["recent_direction"],
                trend_result["confidence"]
            )
            badge_html = get_badge_html(style_label)

            forecast_values = forecast_result["forecast_df"]["forecast"].round(2).tolist()

            with st.spinner("Generating AI stylist response..."):
                ai_text = generate_fashion_response(
                    user_query=user_query,
                    keyword=keyword,
                    latest_trend=trend_result["latest_trend"],
                    previous_trend=trend_result["previous_trend"],
                    recent_direction=trend_result["recent_direction"],
                    prediction_label=trend_result["prediction_label"],
                    confidence=trend_result["confidence"],
                    forecast_values=forecast_values
                )

            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-small">Instant Fashion Signal</div>
                <div class="prediction-main">{trend_result["prediction_label"]}</div>
                <div>
                    <strong>{keyword.replace("_", " ").title()}</strong> is currently
                    <strong>{trend_result["recent_direction"].lower()}</strong> with confidence
                    <strong>{trend_result["confidence"]:.2f}</strong>.
                </div>
                <div style="margin-top: 0.7rem;">{badge_html}</div>
            </div>
            """, unsafe_allow_html=True)

            a, b, c, d = st.columns(4)
            a.metric("Detected Item", keyword.replace("_", " ").title())
            b.metric("Latest Trend", f'{trend_result["latest_trend"]:.2f}')
            c.metric("Previous Trend", f'{trend_result["previous_trend"]:.2f}')
            d.metric("Direction", trend_result["recent_direction"])

            st.markdown(f"""
            <div class="card">
                <div class="card-title">AI Fashion Insight</div>
                <div class="ai-text">{ai_text}</div>
            </div>
            """, unsafe_allow_html=True)

            related = get_related_items(keyword)
            st.caption("You may also like: " + ", ".join(related))

            tab1, tab2, tab3, tab4 = st.tabs([
                "Trend Overview",
                "Forecast",
                "Shop the Item",
                "Forecast Table"
            ])

            with tab1:
                col1, col2 = st.columns([1.5, 1])

                with col1:
                    st.markdown(
                        '<div class="card-title">Trend Graph</div><div class="section-subtitle">Recent movement of the selected fashion item</div>',
                        unsafe_allow_html=True
                    )
                    fig1 = plot_recent_trend(trend_result["chart_data"], keyword)
                    st.pyplot(fig1)

                with col2:
                    st.markdown("""
                    <div class="card">
                        <div class="card-title">Trend Summary</div>
                        <div class="section-subtitle">Key indicators from your model and historical data</div>
                    """, unsafe_allow_html=True)

                    st.write(f"**Prediction:** {trend_result['prediction_label']}")
                    st.write(f"**Confidence:** {trend_result['confidence']:.2f}")
                    st.write(f"**3-Period Moving Average:** {trend_result['ma3']:.2f}")
                    st.write(f"**6-Period Moving Average:** {trend_result['ma6']:.2f}")
                    st.write(f"**Recent Change:** {trend_result['slope']:.2f}")
                    st.write(f"**Percentage Change:** {trend_result['pct_change']:.2%}")
                    st.write(f"**Latest Date in Data:** {trend_result['latest_date']}")
                    st.caption("This section summarizes recent behavior of the selected fashion item based on historical trend values.")
                    st.markdown("</div>", unsafe_allow_html=True)

            with tab2:
                st.markdown(
                    '<div class="card-title">Forecast Chart</div><div class="section-subtitle">Short-term forecast for upcoming months</div>',
                    unsafe_allow_html=True
                )
                fig2 = plot_forecast(
                    forecast_result["history"],
                    forecast_result["forecast_df"],
                    keyword
                )
                st.pyplot(fig2)
                st.caption(f"Forecast horizon selected: {forecast_steps} month(s). Shorter forecasts are usually more reliable.")

            with tab3:
                st.markdown("""
                <div class="card">
                    <div class="card-title">Browse Similar Products</div>
                    <div class="section-subtitle">Open retail search pages for the detected item</div>
                """, unsafe_allow_html=True)

                links = make_store_links(keyword)
                for brand, url in links.items():
                    st.markdown(
                        f'<a class="shop-pill" href="{url}" target="_blank">{brand}</a>',
                        unsafe_allow_html=True
                    )

                st.caption("These links open search pages for the detected item on fashion retail websites.")
                st.markdown("</div>", unsafe_allow_html=True)

            with tab4:
                st.markdown(
                    '<div class="card-title">Forecast Data</div><div class="section-subtitle">Future forecast values in table format</div>',
                    unsafe_allow_html=True
                )
                st.dataframe(forecast_result["forecast_df"], use_container_width=True)

        else:
            if not trend_result["success"]:
                st.error(trend_result["message"])
            if not forecast_result["success"]:
                st.error(forecast_result["message"])