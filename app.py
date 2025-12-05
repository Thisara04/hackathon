import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sentence_transformers import SentenceTransformer

# ------------------------------------
# Load model
# ------------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    classifier = joblib.load("category_model.pkl")
    return embedder, classifier

embedder, classifier = load_models()

sector_map = {
    0: "Energy", 1: "Logistics", 2: "Education", 3: "Health",
    4: "Finance", 5: "Government", 6: "Tourism", 7: "Agriculture",
    8: "Social", 9: "Technology", 10: "Economy", 11: "Other"
}

# ------------------------------------
# Keyword scoring
# ------------------------------------
def calc_score(text, words):
    return sum(1 for w in words if w.lower() in text.lower())

economy_kw = ['stock','rupee','inflation','currency','finance','economic']
weather_kw = ['rain','flood','storm','temperature','drought']
social_kw = ['protest','strike','crowd','community']
logistics_kw = ['traffic','accident','port','delivery','transport']
tourism_kw = ['tourism','travel','hotel','tourist','visa']

def generate_insight(r):
    insights = []
    if r["Economy_Score"] >= 2: insights.append("Economic risk rising")
    if r["Weather_Score"] >= 1: insights.append("Weather disruption possible")
    if r["Social_Score"] >= 1: insights.append("Social unrest warning")
    if r["Logistics_Score"] >= 1: insights.append("Transport/Logistics alert")
    if r["Tourism_Score"] >= 1: insights.append("Tourism opportunity")
    return "; ".join(insights) if insights else "Normal"

# ------------------------------------
# UI SECTION
# ------------------------------------
st.title("📊 News Intelligence Dashboard (Phase 3–5)")

uploaded = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # ------------------------------------------------
    # Prediction (Phase 3)
    # ------------------------------------------------
    st.subheader("🔍 Running Categorization Model...")

    # Embeddings + time features
    text = df["Content"].tolist()
    Xtext = embedder.encode(text, convert_to_numpy=True)
    Xtime = df[["Month_sin","Month_cos","DOW_sin","DOW_cos"]].to_numpy()
    X = np.hstack([Xtext, Xtime])

    df["SectorID"] = classifier.predict(X)
    df["Sector"] = df["SectorID"].map(sector_map)

    # ------------------------------------------------
    # Signal Scoring (Phase 4)
    # ------------------------------------------------
    st.subheader("⚠ Generating Risk Signals...")

    df["Economy_Score"] = df["Content"].apply(lambda x: calc_score(x, economy_kw))
    df["Weather_Score"] = df["Content"].apply(lambda x: calc_score(x, weather_kw))
    df["Social_Score"] = df["Content"].apply(lambda x: calc_score(x, social_kw))
    df["Logistics_Score"] = df["Content"].apply(lambda x: calc_score(x, logistics_kw))
    df["Tourism_Score"] = df["Content"].apply(lambda x: calc_score(x, tourism_kw))
    df["Insight"] = df.apply(generate_insight, axis=1)

    # ------------------------------------------------
    # Phase 5: Visualization
    # ------------------------------------------------
    st.header("📈 Analytics & Visualizations")

    # 1️⃣ Sector Distribution
    st.subheader("1. Sector Distribution")
    fig1 = px.bar(df["Sector"].value_counts(), title="News Count per Sector")
    st.plotly_chart(fig1)

    # 2️⃣ Heatmap
    st.subheader("2. Risk Score Heatmap")
    heat = df.groupby("Sector")[["Economy_Score","Weather_Score","Social_Score","Logistics_Score","Tourism_Score"]].sum()
    fig2 = px.imshow(heat, text_auto=True, title="Risk Heatmap by Sector")
    st.plotly_chart(fig2)

    # 3️⃣ Risk Summary Metrics
    st.subheader("3. Risk Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Economy Alerts", heat["Economy_Score"].sum())
    col2.metric("Weather Alerts", heat["Weather_Score"].sum())
    col3.metric("Social Alerts", heat["Social_Score"].sum())
    col4.metric("Logistics Alerts", heat["Logistics_Score"].sum())
    col5.metric("Tourism Signals", heat["Tourism_Score"].sum())

    # 4️⃣ Top Insights
    st.subheader("4. Insights")
    st.dataframe(df[["Content","Sector","Insight"]])

    # Download results
    st.download_button(
        label="Download Output CSV",
        data=df.to_csv(index=False),
        file_name="signals_output.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV to begin.")
