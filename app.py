import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import requests
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from streamlit_autorefresh import st_autorefresh

# =========================================================
# PAGE CONFIG + AUTO REFRESH
# =========================================================
st.set_page_config(page_title="News Intelligence Dashboard", layout="wide")

# Auto-refresh every 10 minutes (600,000 ms)
st_autorefresh(interval=600_000, key="refresh")

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    classifier = joblib.load("category_model.pkl")  # CatBoost model
    return embedder, classifier

embedder, classifier = load_models()

# Mapping from classifier output → readable sector
sector_map = {
    0: "Energy", 1: "Logistics", 2: "Education", 3: "Health",
    4: "Finance", 5: "Government", 6: "Tourism", 7: "Agriculture",
    8: "Social", 9: "Technology", 10: "Economy", 11: "Other"
}

# =========================================================
# SCORING KEYWORDS
# =========================================================
def calc_score(text, words):
    return sum(1 for w in words if w.lower() in text.lower())

economy_kw = ['stock','rupee','inflation','currency','finance','economic']
weather_kw = ['rain','flood','storm','temperature','drought']
social_kw  = ['protest','strike','crowd','community']
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

# =========================================================
# RSS FEED PARSER
# =========================================================
def fetch_rss(url):
    response = requests.get(url)
    root = ET.fromstring(response.content)

    items = []
    for item in root.findall(".//item"):
        title = item.findtext("title", "")
        pub_date = item.findtext("pubDate", "")
        link = item.findtext("link", "")
        items.append([title, pub_date, link])

    df = pd.DataFrame(items, columns=["Content", "Published", "Link"])

    # time features (sin/cos transforms)
    df["Published"] = pd.to_datetime(df["Published"], errors="coerce")
    df["month"] = df["Published"].dt.month
    df["dow"] = df["Published"].dt.dayofweek

    df["Month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["DOW_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
    df["DOW_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)

    return df

# =========================================================
# PROCESSING FUNCTION
# =========================================================
def process_dataframe(df):
    # embeddings + time features
    texts = df["Content"].tolist()
    Xtext = embedder.encode(texts, convert_to_numpy=True)
    Xtime = df[["Month_sin", "Month_cos", "DOW_sin", "DOW_cos"]].to_numpy()

    X = np.hstack([Xtext, Xtime])

    # Predict category
    df["SectorID"] = classifier.predict(X)
    df["Sector"] = df["SectorID"].map(sector_map)

    # Keyword scoring
    df["Economy_Score"] = df["Content"].apply(lambda x: calc_score(x, economy_kw))
    df["Weather_Score"]  = df["Content"].apply(lambda x: calc_score(x, weather_kw))
    df["Social_Score"]   = df["Content"].apply(lambda x: calc_score(x, social_kw))
    df["Logistics_Score"] = df["Content"].apply(lambda x: calc_score(x, logistics_kw))
    df["Tourism_Score"]   = df["Content"].apply(lambda x: calc_score(x, tourism_kw))

    df["Insight"] = df.apply(generate_insight, axis=1)
    return df

# =========================================================
# UI
# =========================================================
st.title("📊 News Intelligence Dashboard (RSS + CSV + ML)")

mode = st.radio("Choose Input Method:", ["RSS (Auto-Refresh)", "Upload CSV"])

# ---------------------------------------------------------
# OPTION 1 — RSS FEED
# ---------------------------------------------------------
if mode == "RSS (Auto-Refresh)":
    st.subheader("🔄 Auto-Fetching DailyMirror Breaking News (Every 10 Minutes)")
    rss_url = "https://www.dailymirror.lk/RSS_Feeds/breaking_news"

    df_rss = fetch_rss(rss_url)
    df_rss = process_dataframe(df_rss)

    st.dataframe(df_rss[["Content","Published","Sector","Insight","Link"]])

    # VISUALS
    st.subheader("📈 Sector Distribution")
    fig1 = px.bar(df_rss["Sector"].value_counts(), title="News Count per Sector")
    st.plotly_chart(fig1)

    st.subheader("🔥 Risk Heatmap")
    heat = df_rss.groupby("Sector")[["Economy_Score","Weather_Score","Social_Score","Logistics_Score","Tourism_Score"]].sum()
    fig2 = px.imshow(heat, text_auto=True, title="Risk Heatmap by Sector")
    st.plotly_chart(fig2)

# ---------------------------------------------------------
# OPTION 2 — CSV INPUT
# ---------------------------------------------------------
else:
    st.subheader("📂 Upload CSV File")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        df = process_dataframe(df)

        st.dataframe(df[["Content","Sector","Insight"]])

        # VISUALS
        st.subheader("📈 Sector Distribution")
        fig1 = px.bar(df["Sector"].value_counts(), title="News Count per Sector")
        st.plotly_chart(fig1)

        st.subheader("🔥 Risk Heatmap")
        heat = df.groupby("Sector")[["Economy_Score","Weather_Score","Social_Score","Logistics_Score","Tourism_Score"]].sum()
        fig2 = px.imshow(heat, text_auto=True, title="Risk Heatmap by Sector")
        st.plotly_chart(fig2)

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "processed_news.csv",
            "text/csv"
        )
