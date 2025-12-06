import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import feedparser
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =========================================================
# AUTO REFRESH EVERY 10 MINUTES
# =========================================================
# 10 minutes = 600 seconds
st.experimental_set_page_config(page_title="News Intelligence Dashboard", layout="wide")
st_autorefresh = st.experimental_memo
st_autorefresh_counter = st.experimental_rerun

st_autorefresh(interval=600000)   # 600000 ms = 10 minutes


# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_all_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    classifier = joblib.load("category_model.pkl")  # CatBoost classifier
    kmeans = joblib.load("kmeans.pkl")              # K-means clustering model
    return embedder, classifier, kmeans

embedder, classifier, kmeans = load_all_models()

# =========================================================
# SECTOR MAP
# =========================================================
sector_map = {
    0: "Energy", 1: "Logistics", 2: "Education", 3: "Health",
    4: "Finance", 5: "Government", 6: "Tourism", 7: "Agriculture",
    8: "Social", 9: "Technology", 10: "Economy", 11: "Other"
}

# =========================================================
# KEYWORDS
# =========================================================
def calc_score(text, words):
    return sum(1 for w in words if w.lower() in text.lower())

economy_kw = ['stock','rupee','inflation','currency','finance','economic']
weather_kw = ['rain','flood','storm','temperature','drought']
social_kw = ['protest','strike','crowd','community']
logistics_kw = ['traffic','accident','port','delivery','transport']
tourism_kw = ['tourism','travel','hotel','tourist','visa']


def generate_insight(r):
    ideas = []
    if r["Economy_Score"] >= 2: ideas.append("Economic risk rising")
    if r["Weather_Score"] >= 1: ideas.append("Weather disruption possible")
    if r["Social_Score"] >= 1: ideas.append("Social unrest warning")
    if r["Logistics_Score"] >= 1: ideas.append("Transport/Logistics alert")
    if r["Tourism_Score"] >= 1: ideas.append("Tourism opportunity")
    return "; ".join(ideas) if ideas else "Normal"


# =========================================================
# READ RSS FEEDS
# =========================================================
RSS_FEEDS = [
    "https://www.newsfirst.lk/feed/",
    "https://www.dailymirror.lk/rss/latest/",
    "https://economynext.com/feed/",
]

def fetch_rss_news():
    rows = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:25]:   # Take latest 25 per source
            rows.append({
                "Title": entry.title,
                "Content": entry.summary if "summary" in entry else entry.title,
                "Published": entry.published if "published" in entry else "",
            })
    return pd.DataFrame(rows)


# =========================================================
# PAGE TITLE
# =========================================================
st.title("📡 Real-Time News Intelligence Dashboard")

st.caption("Auto-updates every **10 minutes** | Powered by RSS feeds")


# =========================================================
# FETCH NEWS LIVE
# =========================================================
st.subheader("⏳ Fetching latest live news…")

raw_df = fetch_rss_news()


# =========================================================
# ADD TIME FEATURES (Month & Day of Week Encoding)
# =========================================================
raw_df["Published"] = pd.to_datetime(raw_df["Published"], errors="coerce").fillna(datetime.now())
raw_df["Month"] = raw_df["Published"].dt.month
raw_df["DOW"] = raw_df["Published"].dt.dayofweek

raw_df["Month_sin"] = np.sin(2 * np.pi * raw_df["Month"] / 12)
raw_df["Month_cos"] = np.cos(2 * np.pi * raw_df["Month"] / 12)
raw_df["DOW_sin"] = np.sin(2 * np.pi * raw_df["DOW"] / 7)
raw_df["DOW_cos"] = np.cos(2 * np.pi * raw_df["DOW"] / 7)


# =========================================================
# EMBEDDINGS
# =========================================================
st.subheader("📘 Processing text with embeddings…")

text_list = raw_df["Content"].tolist()
X_text = embedder.encode(text_list, convert_to_numpy=True)

X_time = raw_df[["Month_sin","Month_cos","DOW_sin","DOW_cos"]].to_numpy()
X = np.hstack([X_text, X_time])


# =========================================================
# CATBOOST CLASSIFICATION
# =========================================================
raw_df["SectorID"] = classifier.predict(X)
raw_df["Sector"] = raw_df["SectorID"].map(sector_map)


# =========================================================
# K-MEANS CLUSTERING
# =========================================================
raw_df["Cluster"] = kmeans.predict(X_text)


# =========================================================
# SIGNAL SCORING
# =========================================================
raw_df["Economy_Score"] = raw_df["Content"].apply(lambda x: calc_score(x, economy_kw))
raw_df["Weather_Score"] = raw_df["Content"].apply(lambda x: calc_score(x, weather_kw))
raw_df["Social_Score"] = raw_df["Content"].apply(lambda x: calc_score(x, social_kw))
raw_df["Logistics_Score"] = raw_df["Content"].apply(lambda x: calc_score(x, logistics_kw))
raw_df["Tourism_Score"] = raw_df["Content"].apply(lambda x: calc_score(x, tourism_kw))

raw_df["Insight"] = raw_df.apply(generate_insight, axis=1)


# =========================================================
# VISUALIZATIONS
# =========================================================
st.header("📊 Analytics Dashboard")

# -----------------------------
# 1. Sector Distribution
# -----------------------------
st.subheader("1️⃣ Sector Distribution")
fig1 = px.bar(
    raw_df["Sector"].value_counts(),
    title="Distribution of Latest News Across Sectors"
)
st.plotly_chart(fig1, use_container_width=True)


# -----------------------------
# 2. Cluster Distribution
# -----------------------------
st.subheader("2️⃣ News Topic Clusters")
fig_cluster = px.histogram(raw_df, x="Cluster", title="Cluster Frequency")
st.plotly_chart(fig_cluster, use_container_width=True)


# -----------------------------
# 3. Heatmap
# -----------------------------
st.subheader("3️⃣ Risk Heatmap")
heat = raw_df.groupby("Sector")[["Economy_Score","Weather_Score","Social_Score",
                                 "Logistics_Score","Tourism_Score"]].sum()

fig2 = px.imshow(heat, text_auto=True, title="Risk Heatmap by Sector")
st.plotly_chart(fig2, use_container_width=True)


# -----------------------------
# 4. Word Cloud
# -----------------------------
st.subheader("4️⃣ Word Cloud")
all_text = " ".join(raw_df["Content"].astype(str).tolist())
wc = WordCloud(width=1200, height=500).generate(all_text)

fig_wc, ax = plt.subplots(figsize=(12,6))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig_wc)


# -----------------------------
# 5. Insights Table
# -----------------------------
st.subheader("5️⃣ Insights Table")
st.dataframe(raw_df[["Title","Sector","Cluster","Insight","Published"]])


# -----------------------------
# 6. Download CSV
# -----------------------------
st.download_button(
    label="⬇ Download Processed News CSV",
    data=raw_df.to_csv(index=False),
    file_name="live_news_output.csv",
    mime="text/csv"
)
