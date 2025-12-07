import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import re
from datetime import datetime
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import plotly.express as px

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="News Intelligence Dashboard", layout="wide")

# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Latest News", "Analytics", "Risk Signals"]
)

# -----------------------------
# UPDATE BUTTON
# -----------------------------
if st.sidebar.button("🔄 Update Now"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()  # New method for Streamlit >= 1.18

# -----------------------------
# Load Models
# -----------------------------
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

# -----------------------------
# Keyword scoring
# -----------------------------
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

# -----------------------------
# RSS Fetching
# -----------------------------
RSS_FEEDS = [
    "https://www.dailymirror.lk/RSS_Feeds/breaking_news",
    "https://www.dailymirror.lk/rss/business_24_7/395",
    "https://www.dailymirror.lk/rss/top_story/155"
]

def clean_rss_xml(text):
    text = re.sub(r"&(?!(amp;|lt;|gt;|quot;|apos;))", "&amp;", text)
    text = text.replace("]]> ]]>","")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7F]+", " ", text)
    return text

def fetch_rss(url):
    try:
        resp = requests.get(url, timeout=10)
        cleaned = clean_rss_xml(resp.text)
        soup = BeautifulSoup(cleaned, "xml")
        items = soup.find_all("item")
        records = []
        for it in items:
            title = it.title.text.strip() if it.title else ""
            link = it.link.text.strip() if it.link else ""
            pub = it.pubDate.text.strip() if it.pubDate else ""
            img = ""
            enclosure = it.find("enclosure")
            if enclosure and enclosure.get("url"):
                img = enclosure.get("url")
            records.append({"title": title, "link": link, "pubDate": pub, "image": img})
        return pd.DataFrame(records)
    except:
        return pd.DataFrame()

# -----------------------------
# NewsAPI Fetching
# -----------------------------
NEWSAPI_KEY = "681548c940d14836b6edbb62b1d39442"

def fetch_newsapi():
    url = f"https://newsapi.org/v2/everything?q=sri+lanka&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    try:
        resp = requests.get(url).json()
        articles = resp.get("articles", [])
        records = []
        for art in articles:
            published = art.get("publishedAt","")
            try:
                dt = datetime.fromisoformat(published.replace("Z",""))
            except:
                dt = None
            records.append({
                "title": art.get("title",""),
                "link": art.get("url",""),
                "pubDate": dt,
                "image": "",
                "source": art.get("source",{}).get("name","")
            })
        return pd.DataFrame(records)
    except:
        return pd.DataFrame()

# -----------------------------
# Preprocess
# -----------------------------
def preprocess(df):
    if df.empty: return df
    def parse_date(x):
        try: return pd.to_datetime(x)
        except: return None
    df["datetime"] = df["pubDate"].apply(parse_date)
    df = df.dropna(subset=["datetime"])
    df["month"] = df["datetime"].dt.month
    df["dow"] = df["datetime"].dt.dayofweek
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)
    df["Content"] = df["title"]
    return df

# -----------------------------
# Load Existing Data
# -----------------------------
try:
    cache_df = pd.read_csv("news_cache.csv")
except:
    cache_df = pd.DataFrame()

# -----------------------------
# Fetch New Data
# -----------------------------
new_rss = pd.concat([fetch_rss(url) for url in RSS_FEEDS], ignore_index=True)
new_api = fetch_newsapi()
all_news = pd.concat([cache_df, new_rss, new_api], ignore_index=True)
all_news.drop_duplicates(subset=["link"], inplace=True)
all_news = preprocess(all_news)

# Save cache
all_news.to_csv("news_cache.csv", index=False)

# -----------------------------
# Apply ML Predictions & Scoring
# -----------------------------
if not all_news.empty:
    X_text = all_news["Content"].tolist()
    X_emb = embedder.encode(X_text, convert_to_numpy=True)
    X_time = all_news[["month_sin","month_cos","dow_sin","dow_cos"]].to_numpy()
    X = np.hstack([X_emb, X_time])
    all_news["SectorID"] = classifier.predict(X)
    all_news["Sector"] = all_news["SectorID"].map(sector_map)

    all_news["Economy_Score"] = all_news["Content"].apply(lambda x: calc_score(x, economy_kw))
    all_news["Weather_Score"] = all_news["Content"].apply(lambda x: calc_score(x, weather_kw))
    all_news["Social_Score"] = all_news["Content"].apply(lambda x: calc_score(x, social_kw))
    all_news["Logistics_Score"] = all_news["Content"].apply(lambda x: calc_score(x, logistics_kw))
    all_news["Tourism_Score"] = all_news["Content"].apply(lambda x: calc_score(x, tourism_kw))
    all_news["Insight"] = all_news.apply(generate_insight, axis=1)

# -----------------------------
# Function to filter recent news
# -----------------------------
def filter_recent(df, hours=12):
    if df.empty: return df
    now = datetime.now()
    cutoff = now - pd.Timedelta(hours=hours)
    return df[df["datetime"] >= cutoff]

# ============================================================
# PAGE 1 — HOME
# ============================================================
if page == "Home":

    st.image("photo.png", width=800)  # ← resized photo

    st.title("📰 Sri Lanka News Intelligence Dashboard")

    st.write("""
    Welcome to the automated **real-time news intelligence system** for Sri Lanka.
    This dashboard fetches, classifies, scores, and analyzes news in multiple sectors.
    """)

    st.subheader("How the System Works")
    st.markdown("""
    - Fetches **real-time news** from RSS + NewsAPI  
    - Cleans and normalizes article content  
    - Converts text into **MiniLM embeddings**  
    - Classifies articles into **12 sectors** using ML  
    - Detects **risk signals** (Economy, Weather, Social, Logistics, Tourism)  
    - Generates insights and visual summaries  
    """)

    st.info(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if not all_news.empty:
        st.subheader("Quick Summary")

        # Filter news for last 12 hours and 4 hours
        now = datetime.now()
        last_12h = all_news[all_news["datetime"] >= now - pd.Timedelta(hours=12)]
        last_4h = all_news[all_news["datetime"] >= now - pd.Timedelta(hours=4)]

        st.markdown("**Last 12 Hours**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Articles", len(last_12h))
        col2.metric("Sectors Detected", last_12h["Sector"].nunique())
        col3.metric("Risk Alerts", (last_12h["Insight"] != "Normal").sum())

        st.markdown("**Last 4 Hours**")
        col4, col5, col6 = st.columns(3)
        col4.metric("Total Articles", len(last_4h))
        col5.metric("Sectors Detected", last_4h["Sector"].nunique())
        col6.metric("Risk Alerts", (last_4h["Insight"] != "Normal").sum())


# ============================================================
# PAGE 2 — LATEST NEWS
# ============================================================
elif page == "Latest News":

    st.title("📰 Latest News")

    # Time filter
    time_range = st.radio("Select time range:", ["Last 12 hours", "Last 4 hours"])
    hours = 12 if time_range == "Last 12 hours" else 4
    filtered_news = filter_recent(all_news, hours=hours)

    st.dataframe(filtered_news[["datetime","Content","link"]], use_container_width=True)

# ============================================================
# PAGE 3 — ANALYTICS
# ============================================================
elif page == "Analytics":

    st.title("📈 Analytics & Visualizations")

    st.subheader("Sector Distribution")
    fig1 = px.bar(all_news["Sector"].value_counts(), title="News Count per Sector")
    st.plotly_chart(fig1)

    st.subheader("Risk Score Heatmap")
    heat = all_news.groupby("Sector")[["Economy_Score","Weather_Score","Social_Score","Logistics_Score","Tourism_Score"]].sum()
    fig2 = px.imshow(heat, text_auto=True, title="Risk Heatmap by Sector")
    st.plotly_chart(fig2)

# ============================================================
# PAGE 4 — RISK SIGNALS
# ============================================================
elif page == "Risk Signals":

    st.title("⚠️ Risk Signals & Insights")

    # Time filter
    time_range = st.radio("Select time range:", ["Last 12 hours", "Last 4 hours"])
    hours = 12 if time_range == "Last 12 hours" else 4
    filtered_news = filter_recent(all_news, hours=hours)

    heat = filtered_news.groupby("Sector")[["Economy_Score","Weather_Score","Social_Score",
                                            "Logistics_Score","Tourism_Score"]].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Economy Alerts", heat["Economy_Score"].sum())
    col2.metric("Weather Alerts", heat["Weather_Score"].sum())
    col3.metric("Social Alerts", heat["Social_Score"].sum())
    col4.metric("Logistics Alerts", heat["Logistics_Score"].sum())
    col5.metric("Tourism Signals", heat["Tourism_Score"].sum())

    st.subheader("Top Insights")
    st.dataframe(filtered_news[["Content","Sector","Insight"]])

    st.download_button(
        label="Download Output CSV",
        data=filtered_news.to_csv(index=False),
        file_name="signals_output.csv",
        mime="text/csv"
    )
