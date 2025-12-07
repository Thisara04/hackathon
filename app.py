import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests  
import re
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import tweepy  # NEW

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
    st.experimental_rerun()

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
    if r.get("Economy_Score",0) >= 2: insights.append("Economic risk rising")
    if r.get("Weather_Score",0) >= 1: insights.append("Weather disruption possible")
    if r.get("Social_Score",0) >= 1: insights.append("Social unrest warning")
    if r.get("Logistics_Score",0) >= 1: insights.append("Transport/Logistics alert")
    if r.get("Tourism_Score",0) >= 1: insights.append("Tourism opportunity")
    return "; ".join(insights) if insights else "Normal"

# Auto-refresh every 10 minutes
st_autorefresh(interval=10 * 60 * 1000, key="auto_refresh")

# ===================================================================
#  TWITTER API (safe mode – if rate limited → return empty dataframe)
# ===================================================================

TW_API_KEY = "PjdzqbxWlC5gJXtP4rHEmZ2wN"
TW_API_SECRET = "68XS5Q1BLd7Ne23ssgCqHWhursP2ggslnpT3j3mmo5cTyGxkA2"
TW_ACCESS_TOKEN = "1904574098656608256-cmV7U7e8B5VmJjbQ6DRXoMEE5uTPwJ"
TW_ACCESS_SECRET = "HOViVM12Ogm5k47tJ0sOPzuvHPkUPTlBKWb1rtFcCUiK4"

def fetch_twitter():
    try:
        auth = tweepy.OAuth1UserHandler(
            TW_API_KEY, TW_API_SECRET,
            TW_ACCESS_TOKEN, TW_ACCESS_SECRET
        )
        api = tweepy.API(auth, wait_on_rate_limit=True)

        tweets = api.search_tweets(
            q="Sri Lanka -filter:retweets",
            lang="en",
            count=50,
            tweet_mode="extended"
        )

        records = []
        now = datetime.now(timezone.utc)
        cutoff = now - pd.Timedelta(hours=24)

        for tw in tweets:
            created = tw.created_at.replace(tzinfo=timezone.utc)
            if created >= cutoff:
                records.append({
                    "title": tw.full_text.replace("\n", " "),
                    "link": f"https://twitter.com/user/status/{tw.id}",
                    "pubDate": created,
                    "image": "",
                    "source": "Twitter"
                })

        return pd.DataFrame(records)

    except Exception as e:
        return pd.DataFrame()

# -----------------------------
# RSS Feeds
# -----------------------------
RSS_FEEDS = [
    "https://www.dailymirror.lk/RSS_Feeds/breaking_news/108",
    "https://www.dailymirror.lk/rss/business_24_7/395",
    "https://www.dailymirror.lk/rss/top_story/155",
    "https://economynext.com/feed/",
    "https://www.news.lk/news?format=feed",
    "https://www.onlanka.com/feed"
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
            records.append({"title": title, "link": link, "pubDate": pub, "image": ""})
        return pd.DataFrame(records)
    except:
        return pd.DataFrame()

# -----------------------------
# NewsAPI
# -----------------------------
NEWSAPI_KEY = "681548c940d14836b6edbb62b1d39442"

def fetch_newsapi():
    try:
        url = f"https://newsapi.org/v2/everything?q=sri+lanka&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
        resp = requests.get(url).json()
        articles = resp.get("articles", [])
        now = datetime.now(timezone.utc)
        cutoff = now - pd.Timedelta(hours=24)
        records = []

        for art in articles:
            published = art.get("publishedAt","")
            try:
                dt = datetime.fromisoformat(published.replace("Z","")).replace(tzinfo=timezone.utc)
            except:
                dt = None
            if dt and dt >= cutoff:
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
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["pubDate"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])
    df["month"] = df["datetime"].dt.month
    df["dow"] = df["datetime"].dt.dayofweek
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)
    df["Content"] = df["title"].astype(str)
    return df

# -----------------------------
# Load cache
# -----------------------------
try:
    cache_df = pd.read_csv("news_cache.csv")
except:
    cache_df = pd.DataFrame()

# -----------------------------
# Fetch new data
# -----------------------------
new_rss = pd.concat([fetch_rss(url) for url in RSS_FEEDS], ignore_index=True)
new_newsapi = fetch_newsapi()
new_twitter = fetch_twitter()  # NEW

all_news = pd.concat([cache_df, new_rss, new_newsapi, new_twitter], ignore_index=True)
all_news.drop_duplicates(subset=["link"], inplace=True)
all_news = preprocess(all_news)
all_news.to_csv("news_cache.csv", index=False)

# -----------------------------
# ML Prediction & Scoring
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
# Filter recent data
# -----------------------------
def filter_recent(df, hours):
    if df.empty:
        return df
    now = datetime.now(timezone.utc)
    cutoff = now - pd.Timedelta(hours=hours)
    return df[df["datetime"] >= cutoff]

# ============================================================
# HOME PAGE
# ============================================================
if page == "Home":
    st.image("photo.png", width=800)
    st.title("📰 Sri Lanka News Intelligence Dashboard")
    st.write("Real-time news & social intelligence for Sri Lanka.")

    st.subheader("Quick Summary")

    last_24h = filter_recent(all_news, 24)
    last_3h = filter_recent(all_news, 3)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Articles", len(last_24h))
    col2.metric("Sectors Detected", last_24h["Sector"].nunique())
    col3.metric("Risk Alerts", (last_24h["Insight"]!="Normal").sum())

    st.markdown("**Last 3 Hours**")
    col4, col5, col6 = st.columns(3)
    col4.metric("Total Articles", len(last_3h))
    col5.metric("Sectors Detected", last_3h["Sector"].nunique())
    col6.metric("Risk Alerts", (last_3h["Insight"]!="Normal").sum())

# ============================================================
# LATEST NEWS PAGE
# ============================================================
elif page == "Latest News":
    st.title("📰 Latest News")
    st.dataframe(all_news[["datetime","Content","source","link"]])

# ============================================================
# ANALYTICS PAGE
# ============================================================
elif page == "Analytics":
    st.title("📈 Analytics & Visualizations")
    st.subheader("Sector Distribution")
    fig1 = px.bar(all_news["Sector"].value_counts(), title="News Count per Sector")
    st.plotly_chart(fig1)

    st.subheader("Risk Score Heatmap")
    heat = all_news.groupby("Sector")[["Economy_Score","Weather_Score",
                                       "Social_Score","Logistics_Score",
                                       "Tourism_Score"]].sum()
    fig2 = px.imshow(heat, text_auto=True, title="Risk Heatmap by Sector")
    st.plotly_chart(fig2)

# ============================================================
# RISK SIGNALS PAGE
# ============================================================
elif page == "Risk Signals":
    st.title("⚠️ Risk Signals & Insights")
    heat = all_news.groupby("Sector")[["Economy_Score","Weather_Score",
                                       "Social_Score","Logistics_Score",
                                       "Tourism_Score"]].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Economy Alerts", heat["Economy_Score"].sum())
    col2.metric("Weather Alerts", heat["Weather_Score"].sum())
    col3.metric("Social Alerts", heat["Social_Score"].sum())
    col4.metric("Logistics Alerts", heat["Logistics_Score"].sum())
    col5.metric("Tourism Signals", heat["Tourism_Score"].sum())

    st.subheader("Top Insights")
    st.dataframe(all_news[["Content","Sector","Insight"]])

    st.download_button(
        "Download Output CSV",
        all_news.to_csv(index=False),
        "signals_output.csv",
        mime="text/csv"
    )
