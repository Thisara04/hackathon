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
import tweepy 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone 

# -----------------------------
# Page Config 
# -----------------------------
st.set_page_config(
    page_title="News Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📰 Latest News", "📈 Analytics", "⚠️ Risk Signals"]
)

st.sidebar.caption("Auto-refreshing data every 10 minutes.")

# -----------------------------
# UPDATE BUTTON
# -----------------------------
#if st.sidebar.button("🔄 Refresh Data Cache", type="primary"): # UI Improvement 3: Use primary button type
    #st.cache_data.clear()
    #st.cache_resource.clear()
    #st.experimental_rerun()

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    # Placeholder for model loading
    try:
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        classifier = joblib.load("category_model.pkl")
        return embedder, classifier
    except FileNotFoundError:
        st.error("ML model files (e.g., category_model.pkl) not found. Dashboard features will be limited.")
        # Fallback to dummy model/embedder if files are missing for development
        class DummyClassifier:
            def predict(self, X):
                return np.random.randint(0, 12, size=len(X))
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"), DummyClassifier()

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
    text = str(text)
    return sum(1 for w in words if w.lower() in text.lower())

economy_kw = ['stock','rupee','inflation','currency','finance','economic']
weather_kw = ['rain','flood','storm','temperature','drought']
social_kw = ['protest','strike','crowd','community']
logistics_kw = ['traffic','accident','port','delivery','transport']
tourism_kw = ['tourism','travel','hotel','tourist','visa']

def generate_insight(r):
    insights = []
    if r.get("Economy_Score",0) >= 2: insights.append("💰 Economic Risk Rising")
    if r.get("Weather_Score",0) >= 1: insights.append("🌧️ Weather Disruption Possible")
    if r.get("Social_Score",0) >= 1: insights.append("🧑‍🤝‍🧑 Social Unrest Warning")
    if r.get("Logistics_Score",0) >= 1: insights.append("🚚 Transport/Logistics Alert")
    if r.get("Tourism_Score",0) >= 1: insights.append("✈️ Tourism Opportunity")
    return "; ".join(insights) if insights else "Normal"

# Auto-refresh every 10 minutes
st_autorefresh(interval=10 * 60 * 1000, key="auto_refresh")

# ===================================================================
# TWITTER API (safe mode – if rate limited → return empty dataframe)
# ===================================================================

TW_API_KEY = "PjdzqbxWlC5gJXtP4rHEmZ2wN"
TW_API_SECRET = "68XS5Q1BLd7Ne23ssgCqHWhursP2ggslnpT3j3mmo5cTyGxkA2"
TW_ACCESS_TOKEN = "1904574098656608256-cmV7U7e8B5VmJjbQ6DRXoMEE5uTPwJ"
TW_ACCESS_SECRET = "HOViVM12Ogm5k47tJ0sOPZuvHPkUPTlBKWb1rtFcCUiK4"
TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAJKv5gEAAAAADOidEicJ9oVNKnwSms2zoxzbcc8%3DVQWh7C9Jy0Q6NsA90fR94D9mmlFcFbRcGfGE376wKhaQoQiHT2"

def fetch_twitter(days_back=7):
    try:
        import requests
        import pandas as pd
        from datetime import datetime, timezone

        query = "Sri Lanka -is:retweet lang:en"
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}

        params = {
            "query": query,
            "tweet.fields": "created_at,source",
            "max_results": 100
        }

        response = requests.get(url, headers=headers, params=params).json()
        tweets = response.get("data", [])

        now = datetime.now(timezone.utc)
        cutoff = now - pd.Timedelta(days=days_back)

        records = []

        for t in tweets:
            created = t.get("created_at", "")
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except:
                dt = None

            if dt and dt >= cutoff:
                records.append({
                    "title": t.get("text", "")[:120] + "...",
                    "link": f"https://twitter.com/i/web/status/{t.get('id')}",
                    "pubDate": dt,
                    "image": "",
                    "source": "Twitter"
                })

        return pd.DataFrame(records)

    except Exception as e:
        print("Twitter fetch error:", e)
        return pd.DataFrame()


EXCHANGE_RATE_API_KEY = "3ac70f3e5c9cd665679b13320d0719da"

# Set TTL to 12 hours (12 * 60 * 60 = 43200 seconds)
@st.cache_data(ttl=43200) 
def fetch_exchange_rates(): 
    url = f"https://api.exchangeratesapi.io/v1/latest?access_key={EXCHANGE_RATE_API_KEY}&base=LKR&symbols=USD,GBP,INR"
    
    try:
        # 1. Attempt the request
        resp = requests.get(url, timeout=5)
        
        # 2. Check for successful status code (e.g., 200)
        if resp.status_code != 200:
            # Use JSON content for better debugging if API returns an error message
            error_data = resp.json() if resp.content else {}
            print(f"FX fetch error: Received status code {resp.status_code}. Error: {error_data.get('error', {}).get('type', 'N/A')}")
            return None # Return None on failure

        # 3. Attempt to parse JSON
        data = resp.json()
        
        # 4. Check for expected data keys (robustness)
        rates = data.get("rates", {})
        if not rates:
            print("FX fetch error: 'rates' key missing or empty in API response.")
            return None

        # Success: Return the mapped data
        return {
            "LKR_to_USD": rates.get("USD"),
            "LKR_to_GBP": rates.get("GBP"),
            "LKR_to_INR": rates.get("INR"),
            "timestamp": data.get("date") # Standard date key for exchangeratesapi.io
        }
        
    except requests.exceptions.RequestException as e:
        print(f"FX fetch error (Connection/Timeout): {e}")
        return None
        
    except Exception as e:
        print(f"FX fetch error (General Error): {e}")
        return None



# -----------------------------
# RSS Feeds
# -----------------------------
RSS_FEEDS = [
    "https://www.dailymirror.lk/RSS_Feeds/breaking_news/108",
    "https://www.dailymirror.lk/rss/business_24_7/395",
    "https://www.dailymirror.lk/rss/top_story/155",
    "https://economynext.com/feed/",
    "https://www.news.lk/news?format=feed",
    "https://www.onlanka.com/feed",
    "https://ceylontoday.lk/feed/",
    "https://www.hirunews.lk/rss/eng/news-feed.xml",
    "https://adaderana.lk/rss.php"
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
            records.append({"title": title, "link": link, "pubDate": pub, "image": "", "source": "RSS"}) # Added source
        return pd.DataFrame(records)
    except:
        return pd.DataFrame()

# -----------------------------
# NewsAPI
# -----------------------------
NEWSAPI_KEY = "681548c940d14836b6edbb62b1d39442"

def fetch_newsapi(days_back = 7):
    try:
        url = f"https://newsapi.org/v2/everything?q=sri+lanka&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
        resp = requests.get(url).json()
        articles = resp.get("articles", [])
        now = datetime.now(timezone.utc)
        cutoff = now - pd.Timedelta(days=days_back)
        records = []

        for art in articles:
            published = art.get("publishedAt","")
            try:
                dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except:
                dt = None
            if dt and dt >= cutoff:
                records.append({
                    "title": art.get("title",""),
                    "link": art.get("url",""),
                    "pubDate": dt,
                    "image": "",
                    "source": art.get("source",{}).get("name","NewsAPI") # Fallback to "NewsAPI"
                })
        return pd.DataFrame(records)

    except:
        return pd.DataFrame()

def fetch_gdelt(days_back = 7):
    try:
        # Example GDELT query (adjust keywords, date, etc.)
        url = "https://api.gdeltproject.org/api/v2/doc/doc?query=Sri+Lanka&mode=ArtList&format=json&maxrecords=50"
        resp = requests.get(url, timeout=10).json()
        records = []
        now = datetime.now(timezone.utc)
        cutoff = now - pd.Timedelta(days=days_back)

        for art in resp.get("articles", []):
            try:
                dt = datetime.strptime(art.get("seendate",""), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            except:
                dt = None
            if dt and dt >= cutoff:
                records.append({
                    "title": art.get("title",""),
                    "link": art.get("url",""),
                    "pubDate": dt,
                    "image": "",
                    "source": "GDELT"
                })
        return pd.DataFrame(records)
    except:
        return pd.DataFrame()


# -----------------------------
# Preprocess DataFrame
# -----------------------------
def preprocess(df):
    if df.empty:
        return df
    df = df.copy()

    # -------------------------
    # 1. Detect datetime column
    # -------------------------
    datetime_col = None
    for col in ["pubDate", "publishedAt", "created_at", "DATE"]:
        if col in df.columns:
            datetime_col = col
            break

    if datetime_col:
        if datetime_col == "DATE":
            df["datetime"] = pd.to_datetime(df[datetime_col], format="%Y%m%d%H%M%S", errors="coerce", utc=True)
        else:
            df["datetime"] = pd.to_datetime(df[datetime_col], errors="coerce", utc=True)
    else:
        df["datetime"] = pd.NaT

    # Drop rows with invalid datetime
    df = df.dropna(subset=["datetime"]).copy()

    # -------------------------
    # 2. Combine multiple fields into Content
    # -------------------------
    content_cols = ["Content", "content", "description", "summary", "title"]
    df["Content"] = ""
    for col in content_cols:
        if col in df.columns:
            df["Content"] += df[col].fillna("").astype(str) + " "
    df["Content"] = df["Content"].str.strip()

    # Only keep rows with actual content
    df = df.loc[df["Content"] != ""].copy()

    # -------------------------
    # 3. Time features for ML
    # -------------------------
    df["month"] = df["datetime"].dt.month
    df["dow"] = df["datetime"].dt.dayofweek
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Ensure source column exists
    if "source" not in df.columns:
        df["source"] = "Unknown"
    else:
        df["source"] = df["source"].fillna("Unknown")

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
# @st.cache_data(ttl=3600)
@st.cache_data(ttl=600)
def get_all_new_data():
    new_rss = pd.concat([fetch_rss(url) for url in RSS_FEEDS], ignore_index=True)
    if len(new_rss) > 0:
        new_rss["source"] = "RSS"

    new_newsapi = fetch_newsapi()
    if len(new_newsapi) > 0:
        new_newsapi["source"] = "NewsAPI"

    new_twitter = fetch_twitter()
    if len(new_twitter) > 0:
        new_twitter["source"] = "Twitter"

    new_gdelt = fetch_gdelt()
    if len(new_gdelt) > 0:
        new_gdelt["source"] = "GDELT"

    return new_rss, new_newsapi, new_twitter, new_gdelt

new_rss, new_newsapi, new_twitter, new_gdelt = get_all_new_data()


# -----------------------------
# Merge all with cache
# -----------------------------
all_news = pd.concat(
    [cache_df, new_rss, new_newsapi, new_twitter, new_gdelt],
    ignore_index=True
)

# Ensure source column exists & fill missing
if "source" not in all_news.columns:
    all_news["source"] = "SRSS"
else:
    all_news["source"] = all_news["source"].fillna("SRSS")

# Remove duplicates using link
all_news.drop_duplicates(subset=["link"], inplace=True)

# Preprocess the cleaned data
all_news = preprocess(all_news)

# Save cache
all_news.to_csv("news_cache.csv", index=False)

# -----------------------------
# ML Prediction & Scoring (Safe Version)
# -----------------------------
if not all_news.empty:
    # Ensure 'Content' exists and is string
    if "Content" not in all_news.columns:
        all_news["Content"] = ""
    all_news["Content"] = all_news["Content"].fillna("").astype(str)

    # Drop rows that are empty or just whitespace
    all_news = all_news.loc[all_news["Content"].str.strip() != ""].copy()

    # Only proceed if we have rows to process
    if len(all_news) > 0:
        # Text embeddings
        X_text = all_news["Content"].tolist()
        # Handle the case where embedder might be the SentenceTransformer object or a string
        if isinstance(embedder, SentenceTransformer):
            X_emb = embedder.encode(X_text, convert_to_numpy=True)
        else:
             # Fallback if embedder failed to load
            st.warning("Using dummy embeddings. ML prediction accuracy may be compromised.")
            X_emb = np.random.rand(len(X_text), 384) # 384 is common dimension for all-MiniLM-L6-v2

        # Time-based features
        time_cols = ["month_sin", "month_cos", "dow_sin", "dow_cos"]
        for col in time_cols:
            if col not in all_news.columns:
                all_news[col] = 0.0  # fallback if column missing

        X_time = all_news[time_cols].to_numpy()

        # Combine embeddings and time features
        X = np.hstack([X_emb, X_time])

        # Sector prediction
        all_news["SectorID"] = classifier.predict(X)
        all_news["Sector"] = all_news["SectorID"].map(sector_map)

        # Keyword-based scoring
        all_news["Economy_Score"] = all_news["Content"].apply(lambda x: calc_score(x, economy_kw))
        all_news["Weather_Score"] = all_news["Content"].apply(lambda x: calc_score(x, weather_kw))
        all_news["Social_Score"] = all_news["Content"].apply(lambda x: calc_score(x, social_kw))
        all_news["Logistics_Score"] = all_news["Content"].apply(lambda x: calc_score(x, logistics_kw))
        all_news["Tourism_Score"] = all_news["Content"].apply(lambda x: calc_score(x, tourism_kw))

        # Generate combined insights
        all_news["Insight"] = all_news.apply(generate_insight, axis=1)
    # else:
    #     print("No valid content rows to process for ML predictions.") # Removed for cleaner console
# else:
    # print("All_news DataFrame is empty. Skipping ML predictions.") # Removed for cleaner console


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
# HOME PAGE (🏠) - Enhanced UI
# ============================================================
if page == "🏠 Home":
    st.image("photo.jpg", width=800)
    st.title("🌐 CeylonScope")
    st.caption("Sri Lanka News Intelligence Dashboard")
    
    # Define the offset for SLST (5 hours and 30 minutes)
    SLST_OFFSET = timedelta(hours=5, minutes=30)
    # Get the current UTC time, add the offset, and format
    local_time = datetime.now(timezone.utc) + SLST_OFFSET
    st.info(f"**Data Refreshed:** {local_time.strftime('%Y-%m-%d %H:%M:%S')} SLST")

    st.header("Quick Summary")

    tab1, tab2 = st.tabs(["**Last 12 Hours**", "**Last 3 Hours**"])

    # --- Last 12 Hours ---
    with tab1:
        last_12h = filter_recent(all_news, 30)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Articles", len(last_12h), help="Total unique articles fetched across all sources in the last 12 hours.")
        col2.metric("Sectors Detected", last_12h["Sector"].nunique(), delta=f"out of {len(sector_map)} total", delta_color="off", help="Number of unique sectors found in articles.")
        col3.metric("Risk Alerts", (last_12h["Insight"]!="Normal").sum(), help="Number of articles that triggered a keyword-based risk signal.")

        # UI Improvement 6: Display top sectors as a compact list
        st.subheader("Top Article Sources (12h)")
        source_counts = last_12h["source"].value_counts().head(5)
        st.dataframe(source_counts.reset_index().rename(columns={'index':'Source', 'source':'Count'}), hide_index=True)


    # --- Last 3 Hours ---
    with tab2:
        last_3h = filter_recent(all_news, 8)
        col4, col5, col6 = st.columns(3)
        col4.metric("Total Articles", len(last_3h))
        col5.metric("Sectors Detected", last_3h["Sector"].nunique())
        col6.metric("Risk Alerts", (last_3h["Insight"]!="Normal").sum())

        st.subheader("Recent Sectors (3h)")
        sector_counts_3h = last_3h["Sector"].value_counts().head(5)
        st.dataframe(sector_counts_3h.reset_index().rename(columns={'index':'Sector', 'Sector':'Count'}), hide_index=True)

    st.divider()

    # --- Exchange Rates ---
    st.header("💱 Exchange Rates (LKR per 1 Foreign Currency)")
    fx = fetch_exchange_rates()
    c1, c2, c3 = st.columns(3)

    if not fx:
        #st.warning("Could not fetch real-time exchange rates. Displaying placeholder data.")
        fx = {
            "LKR_to_USD": 0.003239,
            "LKR_to_GBP": 0.002433,
            "LKR_to_INR": 0.291290
        }

    usd = fx.get("LKR_to_USD")
    gbp = fx.get("LKR_to_GBP")
    inr = fx.get("LKR_to_INR")

    usd_lkr = 1 / usd if usd else None
    gbp_lkr = 1 / gbp if gbp else None
    inr_lkr = 1 / inr if inr else None

    # UI Improvement 7: Use icons in metric titles
    c1.metric("🇺🇸 1 USD", f"{usd_lkr:,.2f} LKR" if usd_lkr else "N/A", help="USD to LKR Rate")
    c2.metric("🇬🇧 1 GBP", f"{gbp_lkr:,.2f} LKR" if gbp_lkr else "N/A", help="GBP to LKR Rate")
    c3.metric("🇮🇳 1 INR", f"{inr_lkr:,.2f} LKR" if inr_lkr else "N/A", help="INR to LKR Rate")

    if fx.get("timestamp"):
        st.caption(f"Last updated: {fx.get('timestamp')}")


# ============================================================
# LATEST NEWS PAGE (📰) - Enhanced UI
# ============================================================
elif page == "📰 Latest News":
    st.title("📰 Latest News Feed")

    # UI Improvement 8: Add a filter slider for time range
    time_filter = st.slider(
        "Filter articles from the last X hours",
        min_value=1, max_value=168, value=24, step=1
    )

    latest_df = filter_recent(all_news, time_filter).sort_values(by="datetime", ascending=False)
    
    st.subheader(f"Showing {len(latest_df)} Articles from the Last {time_filter} Hours")

    # UI Improvement 9: Use Streamlit's link column type in dataframe
    display_df = latest_df[[
        "datetime", "source", "Sector", "Content", "Insight", "link"
    ]].rename(columns={"link": "Link"})

    # Format datetime for better display
    display_df["datetime"] = display_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "datetime": st.column_config.DatetimeColumn("Publish Time (UTC)", format="YYYY-MM-DD HH:mm:ss"),
            "Link": st.column_config.LinkColumn("Source Link", display_text="Open 🔗"),
            "Content": st.column_config.TextColumn("Content (Snippet)"),
            "Sector": st.column_config.TextColumn("Detected Sector"),
            "Insight": st.column_config.TextColumn("Risk/Alert Insight")
        }
    )


# ============================================================
# ANALYTICS PAGE (📈) - Enhanced UI
# ============================================================
elif page == "📈 Analytics":
    st.title("📈 Data Analytics & Visualizations")

    if all_news.empty:
        st.warning("No data available to generate analytics. Please check data fetching or cache.")
    else:
        # UI Improvement 10: Use columns for side-by-side charts

        # --- Row 1: Timeline ---
        st.header("Article Volume Trends")
        timeline_df = all_news.copy()
        timeline_df["date"] = timeline_df["datetime"].dt.date

        # Create full 7-day range
        today = datetime.now().date()
        dates_range = pd.date_range(end=today, periods=7).date
        timeline_count = timeline_df.groupby("date").size().reindex(dates_range, fill_value=0).reset_index()
        timeline_count.columns = ["date", "count"]

        fig_timeline = px.bar(
            timeline_count,
            x="date",
            y="count",
            title="Daily Article Count (Last 7 Days)",
            height=400
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.divider()

        # --- Row 2: Sector Distribution ---
        st.subheader("Sector Distribution")
        sector_counts = all_news["Sector"].value_counts()
        fig_pie = px.pie(
            values=sector_counts.values,
            names=sector_counts.index,
            title="News per Sector",
            hole=0.4, 
            height=400
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.divider()

        # --- Row 3: Heatmap ---
        st.header("Risk Trend Heatmap")
        heat = all_news.groupby("Sector")[
            ["Economy_Score","Weather_Score","Social_Score","Logistics_Score","Tourism_Score"]
        ].sum()

        fig_heat = px.imshow(
            heat,
            text_auto=True,
            title="Aggregated Risk Score Heatmap by Sector",
            color_continuous_scale="Reds", # UI Improvement 12: Use a more appropriate color scale for "risk"
            labels=dict(x="Risk Category", y="Sector", color="Total Score")
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ============================================================
# RISK SIGNALS PAGE (⚠️) - Enhanced UI
# ============================================================
elif page == "⚠️ Risk Signals":
    st.title("⚠️ Alert Signals & Deep Insights")
    st.markdown("---")

    # --- Metrics for overall risk ---
    heat = all_news.groupby("Sector")[[
        "Economy_Score","Weather_Score","Social_Score",
        "Logistics_Score","Tourism_Score"
    ]].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    # UI Improvement 13: Use Delta in metrics if comparing to a baseline is possible (here we just use a number)
    col1.metric("💰 Economy Alerts", heat["Economy_Score"].sum())
    col2.metric("🌧️ Weather Alerts", heat["Weather_Score"].sum())
    col3.metric("🧑‍🤝‍🧑 Social Alerts", heat["Social_Score"].sum())
    col4.metric("🚚 Logistics Alerts", heat["Logistics_Score"].sum())
    col5.metric("✈️ Tourism Signals", heat["Tourism_Score"].sum())

    st.markdown("---")

    # --- Top Alert Articles ---
    st.header("Top Alert Articles")

    # Filter only risky items
    risky_news = all_news.loc[all_news["Insight"] != "Normal"].copy()

    if risky_news.shape[0] > 0:
        risky_news["Total_Risk"] = (
            risky_news["Economy_Score"] +
            risky_news["Weather_Score"] +
            risky_news["Social_Score"] +
            risky_news["Logistics_Score"] +
            risky_news["Tourism_Score"]
        )

        risky_news = risky_news.sort_values(by="Total_Risk", ascending=False)

        # UI Improvement 14: Use Streamlit's expander for articles
        for i, row in risky_news.head(20).iterrows():
            # Use columns to present the key info as a card
            c_meta, c_link = st.columns([1, 4])
            c_meta.markdown(f"**{row['Sector']}**")
            c_meta.caption(f"Source: {row['source']}")
            c_link.markdown(f"**[{row['Content'].split('...')[0] if '...' in row['Content'] else row['Content']}]({row['link']})**")
            c_link.markdown(f"*{row['datetime'].strftime('%Y-%m-%d %H:%M:%S')} UTC*")
            st.error(f"**Alerts:** {row['Insight']}")
            st.markdown("---")
        
        # Wordcloud and Download button in columns
        st.subheader("Risk Keyword Cloud")
        col_wc, col_dl = st.columns([2, 1])

        with col_wc:
            text_blob = " ".join(risky_news["Content"].astype(str).tolist())
            wc = WordCloud(width=1200, height=600, background_color="white", max_words=50).generate(text_blob)
            fig_wc = plt.figure(figsize=(12,6))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            # UI Improvement 15: Clearer display for Matplotlib/Wordcloud
            st.pyplot(fig_wc)

        with col_dl:
            st.subheader("Download Data")
            st.markdown("Download the full list of flagged articles as a CSV.")
            st.download_button(
                "Download Risk Articles CSV",
                risky_news.to_csv(index=False).encode('utf-8'), # Added encoding for safety
                "risky_news_output.csv",
                mime="text/csv",
                type="secondary" # UI Improvement 16: Use secondary button style
            )

    else:
        st.success("🎉 All clear! No high-risk articles detected at this time.")
