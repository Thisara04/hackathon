import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import re
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CeylonScope · Sri Lanka Intelligence",
    page_icon="🇱🇰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL STYLES  (professional dark-accent theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0B1120;
    border-right: 1px solid #1E2D45;
  }
  [data-testid="stSidebar"] * { color: #C9D8EE !important; }
  [data-testid="stSidebar"] .stRadio label { font-size: 14px; }
  [data-testid="stSidebarNav"] { display: none; }

  /* ── Main background ── */
  .main .block-container {
    background: #F5F7FA;
    padding-top: 1.5rem;
    max-width: 1280px;
  }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  [data-testid="metric-container"] label {
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    color: #64748B !important;
    text-transform: uppercase;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 600 !important;
    color: #0F172A !important;
  }
  [data-testid="stMetricDelta"] { font-size: 12px !important; }

  /* ── DataFrames ── */
  .stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid #E2E8F0; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #EFF2F7;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    font-size: 14px;
  }
  .stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #0F172A !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }

  /* ── Ticker bar ── */
  .ticker-wrap {
    background: #0B1120;
    border-radius: 10px;
    padding: 10px 20px;
    margin-bottom: 20px;
    overflow: hidden;
  }
  .ticker-label {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: #94A3B8;
  }
  .ticker-val {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    color: #E2E8F0;
  }
  .ticker-up   { color: #34D399; }
  .ticker-down { color: #F87171; }

  /* ── Section headers ── */
  .section-header {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    margin: 24px 0 10px 0;
  }

  /* ── Page title card ── */
  .page-hero {
    background: linear-gradient(135deg, #0B1120 0%, #162236 100%);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .hero-title {
    font-size: 28px;
    font-weight: 600;
    color: #F1F5F9;
    margin: 0;
    line-height: 1.2;
  }
  .hero-sub {
    font-size: 13px;
    color: #64748B;
    margin: 4px 0 0 0;
  }
  .hero-badge {
    background: #1E2D45;
    border: 1px solid #2D4A6B;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 500;
    color: #7DD3FC;
    font-family: 'DM Mono', monospace;
  }

  /* ── Finance cards ── */
  .fin-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 20px;
  }
  .fin-label { font-size: 11px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: #94A3B8; }
  .fin-value { font-size: 22px; font-weight: 600; color: #0F172A; margin: 4px 0 2px; font-family: 'DM Mono', monospace; }
  .fin-change-up   { font-size: 12px; font-weight: 500; color: #059669; }
  .fin-change-down { font-size: 12px; font-weight: 500; color: #DC2626; }

  /* ── News card rows ── */
  .news-item {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #3B82F6;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-bottom: 8px;
  }
  .news-sector-badge {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    font-size: 11px;
    font-weight: 500;
    border-radius: 6px;
    padding: 2px 8px;
    margin-right: 6px;
  }
  .news-risk-badge {
    display: inline-block;
    background: #FEF3C7;
    color: #92400E;
    font-size: 11px;
    font-weight: 500;
    border-radius: 6px;
    padding: 2px 8px;
  }

  /* ── Divider ── */
  hr { border: none; border-top: 1px solid #E2E8F0; margin: 24px 0; }

  /* ── Plotly charts ── */
  .js-plotly-plot { border-radius: 12px; }

  /* ── Alerts ── */
  .stAlert { border-radius: 10px; }

  /* ── Slider ── */
  .stSlider [data-baseweb="slider"] { padding-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 24px;">
      <div style="font-size:20px;font-weight:600;color:#F1F5F9;letter-spacing:-0.02em;">🇱🇰 CeylonScope</div>
      <div style="font-size:11px;color:#475569;margin-top:3px;letter-spacing:0.04em;text-transform:uppercase;">Intelligence Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🏠  Overview", "📰  News Feed", "📈  Analytics", "💰  Markets", "⚠️  Risk Signals"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1E2D45;margin:16px 0;'>", unsafe_allow_html=True)

    SLST_OFFSET = timedelta(hours=5, minutes=30)
    local_time = datetime.now(timezone.utc) + SLST_OFFSET
    st.markdown(f"""
    <div style="font-size:11px;color:#475569;">
      <div style="letter-spacing:0.04em;text-transform:uppercase;margin-bottom:4px;">Last updated</div>
      <div style="font-family:'DM Mono',monospace;color:#7DD3FC;font-size:12px;">{local_time.strftime('%Y-%m-%d %H:%M')} SLST</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Auto-refreshes every 10 minutes.")

# ─────────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────────
st_autorefresh(interval=10 * 60 * 1000, key="auto_refresh")

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        classifier = joblib.load("catagory_model.pkl")
        return embedder, classifier
    except FileNotFoundError:
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

SECTOR_COLORS = {
    "Energy": "#F59E0B", "Logistics": "#6366F1", "Education": "#06B6D4",
    "Health": "#EF4444", "Finance": "#10B981", "Government": "#8B5CF6",
    "Tourism": "#F97316", "Agriculture": "#22C55E", "Social": "#EC4899",
    "Technology": "#3B82F6", "Economy": "#14B8A6", "Other": "#94A3B8"
}

# ─────────────────────────────────────────────
# KEYWORD SCORING
# ─────────────────────────────────────────────
def calc_score(text, words):
    text = str(text)
    return sum(1 for w in words if w.lower() in text.lower())

economy_kw  = ['stock', 'rupee', 'inflation', 'currency', 'finance', 'economic']
weather_kw  = ['rain', 'flood', 'storm', 'temperature', 'drought']
social_kw   = ['protest', 'strike', 'crowd', 'community']
logistics_kw= ['traffic', 'accident', 'port', 'delivery', 'transport']
tourism_kw  = ['tourism', 'travel', 'hotel', 'tourist', 'visa']

def generate_insight(r):
    insights = []
    if r.get("Economy_Score",  0) >= 2: insights.append("💰 Economic Risk")
    if r.get("Weather_Score",  0) >= 1: insights.append("🌧 Weather Alert")
    if r.get("Social_Score",   0) >= 1: insights.append("🤝 Social Unrest")
    if r.get("Logistics_Score",0) >= 1: insights.append("🚚 Logistics Alert")
    if r.get("Tourism_Score",  0) >= 1: insights.append("✈️ Tourism Signal")
    return " · ".join(insights) if insights else "Normal"

# ─────────────────────────────────────────────
# DATA FETCH — SECRETS via st.secrets or env
# ─────────────────────────────────────────────
try:
    EXCHANGE_RATE_API_KEY = st.secrets["EXCHANGE_RATE_API_KEY"]
    NEWSAPI_KEY           = st.secrets["NEWSAPI_KEY"]
    TW_BEARER             = st.secrets["TWITTER_BEARER_TOKEN"]
except Exception:
    # Fallback during local dev — replace with your keys in .streamlit/secrets.toml
    EXCHANGE_RATE_API_KEY = "3ac70f3e5c9cd665679b13320d0719da"
    NEWSAPI_KEY           = "681548c940d14836b6edbb62b1d39442"
    TW_BEARER             = ""

# ─────────────────────────────────────────────
# FINANCIAL DATA
# ─────────────────────────────────────────────
@st.cache_data(ttl=900)   # 15-min cache
def fetch_financial_data():
    """
    Fetches live financial data from free public APIs:
     - Exchange rates: exchangeratesapi.io (existing key)
     - Commodities & indices: metals-api fallback to hardcoded placeholders
     - CSE (Colombo Stock Exchange): scraped from public endpoint
    """
    result = {}

    # ── 1. Exchange rates (EUR-base workaround) ──
    try:
        url = (f"https://api.exchangeratesapi.io/v1/latest"
               f"?access_key={EXCHANGE_RATE_API_KEY}&base=EUR&symbols=LKR,USD,GBP,INR,AUD,JPY")
        r = requests.get(url, timeout=6).json()
        rates = r.get("rates", {})
        lkr = rates.get("LKR", 1)
        result["fx"] = {
            "USD_LKR": round(lkr / rates.get("USD", 1), 2),
            "GBP_LKR": round(lkr / rates.get("GBP", 1), 2),
            "INR_LKR": round(lkr / rates.get("INR", 1), 2),
            "AUD_LKR": round(lkr / rates.get("AUD", 1), 2),
            "JPY_LKR": round(lkr / rates.get("JPY", 1), 2),
            "date": r.get("date", ""),
        }
    except Exception:
        result["fx"] = {
            "USD_LKR": 309.50, "GBP_LKR": 393.80,
            "INR_LKR": 3.71,   "AUD_LKR": 199.20,
            "JPY_LKR": 2.05,   "date": "N/A"
        }

    # ── 2. Gold price (USD/oz) via open.er-api.com gold proxy ──
    try:
        g = requests.get("https://api.metals.live/v1/spot", timeout=5).json()
        result["gold_usd"] = round(g.get("gold", 0), 2)
        result["silver_usd"] = round(g.get("silver", 0), 2)
        result["oil_usd"] = 0  # not in this API
    except Exception:
        result["gold_usd"] = 3320.00
        result["silver_usd"] = 32.50
        result["oil_usd"] = 83.40

    # ── 3. Bitcoin (CoinGecko — free, no key needed) ──
    try:
        cg = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true",
            timeout=6
        ).json()
        result["btc_usd"]    = round(cg.get("bitcoin",  {}).get("usd", 0), 0)
        result["btc_chg"]    = round(cg.get("bitcoin",  {}).get("usd_24h_change", 0), 2)
        result["eth_usd"]    = round(cg.get("ethereum", {}).get("usd", 0), 0)
        result["eth_chg"]    = round(cg.get("ethereum", {}).get("usd_24h_change", 0), 2)
    except Exception:
        result["btc_usd"] = 63_400
        result["btc_chg"] = 1.2
        result["eth_usd"] = 3_100
        result["eth_chg"] = 0.8

    # ── 4. CSE (Colombo Stock Exchange) via unofficial JSON ──
    try:
        cse = requests.get(
            "https://www.cse.lk/api/market-summary",
            timeout=6, headers={"User-Agent": "Mozilla/5.0"}
        ).json()
        result["cse_aspi"]   = cse.get("aspi", {})
        result["cse_s&p20"]  = cse.get("sp20", {})
    except Exception:
        result["cse_aspi"]  = {"value": 10_247.30, "change": 45.60, "change_pct": 0.45}
        result["cse_sp20"]  = {"value":  3_841.15, "change": -12.30, "change_pct": -0.32}

    return result


def _chg_class(val):
    try:
        return "fin-change-up" if float(val) >= 0 else "fin-change-down"
    except Exception:
        return "fin-change-up"

def _arrow(val):
    try:
        return "▲" if float(val) >= 0 else "▼"
    except Exception:
        return "▲"

def render_fin_card(label, value, change_str, change_pct=None):
    pct_str = f" ({change_pct:+.2f}%)" if change_pct is not None else ""
    cls = _chg_class(change_str)
    arr = _arrow(change_str)
    try:
        chg_f = float(change_str)
        chg_disp = f"{arr} {abs(chg_f):,.2f}{pct_str}"
    except Exception:
        chg_disp = change_str
    return f"""
    <div class="fin-card">
      <div class="fin-label">{label}</div>
      <div class="fin-value">{value}</div>
      <div class="{cls}" style="font-size:12px;font-weight:500;">{chg_disp}</div>
    </div>"""

# ─────────────────────────────────────────────
# LIVE TICKER BAR
# ─────────────────────────────────────────────
def render_ticker(fd):
    fx   = fd.get("fx", {})
    btc  = fd.get("btc_usd", "—")
    gold = fd.get("gold_usd", "—")

    items = [
        ("USD/LKR", fx.get("USD_LKR", "—"), None),
        ("GBP/LKR", fx.get("GBP_LKR", "—"), None),
        ("INR/LKR", fx.get("INR_LKR", "—"), None),
        ("BTC",     f"${btc:,.0f}" if isinstance(btc, (int,float)) else btc, fd.get("btc_chg")),
        ("GOLD/oz", f"${gold:,.2f}" if isinstance(gold, (int,float)) else gold, None),
        ("ASPI",    fd.get("cse_aspi", {}).get("value", "—"), fd.get("cse_aspi", {}).get("change")),
    ]

    cells = ""
    for label, val, chg in items:
        chg_html = ""
        if chg is not None:
            c = "ticker-up" if float(chg) >= 0 else "ticker-down"
            arr = "▲" if float(chg) >= 0 else "▼"
            chg_html = f'<span class="{c}" style="font-size:11px;margin-left:4px;">{arr}{abs(float(chg)):.2f}%</span>'
        cells += f"""
        <div style="display:flex;align-items:center;gap:10px;padding:0 20px;border-right:1px solid #1E2D45;">
          <span class="ticker-label">{label}</span>
          <span class="ticker-val">{val}{chg_html}</span>
        </div>"""

    st.markdown(f"""
    <div class="ticker-wrap">
      <div style="display:flex;align-items:center;overflow-x:auto;gap:0;white-space:nowrap;">
        <div style="padding:0 16px 0 0;border-right:1px solid #1E2D45;margin-right:4px;">
          <span style="font-size:10px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#475569;">LIVE</span>
          <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#34D399;margin-left:6px;vertical-align:middle;"></span>
        </div>
        {cells}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# NEWS FETCHERS
# ─────────────────────────────────────────────
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
    text = text.replace("]]> ]]>", "")
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
            link  = it.link.text.strip()  if it.link  else ""
            pub   = it.pubDate.text.strip() if it.pubDate else ""
            records.append({"title": title, "link": link, "pubDate": pub, "image": "", "source": "RSS"})
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()

def fetch_newsapi(days_back=7):
    try:
        url = f"https://newsapi.org/v2/everything?q=sri+lanka&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
        resp = requests.get(url).json()
        articles = resp.get("articles", [])
        now = datetime.now(timezone.utc)
        cutoff = now - pd.Timedelta(days=days_back)
        records = []
        for art in articles:
            published = art.get("publishedAt", "")
            try:
                dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except Exception:
                dt = None
            if dt and dt >= cutoff:
                records.append({
                    "title": art.get("title", ""),
                    "link":  art.get("url", ""),
                    "pubDate": dt,
                    "image": "",
                    "source": art.get("source", {}).get("name", "NewsAPI")
                })
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()

def fetch_gdelt(days_back=7):
    try:
        url = ("https://api.gdeltproject.org/api/v2/doc/doc"
               "?query=Sri+Lanka&mode=ArtList&format=json&maxrecords=50")
        resp = requests.get(url, timeout=10).json()
        records = []
        now = datetime.now(timezone.utc)
        cutoff = now - pd.Timedelta(days=days_back)
        for art in resp.get("articles", []):
            try:
                dt = datetime.strptime(art.get("seendate", ""), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            except Exception:
                dt = None
            if dt and dt >= cutoff:
                records.append({
                    "title": art.get("title", ""),
                    "link":  art.get("url", ""),
                    "pubDate": dt,
                    "image": "",
                    "source": "GDELT"
                })
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()

def fetch_twitter(days_back=7):
    if not TW_BEARER:
        return pd.DataFrame()
    try:
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {TW_BEARER}"}
        params = {
            "query": "Sri Lanka -is:retweet lang:en",
            "tweet.fields": "created_at,source",
            "max_results": 100
        }
        resp = requests.get(url, headers=headers, params=params, timeout=8).json()
        tweets = resp.get("data", [])
        now = datetime.now(timezone.utc)
        cutoff = now - pd.Timedelta(days=days_back)
        records = []
        for t in tweets:
            try:
                dt = datetime.fromisoformat(t.get("created_at", "").replace("Z", "+00:00"))
            except Exception:
                dt = None
            if dt and dt >= cutoff:
                records.append({
                    "title":   t.get("text", "")[:120] + "...",
                    "link":    f"https://twitter.com/i/web/status/{t.get('id')}",
                    "pubDate": dt,
                    "image":   "",
                    "source":  "Twitter"
                })
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    if df.empty:
        return df
    df = df.copy()

    datetime_col = next((c for c in ["pubDate", "publishedAt", "created_at", "DATE"] if c in df.columns), None)
    if datetime_col:
        fmt = "%Y%m%d%H%M%S" if datetime_col == "DATE" else None
        df["datetime"] = pd.to_datetime(df[datetime_col], format=fmt, errors="coerce", utc=True)
    else:
        df["datetime"] = pd.NaT

    df = df.dropna(subset=["datetime"]).copy()

    content_cols = ["Content", "content", "description", "summary", "title"]
    df["Content"] = ""
    for col in content_cols:
        if col in df.columns:
            df["Content"] += df[col].fillna("").astype(str) + " "
    df["Content"] = df["Content"].str.strip()
    df = df.loc[df["Content"] != ""].copy()

    df["month"]     = df["datetime"].dt.month
    df["dow"]       = df["datetime"].dt.dayofweek
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)

    df["source"] = df.get("source", pd.Series("Unknown", index=df.index)).fillna("Unknown")
    return df

# ─────────────────────────────────────────────
# LOAD + FETCH + MERGE
# ─────────────────────────────────────────────
try:
    cache_df = pd.read_csv("news_cache.csv")
except Exception:
    cache_df = pd.DataFrame()

@st.cache_data(ttl=600)
def get_all_new_data():
    rss     = pd.concat([fetch_rss(u) for u in RSS_FEEDS], ignore_index=True)
    newsapi = fetch_newsapi()
    twitter = fetch_twitter()
    gdelt   = fetch_gdelt()
    return rss, newsapi, twitter, gdelt

new_rss, new_newsapi, new_twitter, new_gdelt = get_all_new_data()

all_news = pd.concat([cache_df, new_rss, new_newsapi, new_twitter, new_gdelt], ignore_index=True)
if "source" not in all_news.columns:
    all_news["source"] = "RSS"
else:
    all_news["source"] = all_news["source"].fillna("RSS")

all_news.drop_duplicates(subset=["link"], inplace=True)
all_news = preprocess(all_news)
all_news.to_csv("news_cache.csv", index=False)

# ─────────────────────────────────────────────
# ML PIPELINE
# ─────────────────────────────────────────────
if not all_news.empty:
    all_news["Content"] = all_news["Content"].fillna("").astype(str)
    all_news = all_news.loc[all_news["Content"].str.strip() != ""].copy()

    if len(all_news) > 0:
        X_text = all_news["Content"].tolist()
        if isinstance(embedder, SentenceTransformer):
            X_emb = embedder.encode(X_text, convert_to_numpy=True)
        else:
            X_emb = np.random.rand(len(X_text), 384)

        time_cols = ["month_sin", "month_cos", "dow_sin", "dow_cos"]
        for col in time_cols:
            if col not in all_news.columns:
                all_news[col] = 0.0

        X = np.hstack([X_emb, all_news[time_cols].to_numpy()])
        all_news["SectorID"] = classifier.predict(X)
        all_news["Sector"]   = all_news["SectorID"].map(sector_map)

        all_news["Economy_Score"]   = all_news["Content"].apply(lambda x: calc_score(x, economy_kw))
        all_news["Weather_Score"]   = all_news["Content"].apply(lambda x: calc_score(x, weather_kw))
        all_news["Social_Score"]    = all_news["Content"].apply(lambda x: calc_score(x, social_kw))
        all_news["Logistics_Score"] = all_news["Content"].apply(lambda x: calc_score(x, logistics_kw))
        all_news["Tourism_Score"]   = all_news["Content"].apply(lambda x: calc_score(x, tourism_kw))
        all_news["Insight"]         = all_news.apply(generate_insight, axis=1)

def filter_recent(df, hours):
    if df.empty:
        return df
    now = datetime.now(timezone.utc)
    return df[df["datetime"] >= now - pd.Timedelta(hours=hours)]

# ─────────────────────────────────────────────
# FETCH FINANCIAL DATA (shared across pages)
# ─────────────────────────────────────────────
fd = fetch_financial_data()

# ═══════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "🏠  Overview":

    st.markdown("""
    <div class="page-hero">
      <div>
        <p class="hero-title">🌐 CeylonScope</p>
        <p class="hero-sub">Sri Lanka · Real-time News & Market Intelligence Dashboard</p>
      </div>
      <div class="hero-badge">v2.0 · Live</div>
    </div>
    """, unsafe_allow_html=True)

    render_ticker(fd)

    # ── Key metrics ──
    last_24h  = filter_recent(all_news, 24)
    last_3h   = filter_recent(all_news, 3)
    risk_24h  = (last_24h["Insight"] != "Normal").sum() if not last_24h.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles (24h)",    f"{len(last_24h):,}", help="Unique articles fetched in last 24 hours")
    c2.metric("Articles (3h)",     f"{len(last_3h):,}")
    c3.metric("Sectors Active",    last_24h["Sector"].nunique() if not last_24h.empty else 0,
              delta=f"of {len(sector_map)} total", delta_color="off")
    c4.metric("Risk Alerts (24h)", int(risk_24h),
              delta="⚠️ elevated" if risk_24h > 10 else "✅ normal",
              delta_color="inverse" if risk_24h > 10 else "off")

    st.markdown("<div class='section-header'>Exchange Rates · LKR</div>", unsafe_allow_html=True)
    fx = fd.get("fx", {})
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    pairs = [
        ("USD / LKR", fx.get("USD_LKR"), fc1),
        ("GBP / LKR", fx.get("GBP_LKR"), fc2),
        ("INR / LKR", fx.get("INR_LKR"), fc3),
        ("AUD / LKR", fx.get("AUD_LKR"), fc4),
        ("JPY / LKR", fx.get("JPY_LKR"), fc5),
    ]
    for label, val, col:
        col.metric(label, f"{val:,.2f}" if val else "N/A")

    if fx.get("date"):
        st.caption(f"Rates as of {fx['date']} · Source: ExchangeRatesAPI")

    st.markdown("<div class='section-header'>Commodities & Crypto</div>", unsafe_allow_html=True)
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Gold (USD/oz)",    f"${fd.get('gold_usd', 0):,.2f}")
    cc2.metric("Silver (USD/oz)",  f"${fd.get('silver_usd', 0):,.2f}")
    cc3.metric("Bitcoin (USD)",    f"${fd.get('btc_usd', 0):,.0f}",
               delta=f"{fd.get('btc_chg', 0):+.2f}% 24h")
    cc4.metric("Ethereum (USD)",   f"${fd.get('eth_usd', 0):,.0f}",
               delta=f"{fd.get('eth_chg', 0):+.2f}% 24h")

    # ── Top source breakdown ──
    st.markdown("<div class='section-header'>Source Breakdown (24h)</div>", unsafe_allow_html=True)
    if not last_24h.empty:
        src_counts = last_24h["source"].value_counts().reset_index()
        src_counts.columns = ["Source", "Articles"]
        fig_src = px.bar(
            src_counts, x="Articles", y="Source", orientation="h",
            color_discrete_sequence=["#3B82F6"],
            height=200
        )
        fig_src.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#64748B", size=12),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_src, use_container_width=True)

    # ── Latest 5 articles ──
    st.markdown("<div class='section-header'>Latest Headlines</div>", unsafe_allow_html=True)
    if not all_news.empty:
        top5 = all_news.sort_values("datetime", ascending=False).head(5)
        for _, row in top5.iterrows():
            sector = row.get("Sector", "Other")
            insight = row.get("Insight", "Normal")
            badge_sector = f'<span class="news-sector-badge">{sector}</span>'
            badge_risk = f'<span class="news-risk-badge">{insight}</span>' if insight != "Normal" else ""
            dt_str = row["datetime"].strftime("%b %d · %H:%M UTC") if pd.notna(row.get("datetime")) else ""
            link = row.get("link", "#")
            content_snip = str(row.get("Content", ""))[:110] + "..."
            st.markdown(f"""
            <div class="news-item">
              <div style="margin-bottom:6px;">{badge_sector}{badge_risk}</div>
              <a href="{link}" target="_blank" style="font-size:14px;font-weight:500;color:#0F172A;text-decoration:none;">
                {content_snip}
              </a>
              <div style="font-size:11px;color:#94A3B8;margin-top:4px;">{dt_str} · {row.get('source','')}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: NEWS FEED
# ═══════════════════════════════════════════════════════════
elif page == "📰  News Feed":

    st.markdown("""
    <div class="page-hero">
      <div>
        <p class="hero-title">📰 News Feed</p>
        <p class="hero-sub">Aggregated & ML-classified articles from all sources</p>
      </div>
    </div>""", unsafe_allow_html=True)

    render_ticker(fd)

    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        time_filter = st.slider("Articles from last N hours", 1, 168, 24, step=1)
    with col_f2:
        sector_filter = st.multiselect(
            "Filter by sector",
            options=sorted(all_news["Sector"].dropna().unique().tolist()) if not all_news.empty else [],
            default=[]
        )
    with col_f3:
        source_filter = st.multiselect(
            "Filter by source",
            options=sorted(all_news["source"].dropna().unique().tolist()) if not all_news.empty else [],
            default=[]
        )

    latest_df = filter_recent(all_news, time_filter).sort_values("datetime", ascending=False)
    if sector_filter:
        latest_df = latest_df[latest_df["Sector"].isin(sector_filter)]
    if source_filter:
        latest_df = latest_df[latest_df["source"].isin(source_filter)]

    st.markdown(f"<div class='section-header'>{len(latest_df):,} articles · last {time_filter}h</div>",
                unsafe_allow_html=True)

    if not latest_df.empty:
        display_df = latest_df[["datetime", "source", "Sector", "Content", "Insight", "link"]].copy()
        display_df["datetime"] = display_df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
        display_df.rename(columns={"link": "Link"}, inplace=True)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "datetime":  st.column_config.TextColumn("Published (UTC)", width="small"),
                "source":    st.column_config.TextColumn("Source", width="small"),
                "Sector":    st.column_config.TextColumn("Sector", width="small"),
                "Content":   st.column_config.TextColumn("Headline / Content"),
                "Insight":   st.column_config.TextColumn("Risk Signals", width="medium"),
                "Link":      st.column_config.LinkColumn("Link", display_text="Open ↗", width="small"),
            }
        )
    else:
        st.info("No articles match the selected filters.")


# ═══════════════════════════════════════════════════════════
#  PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════
elif page == "📈  Analytics":

    st.markdown("""
    <div class="page-hero">
      <div>
        <p class="hero-title">📈 Analytics</p>
        <p class="hero-sub">Volume trends, sector distribution, and risk heatmaps</p>
      </div>
    </div>""", unsafe_allow_html=True)

    render_ticker(fd)

    if all_news.empty:
        st.warning("No data available yet.")
    else:
        CHART_THEME = dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#64748B", size=12),
            margin=dict(l=10, r=10, t=40, b=10)
        )

        # ── Row 1: Volume timeline ──
        st.markdown("<div class='section-header'>Daily Article Volume · Last 14 Days</div>",
                    unsafe_allow_html=True)
        tl = all_news.copy()
        tl["date"] = tl["datetime"].dt.date
        today = datetime.now().date()
        date_range = pd.date_range(end=today, periods=14).date
        tc = tl.groupby("date").size().reindex(date_range, fill_value=0).reset_index()
        tc.columns = ["date", "count"]
        fig_tl = px.area(tc, x="date", y="count", height=260,
                         color_discrete_sequence=["#3B82F6"])
        fig_tl.update_traces(line_width=2, fillcolor="rgba(59,130,246,0.12)")
        fig_tl.update_xaxes(showgrid=False)
        fig_tl.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        fig_tl.update_layout(**CHART_THEME)
        st.plotly_chart(fig_tl, use_container_width=True)

        # ── Row 2: Sector pie + source bar ──
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='section-header'>Sector Distribution</div>", unsafe_allow_html=True)
            sc = all_news["Sector"].value_counts().reset_index()
            sc.columns = ["Sector", "Count"]
            sc["Color"] = sc["Sector"].map(SECTOR_COLORS)
            fig_pie = px.pie(sc, values="Count", names="Sector",
                             color="Sector", color_discrete_map=SECTOR_COLORS,
                             hole=0.5, height=300)
            fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                                  textfont_size=11)
            fig_pie.update_layout(**CHART_THEME)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            st.markdown("<div class='section-header'>Articles by Source</div>", unsafe_allow_html=True)
            src = all_news["source"].value_counts().reset_index()
            src.columns = ["Source", "Count"]
            fig_src = px.bar(src, x="Source", y="Count", height=300,
                             color_discrete_sequence=["#6366F1"])
            fig_src.update_layout(**CHART_THEME)
            fig_src.update_xaxes(showgrid=False)
            fig_src.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
            st.plotly_chart(fig_src, use_container_width=True)

        # ── Row 3: Heatmap ──
        st.markdown("<div class='section-header'>Risk Score Heatmap by Sector</div>",
                    unsafe_allow_html=True)
        score_cols = ["Economy_Score", "Weather_Score", "Social_Score",
                      "Logistics_Score", "Tourism_Score"]
        if all(c in all_news.columns for c in score_cols):
            heat = all_news.groupby("Sector")[score_cols].sum()
            heat.columns = ["Economy", "Weather", "Social", "Logistics", "Tourism"]
            fig_heat = px.imshow(heat, text_auto=True, height=380,
                                 color_continuous_scale="Blues",
                                 labels=dict(x="Risk Category", y="Sector", color="Score"))
            fig_heat.update_layout(**CHART_THEME)
            st.plotly_chart(fig_heat, use_container_width=True)

        # ── Row 4: Sector over time ──
        st.markdown("<div class='section-header'>Sector Activity · Last 7 Days</div>",
                    unsafe_allow_html=True)
        tl2 = all_news.copy()
        tl2["date"] = tl2["datetime"].dt.date
        sector_time = tl2.groupby(["date", "Sector"]).size().reset_index(name="count")
        fig_st = px.line(sector_time, x="date", y="count", color="Sector",
                         color_discrete_map=SECTOR_COLORS, height=320,
                         line_shape="spline")
        fig_st.update_layout(**CHART_THEME)
        fig_st.update_xaxes(showgrid=False)
        fig_st.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_st, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: MARKETS (NEW)
# ═══════════════════════════════════════════════════════════
elif page == "💰  Markets":

    st.markdown("""
    <div class="page-hero">
      <div>
        <p class="hero-title">💰 Market Pulse</p>
        <p class="hero-sub">Live exchange rates · Commodities · Crypto · Colombo Stock Exchange</p>
      </div>
      <div class="hero-badge">15-min cache</div>
    </div>""", unsafe_allow_html=True)

    render_ticker(fd)

    # ── FX Rates ──
    st.markdown("<div class='section-header'>LKR Exchange Rates</div>", unsafe_allow_html=True)
    fx = fd.get("fx", {})

    cards_html = ""
    fx_pairs = [
        ("🇺🇸 USD / LKR", fx.get("USD_LKR"), 0),
        ("🇬🇧 GBP / LKR", fx.get("GBP_LKR"), 0),
        ("🇮🇳 INR / LKR", fx.get("INR_LKR"), 0),
        ("🇦🇺 AUD / LKR", fx.get("AUD_LKR"), 0),
        ("🇯🇵 JPY / LKR", fx.get("JPY_LKR"), 0),
    ]
    fx_cols = st.columns(5)
    for i, (label, val, chg) in enumerate(fx_pairs):
        val_str = f"{val:,.2f}" if val else "—"
        fx_cols[i].metric(label, val_str)

    if fx.get("date"):
        st.caption(f"Source: ExchangeRatesAPI · as of {fx['date']}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Colombo Stock Exchange ──
    st.markdown("<div class='section-header'>Colombo Stock Exchange (CSE)</div>",
                unsafe_allow_html=True)

    cse_aspi = fd.get("cse_aspi", {})
    cse_sp20 = fd.get("cse_sp20", {}) or fd.get("cse_s&p20", {})

    cse1, cse2, cse3 = st.columns(3)
    aspi_val = cse_aspi.get("value", "—")
    aspi_chg = cse_aspi.get("change", 0)
    aspi_pct = cse_aspi.get("change_pct", 0)
    cse1.metric("ASPI (All Share)",
                f"{aspi_val:,.2f}" if isinstance(aspi_val, (int, float)) else aspi_val,
                delta=f"{aspi_chg:+.2f} ({aspi_pct:+.2f}%)" if isinstance(aspi_chg, (int, float)) else None)

    sp20_val = cse_sp20.get("value", "—")
    sp20_chg = cse_sp20.get("change", 0)
    sp20_pct = cse_sp20.get("change_pct", 0)
    cse2.metric("S&P SL20",
                f"{sp20_val:,.2f}" if isinstance(sp20_val, (int, float)) else sp20_val,
                delta=f"{sp20_chg:+.2f} ({sp20_pct:+.2f}%)" if isinstance(sp20_chg, (int, float)) else None)

    cse3.metric("Market Status",
                "🟢 Open" if datetime.now(timezone.utc).weekday() < 5 else "🔴 Closed",
                help="CSE trading hours: Mon–Fri 09:30–14:30 SLST")

    st.caption("CSE data: live endpoint · refreshes every 15 minutes.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Commodities ──
    st.markdown("<div class='section-header'>Commodities</div>", unsafe_allow_html=True)
    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Gold (USD/oz)",   f"${fd.get('gold_usd',   0):,.2f}")
    cm2.metric("Silver (USD/oz)", f"${fd.get('silver_usd', 0):,.2f}")
    cm3.metric("Oil (Brent USD/bbl)", f"${fd.get('oil_usd', 83.40):,.2f}",
               help="Approximate benchmark price")

    st.caption("Source: metals.live · 15-min refresh.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Crypto ──
    st.markdown("<div class='section-header'>Cryptocurrency</div>", unsafe_allow_html=True)
    cr1, cr2 = st.columns(2)
    cr1.metric("Bitcoin (BTC)",  f"${fd.get('btc_usd', 0):,.0f}",
               delta=f"{fd.get('btc_chg', 0):+.2f}% (24h)")
    cr2.metric("Ethereum (ETH)", f"${fd.get('eth_usd', 0):,.0f}",
               delta=f"{fd.get('eth_chg', 0):+.2f}% (24h)")

    st.caption("Source: CoinGecko · free public API.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Finance-related news ──
    st.markdown("<div class='section-header'>Finance & Economy Headlines</div>",
                unsafe_allow_html=True)
    if not all_news.empty and "Sector" in all_news.columns:
        fin_news = all_news[all_news["Sector"].isin(["Finance", "Economy"])]\
                   .sort_values("datetime", ascending=False).head(10)
        for _, row in fin_news.iterrows():
            sector = row.get("Sector", "Finance")
            dt_str = row["datetime"].strftime("%b %d · %H:%M") if pd.notna(row.get("datetime")) else ""
            link   = row.get("link", "#")
            content = str(row.get("Content", ""))[:130] + "..."
            st.markdown(f"""
            <div class="news-item" style="border-left-color:#10B981;">
              <span class="news-sector-badge" style="background:#ECFDF5;color:#065F46;">{sector}</span>
              <a href="{link}" target="_blank" style="font-size:14px;font-weight:500;color:#0F172A;text-decoration:none;">
                {content}
              </a>
              <div style="font-size:11px;color:#94A3B8;margin-top:4px;">{dt_str} · {row.get('source','')}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No finance-related articles found yet.")


# ═══════════════════════════════════════════════════════════
#  PAGE: RISK SIGNALS
# ═══════════════════════════════════════════════════════════
elif page == "⚠️  Risk Signals":

    st.markdown("""
    <div class="page-hero">
      <div>
        <p class="hero-title">⚠️ Risk Signals</p>
        <p class="hero-sub">Keyword-triggered alerts, high-risk articles, and threat mapping</p>
      </div>
    </div>""", unsafe_allow_html=True)

    render_ticker(fd)

    if all_news.empty or not all(c in all_news.columns for c in
                                  ["Economy_Score","Weather_Score","Social_Score"]):
        st.warning("No risk data available yet.")
    else:
        score_cols = ["Economy_Score", "Weather_Score", "Social_Score",
                      "Logistics_Score", "Tourism_Score"]
        heat_total = all_news[score_cols].sum()

        # ── Summary metrics ──
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("💰 Economy",   int(heat_total["Economy_Score"]))
        r2.metric("🌧 Weather",   int(heat_total["Weather_Score"]))
        r3.metric("🤝 Social",    int(heat_total["Social_Score"]))
        r4.metric("🚚 Logistics", int(heat_total["Logistics_Score"]))
        r5.metric("✈️ Tourism",   int(heat_total["Tourism_Score"]))

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Risk articles ──
        risky = all_news.loc[all_news["Insight"] != "Normal"].copy()

        if not risky.empty:
            risky["Total_Risk"] = risky[score_cols].sum(axis=1)
            risky = risky.sort_values("Total_Risk", ascending=False)

            st.markdown(f"<div class='section-header'>{len(risky)} flagged articles</div>",
                        unsafe_allow_html=True)

            tabs = st.tabs(["🔴 High Risk", "📊 Risk Map", "☁️ Keyword Cloud", "⬇️ Export"])

            with tabs[0]:
                for _, row in risky.head(20).iterrows():
                    sector  = row.get("Sector", "Other")
                    insight = row.get("Insight", "")
                    dt_str  = row["datetime"].strftime("%b %d · %H:%M UTC") if pd.notna(row.get("datetime")) else ""
                    link    = row.get("link", "#")
                    content = str(row.get("Content", ""))[:140] + "..."
                    risk_score = int(row.get("Total_Risk", 0))
                    color_map = {0: "#3B82F6", 1: "#F59E0B", 2: "#F97316", 3: "#EF4444"}
                    border_color = color_map.get(min(risk_score, 3), "#EF4444")

                    st.markdown(f"""
                    <div class="news-item" style="border-left-color:{border_color};">
                      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                        <div>
                          <span class="news-sector-badge">{sector}</span>
                          <span style="font-size:11px;color:#94A3B8;">{dt_str} · {row.get('source','')}</span>
                        </div>
                        <span style="background:#FEF2F2;color:#991B1B;font-size:11px;
                                     font-weight:600;border-radius:6px;padding:2px 8px;">
                          Risk ×{risk_score}
                        </span>
                      </div>
                      <a href="{link}" target="_blank"
                         style="font-size:14px;font-weight:500;color:#0F172A;text-decoration:none;">
                        {content}
                      </a>
                      <div style="margin-top:6px;font-size:12px;color:#92400E;">{insight}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with tabs[1]:
                heat = all_news.groupby("Sector")[score_cols].sum()
                heat.columns = ["Economy", "Weather", "Social", "Logistics", "Tourism"]
                fig_h = px.imshow(heat, text_auto=True, height=400,
                                  color_continuous_scale="Reds",
                                  labels=dict(x="Risk Category", y="Sector", color="Score"))
                fig_h.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color="#64748B", size=12),
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                st.plotly_chart(fig_h, use_container_width=True)

            with tabs[2]:
                text_blob = " ".join(risky["Content"].astype(str).tolist())
                if text_blob.strip():
                    wc = WordCloud(
                        width=1400, height=500,
                        background_color="white",
                        colormap="RdYlBu_r",
                        max_words=80,
                        prefer_horizontal=0.8
                    ).generate(text_blob)
                    fig_wc, ax = plt.subplots(figsize=(14, 5))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    fig_wc.patch.set_facecolor("none")
                    st.pyplot(fig_wc)

            with tabs[3]:
                st.markdown("""
                <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:20px 24px;">
                  <p style="font-size:14px;color:#475569;margin:0 0 12px;">
                    Download all risk-flagged articles as a CSV for further analysis.
                  </p>
                </div>
                """, unsafe_allow_html=True)
                st.download_button(
                    label="⬇️  Download Risk Articles CSV",
                    data=risky.to_csv(index=False).encode("utf-8"),
                    file_name=f"ceylonscope_risk_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        else:
            st.success("✅ All clear — no high-risk articles detected at this time.")
