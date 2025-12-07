import streamlit as st
from datetime import datetime
from PIL import Image

# -------------------------------
# 🔧 IMAGE (RESIZED BANNER)
# -------------------------------
def load_banner():
    img = Image.open("photo.png")
    img = img.resize((900, 300))  # resized small banner
    return img

# -------------------------------
# 🔧 DATA FETCH FUNCTIONS (PLACEHOLDERS)
# -------------------------------
@st.cache_data
def fetch_latest_news():
    # Replace with your actual fetching code
    return [
        {"title": "News 1", "content": "Content here..."},
        {"title": "News 2", "content": "Content here..."}
    ]

@st.cache_data
def generate_analytics():
    # Replace with your analytics logic
    return {"total_news": 42, "economy": 12, "weather": 5}

@st.cache_data
def generate_risk_signals():
    # Replace with your risk detection code
    return {"high_risk": 3, "medium_risk": 5, "low_risk": 10}

# -------------------------------
# 🔧 SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Go to:",
    ["Home", "Latest News", "Analytics", "Risk Signals"]
)

# -------------------------------
# 🔄 UPDATE BUTTON
# -------------------------------
if st.sidebar.button("🔄 Update Now"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

st.sidebar.write(
    "Last update:",
    datetime.now().strftime("%Y-%m-%d %H:%M")
)

# =====================================================================
#                            PAGE 1 — HOME
# =====================================================================
if page == "Home":
    st.title("📰 Sri Lanka News Intelligence Dashboard")

    st.image(load_banner())

    st.subheader("📌 Overview")
    st.write("""
    This system collects Sri Lankan news in real time, categorizes articles,
    detects risk signals, and generates insight summaries across multiple sectors.
    """)

    st.subheader("⚙️ How It Works")
    st.markdown("""
    - Fetches real-time news from multiple sources  
    - Cleans & preprocesses text  
    - Encodes content with MiniLM embeddings  
    - Predicts sector using CatBoost classification  
    - Detects risk levels (Economy, Weather, Social, Logistics, Tourism)  
    - Generates summaries and analytics  
    """)

    st.subheader("📊 Quick Stats Preview")
    analytics = generate_analytics()
    st.metric("Total News Processed", analytics["total_news"])
    st.metric("Economy Articles", analytics["economy"])
    st.metric("Weather Articles", analytics["weather"])

    st.info("Use the sidebar to navigate between sections.")

# =====================================================================
#                          PAGE 2 — LATEST NEWS
# =====================================================================
elif page == "Latest News":
    st.title("🗞️ Latest News")

    news_list = fetch_latest_news()

    for article in news_list:
        st.subheader(article["title"])
        st.write(article["content"])
        st.markdown("---")

# =====================================================================
#                          PAGE 3 — ANALYTICS
# =====================================================================
elif page == "Analytics":
    st.title("📈 Analytics Overview")

    analytics = generate_analytics()

    st.metric("Total News", analytics["total_news"])
    st.metric("Economy Articles", analytics["economy"])
    st.metric("Weather Articles", analytics["weather"])

    st.write("Add your graphs, plots, and charts here…")

# =====================================================================
#                          PAGE 4 — RISK SIGNALS
# =====================================================================
elif page == "Risk Signals":
    st.title("🚨 Risk Signals")

    risks = generate_risk_signals()

    st.metric("High Risk Alerts", risks["high_risk"])
    st.metric("Medium Risk Alerts", risks["medium_risk"])
    st.metric("Low Risk Alerts", risks["low_risk"])

    st.write("Insert your detailed risk summaries here…")
