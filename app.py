import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime
from bs4 import BeautifulSoup

# --------------------------------------------------------------
# Page config
# --------------------------------------------------------------
st.set_page_config(page_title="DailyMirror RSS Dashboard", layout="wide")

# --------------------------------------------------------------
# Auto-refresh every 10 minutes
# --------------------------------------------------------------
# 10 minutes = 600,000 milliseconds
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=600_000, key="refresh10min")

RSS_URL = "https://www.dailymirror.lk/RSS_Feeds/breaking_news"

# --------------------------------------------------------------
# Clean malformed RSS text
# --------------------------------------------------------------
def clean_rss_xml(text: str) -> str:
    # Remove unescaped '&'
    text = re.sub(r"&(?!(amp;|lt;|gt;|quot;|apos;))", "&amp;", text)
    # Remove stray CDATA endings
    text = text.replace("]]> ]]>","")
    # Remove bad unicode blocks
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7F]+", " ", text)
    return text

# --------------------------------------------------------------
# Parse RSS safely even if corrupted
# --------------------------------------------------------------
def fetch_rss():
    try:
        resp = requests.get(RSS_URL, timeout=10)
        raw = resp.text
        cleaned = clean_rss_xml(raw)

        # Parse using BeautifulSoup
        soup = BeautifulSoup(cleaned, "xml")
        items = soup.find_all("item")

        records = []
        for it in items:
            title = it.title.text.strip() if it.title else ""
            link = it.link.text.strip() if it.link else ""
            pub = it.pubDate.text.strip() if it.pubDate else ""

            # Try extract image
            img = ""
            enclosure = it.find("enclosure")
            if enclosure and enclosure.get("url"):
                img = enclosure.get("url")

            records.append({
                "title": title,
                "link": link,
                "pubDate": pub,
                "image": img,
            })
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Failed to fetch RSS: {e}")
        return pd.DataFrame()

# --------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------
def preprocess(df):
    if df.empty:
        return df

    def parse_date(x):
        try:
            return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        except:
            try:
                return pd.to_datetime(x)
            except:
                return None

    df["datetime"] = df["pubDate"].apply(parse_date)
    df = df.dropna(subset=["datetime"])

    df["month"] = df["datetime"].dt.month
    df["dow"] = df["datetime"].dt.dayofweek

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    df["content"] = df["title"]  # we don't have description field

    return df

# --------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------
st.title("📰 DailyMirror Live RSS Dashboard")

with st.spinner("Fetching RSS feed…"):
    df = fetch_rss()

st.subheader("Raw News Items")
st.dataframe(df, use_container_width=True)

if not df.empty:
    df = preprocess(df)

    st.subheader("Pre-processed Features")
    st.dataframe(df, use_container_width=True)

    # Monthly count
    st.subheader("Monthly News Distribution")
    monthly = df.groupby("month").size().reset_index(name="count")
    st.bar_chart(monthly.set_index("month")["count"])

    # Day of week
    st.subheader("Day-Of-Week Distribution")
    dow = df.groupby("dow").size().reset_index(name="count")
    st.bar_chart(dow.set_index("dow")["count"])

st.success("Dashboard updated!")
