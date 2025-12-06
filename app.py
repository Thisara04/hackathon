import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# --------------------------------------------------------------
# Page config
# --------------------------------------------------------------
st.set_page_config(page_title="DailyMirror RSS Dashboard", layout="wide")

# --------------------------------------------------------------
# Auto-refresh every 10 minutes
# --------------------------------------------------------------
st_autorefresh = st.experimental_rerun
st_autorefresh_counter = st.experimental_memo

st.experimental_set_query_params()   # lightweight refresh anchor
st.toast("Auto refreshing every 10 minutes…")
st.experimental_autorefresh(interval=600000, key="refresh10min")

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
    resp = requests.get(RSS_URL, timeout=10)
    raw = resp.text
    cleaned = clean_rss_xml(raw)

    # Try parsing using BeautifulSoup (more forgiving)
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
    st.bar_chart(monthly, x="month", y="count")

    # Day of week
    st.subheader("Day-Of-Week Distribution")
    dow = df.groupby("dow").size().reset_index(name="count")
    st.bar_chart(dow, x="dow", y="count")

st.success("Dashboard updated!")
