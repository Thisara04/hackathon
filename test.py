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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (UNCHANGED CODE ABOVE...)

# ─────────────────────────────────────────────

# FINANCIAL DATA

# ─────────────────────────────────────────────

@st.cache_data(ttl=900)
def fetch_financial_data():
result = {}

```
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

# FIXED: metals API parsing
try:
    g = requests.get("https://api.metals.live/v1/spot", timeout=5).json()
    if isinstance(g, list):
        metals = {item["metal"]: item["price"] for item in g}
        result["gold_usd"] = round(metals.get("gold", 0), 2)
        result["silver_usd"] = round(metals.get("silver", 0), 2)
    else:
        result["gold_usd"] = 3320.00
        result["silver_usd"] = 32.50
    result["oil_usd"] = 0
except Exception:
    result["gold_usd"] = 3320.00
    result["silver_usd"] = 32.50
    result["oil_usd"] = 83.40

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

# FIXED: invalid key name
try:
    cse = requests.get(
        "https://www.cse.lk/api/market-summary",
        timeout=6, headers={"User-Agent": "Mozilla/5.0"}
    ).json()
    result["cse_aspi"] = cse.get("aspi", {})
    result["cse_sp20"] = cse.get("sp20", {})
except Exception:
    result["cse_aspi"] = {"value": 10_247.30, "change": 45.60, "change_pct": 0.45}
    result["cse_sp20"] = {"value": 3_841.15, "change": -12.30, "change_pct": -0.32}

return result
```

# (UNCHANGED CODE CONTINUES...)

# ─────────────────────────────────────────────

# MARKETS PAGE FIX

# ─────────────────────────────────────────────

elif page == "💰  Markets":

```
# (UNCHANGED ABOVE...)

cse_aspi = fd.get("cse_aspi", {})
cse_sp20 = fd.get("cse_sp20", {})  # FIXED (removed invalid fallback)
```

# (UNCHANGED CODE...)

# ─────────────────────────────────────────────

# FINAL FIX (removed syntax error)

# ─────────────────────────────────────────────

else:
st.success("✅ All clear — no high-risk articles detected at this time.")
