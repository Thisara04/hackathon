import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# --------------------------
# Load Models
# --------------------------
@st.cache_resource
def load_models():
    category_model = joblib.load("meka use karanna.pkl")  # your CatBoost model
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return category_model, embedding_model

category_model, embedding_model = load_models()

# --------------------------
# Category mapping
# --------------------------
sector_map = {
    0: "Energy",
    1: "Logistics",
    2: "Education",
    3: "Health",
    4: "Finance",
    5: "Government",
    6: "Tourism",
    7: "Agriculture",
    8: "Social",
    9: "Technology",
    10: "Economy",
    11: "Other"
}

# --------------------------
# Keyword sets for scores
# --------------------------
economy_keywords = ['stock', 'rupee', 'inflation', 'currency', 'finance', 'economic']
weather_keywords = ['rain', 'flood', 'storm', 'temperature', 'drought']
social_keywords = ['protest', 'strike', 'crowd', 'social', 'community']
logistics_keywords = ['traffic', 'accident', 'port', 'delivery', 'transport']
tourism_keywords = ['tourism', 'travel', 'hotel', 'tourist', 'visa']


def calc_signal_score(text, keywords):
    return sum(1 for word in keywords if word.lower() in text.lower())


def generate_insight(row):
    insights = []
    if row['Economy_Score'] >= 2:
        insights.append("Economic risk rising")
    if row['Weather_Score'] >= 1:
        insights.append("Weather disruption possible")
    if row['Social_Score'] >= 1:
        insights.append("Social unrest warning")
    if row['Logistics_Score'] >= 1:
        insights.append("Logistics/transport alert")
    if row['Tourism_Score'] >= 1:
        insights.append("Tourism opportunity")

    return "; ".join(insights) if insights else "Normal"


# --------------------------
# Streamlit UI
# --------------------------
st.title("📊 Smart News Signal Classifier")
st.write("Upload a CSV file containing news headlines + time features.")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    required_cols = ["Content", "Month_sin", "Month_cos", "DOW_sin", "DOW_cos"]

    # Validate columns
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Missing required columns: {required_cols}")
        st.stop()

    # -----------------------------------------------------
    # Generate MiniLM embeddings
    # -----------------------------------------------------
    st.write("### 🔄 Generating MiniLM Embeddings...")
    X_text = df["Content"].tolist()
    X_text_emb = embedding_model.encode(
        X_text,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # -----------------------------------------------------
    # Combine embeddings with numeric features
    # -----------------------------------------------------
    X_time = df[["Month_sin", "Month_cos", "DOW_sin", "DOW_cos"]].to_numpy()
    X = np.hstack([X_text_emb, X_time])

    # -----------------------------------------------------
    # Predict category
    # -----------------------------------------------------
    st.write("### 📂 Predicting Sectors...")
    df["SectorID"] = category_model.predict(X)
    df["Sector"] = df["SectorID"].map(sector_map)

    # --------------------------
    # Scoring (Phase 4)
    # --------------------------
    st.write("### 📡 Calculating Signal Scores...")

    df['Economy_Score'] = df['Content'].apply(lambda x: calc_signal_score(x, economy_keywords))
    df['Weather_Score'] = df['Content'].apply(lambda x: calc_signal_score(x, weather_keywords))
    df['Social_Score'] = df['Content'].apply(lambda x: calc_signal_score(x, social_keywords))
    df['Logistics_Score'] = df['Content'].apply(lambda x: calc_signal_score(x, logistics_keywords))
    df['Tourism_Score'] = df['Content'].apply(lambda x: calc_signal_score(x, tourism_keywords))

    df["Insight"] = df.apply(generate_insight, axis=1)

    # --------------------------
    # Display results
    # --------------------------
    st.write("### ✅ Results")
    st.dataframe(df)

    # --------------------------
    # Download output CSV
    # --------------------------
    csv_output = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Output CSV",
        data=csv_output,
        file_name="news_signals_output.csv",
        mime="text/csv"
    )

    st.success("✔ Processing complete!")
