import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model + Preprocessor
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/category_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    return model, preprocessor


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Sri Lanka News Category Classifier")
st.write("Upload your CSV → preprocessing → prediction → accuracy → download results")

model, preprocessor = load_model()

# File uploader
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📝 Uploaded Data Preview")
    st.dataframe(df.head())

    required_cols = ["Content", "MonthsSign", "MonthCost", "DateOfWeekSign", "DateOfWeekCost"]

    # Check required columns
    if not all(col in df.columns for col in required_cols):
        st.error(f"Your CSV must contain the following columns:\n{required_cols}")
    else:
        # -----------------------------
        # Preprocess + Predict
        # -----------------------------
        X = df[required_cols]
        X_transformed = preprocessor.transform(X)
        preds = model.predict(X_transformed)

        df["PredictedCategory"] = preds

        st.subheader("📌 Predictions")
        st.dataframe(df[["Content", "PredictedCategory"]].head(10))

        # -----------------------------
        # Phase 4 — Accuracy (OPTIONAL)
        # -----------------------------
        if "TrueCategory" in df.columns:
            from sklearn.metrics import accuracy_score, classification_report

            accuracy = accuracy_score(df["TrueCategory"], preds)
            st.subheader("📈 Accuracy (Phase 4)")
            st.write(f"**Accuracy:** {accuracy:.4f}")

            st.text("Classification report:")
            st.text(classification_report(df["TrueCategory"], preds))

        else:
            st.info("Phase 4 accuracy not available (CSV does not contain TrueCategory column).")

        # -----------------------------
        # Download results
        # -----------------------------
        st.subheader("📥 Download Results")
        st.download_button(
            "Download CSV with Predictions",
            df.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv"
        )
