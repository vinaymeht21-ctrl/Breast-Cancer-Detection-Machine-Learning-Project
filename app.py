"""
app.py
------
Interactive Streamlit website for the Breast Cancer Wisconsin
Logistic Regression classifier trained in train_model.py.

Features:
  - Sliders for all 30 tumor measurements, grouped into
    Mean / Standard Error / Worst tabs
  - "Load sample patient" (random row from the dataset) and
    "Reset to average" buttons
  - Live prediction (Benign / Malignant) with probability gauge
  - Batch prediction via CSV upload
  - Dataset overview & model performance page

Run with:
    streamlit run app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
META_PATH = "feature_meta.json"
DATA_PATH = "data.csv"

st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon="🩺",
    layout="wide",
)

# --------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    missing = [p for p in (MODEL_PATH, SCALER_PATH, META_PATH) if not os.path.exists(p)]
    if missing:
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, scaler, meta


@st.cache_data
def load_raw_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    drop_cols = [c for c in ["Unnamed: 32", "id"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


model, scaler, meta = load_artifacts()
raw_df = load_raw_data()

if model is None:
    st.error(
        "Model artifacts not found. Please run `python train_model.py` first "
        "(with data.csv in the same folder) to generate model.pkl, scaler.pkl, "
        "and feature_meta.json."
    )
    st.stop()

FEATURES = meta["feature_names"]
GROUPS = {
    "Mean": [f for f in FEATURES if f.endswith("_mean")],
    "Standard Error": [f for f in FEATURES if f.endswith("_se")],
    "Worst": [f for f in FEATURES if f.endswith("_worst")],
}

if "patient_values" not in st.session_state:
    st.session_state.patient_values = {f: meta["mean"][f] for f in FEATURES}


def set_values(source_row: dict):
    for f in FEATURES:
        st.session_state.patient_values[f] = float(source_row[f])


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("🩺 Breast Cancer Predictor")
    st.caption(
        "Logistic Regression model trained on the Breast Cancer Wisconsin "
        "(Diagnostic) dataset."
    )
    st.metric("Held-out test accuracy", f"{meta['accuracy'] * 100:.2f}%")

    st.divider()
    st.subheader("Quick actions")

    if raw_df is not None:
        if st.button("🎲 Load random sample patient", use_container_width=True):
            row = raw_df.drop(columns=["diagnosis"]).sample(1).iloc[0]
            set_values(row.to_dict())
            st.rerun()

    if st.button("↩️ Reset to dataset average", use_container_width=True):
        set_values(meta["mean"])
        st.rerun()

    st.divider()
    page = st.radio(
        "Navigate",
        ["Single Prediction", "Batch Prediction (CSV)", "Dataset & Model Info"],
    )

    st.divider()
    st.caption(
        "⚠️ Educational demo only — not a medical device. Do not use for "
        "real diagnostic decisions."
    )


# --------------------------------------------------------------------------
# Page: Single prediction
# --------------------------------------------------------------------------

def render_single_prediction():
    st.header("Single Patient Prediction")
    st.write(
        "Adjust the tumor cell-nuclei measurements below (or load a random "
        "sample patient from the sidebar), then predict the diagnosis."
    )

    tabs = st.tabs(list(GROUPS.keys()))
    for tab, group_name in zip(tabs, GROUPS.keys()):
        with tab:
            cols = st.columns(2)
            for i, feat in enumerate(GROUPS[group_name]):
                lo = float(meta["min"][feat])
                hi = float(meta["max"][feat])
                # give a little headroom beyond observed min/max
                span = hi - lo if hi > lo else 1.0
                lo_padded = max(0.0, lo - 0.05 * span)
                hi_padded = hi + 0.05 * span
                with cols[i % 2]:
                    st.session_state.patient_values[feat] = st.slider(
                        pretty(feat),
                        min_value=float(lo_padded),
                        max_value=float(hi_padded),
                        value=float(st.session_state.patient_values[feat]),
                        step=float(span / 200 if span > 0 else 0.01),
                        key=f"slider_{feat}",
                    )

    st.divider()

    input_vector = np.array([[st.session_state.patient_values[f] for f in FEATURES]])
    scaled = scaler.transform(input_vector)
    proba = model.predict_proba(scaled)[0]
    pred = model.predict(scaled)[0]

    malignant_prob = proba[1]
    benign_prob = proba[0]

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.subheader("Prediction")
        if pred == 1:
            st.error(f"### 🔴 Malignant\nConfidence: {malignant_prob * 100:.1f}%")
        else:
            st.success(f"### 🟢 Benign\nConfidence: {benign_prob * 100:.1f}%")

        st.write("")
        st.progress(float(malignant_prob), text=f"Malignant probability: {malignant_prob*100:.1f}%")

    with col2:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=malignant_prob * 100,
                number={"suffix": "%"},
                title={"text": "Malignancy Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred" if pred == 1 else "seagreen"},
                    "steps": [
                        {"range": [0, 40], "color": "#d4f4dd"},
                        {"range": [40, 70], "color": "#fff3cd"},
                        {"range": [70, 100], "color": "#f8d7da"},
                    ],
                },
            )
        )
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("How this patient compares to the dataset average"):
        compare_df = pd.DataFrame(
            {
                "Feature": [pretty(f) for f in FEATURES],
                "Patient value": [st.session_state.patient_values[f] for f in FEATURES],
                "Dataset average": [meta["mean"][f] for f in FEATURES],
            }
        )
        st.dataframe(compare_df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Page: Batch prediction
# --------------------------------------------------------------------------

def render_batch_prediction():
    st.header("Batch Prediction from CSV")
    st.write(
        "Upload a CSV file containing the 30 measurement columns "
        f"({', '.join(FEATURES[:3])}, ... ). Extra columns such as `id` or "
        "`diagnosis` are ignored automatically."
    )

    st.download_button(
        "Download example CSV template",
        data=pd.DataFrame([meta["mean"]])[FEATURES].to_csv(index=False),
        file_name="sample_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            return

        missing_cols = [f for f in FEATURES if f not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return

        X_batch = batch_df[FEATURES]
        X_scaled = scaler.transform(X_batch)
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]

        result_df = batch_df.copy()
        result_df["prediction"] = np.where(preds == 1, "Malignant", "Benign")
        result_df["malignant_probability"] = probs.round(4)

        st.success(f"Predicted {len(result_df)} rows.")
        st.dataframe(result_df, use_container_width=True)

        st.download_button(
            "Download predictions as CSV",
            data=result_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
        )

        counts = result_df["prediction"].value_counts()
        st.bar_chart(counts)


# --------------------------------------------------------------------------
# Page: Dataset & model info
# --------------------------------------------------------------------------

def render_info_page():
    st.header("Dataset & Model Information")

    col1, col2, col3 = st.columns(3)
    if raw_df is not None:
        col1.metric("Rows", len(raw_df))
        col2.metric("Features", len(FEATURES))
        col3.metric("Test Accuracy", f"{meta['accuracy']*100:.2f}%")

        st.subheader("Diagnosis distribution")
        dist = raw_df["diagnosis"].value_counts().rename({"M": "Malignant", "B": "Benign"})
        st.bar_chart(dist)

        st.subheader("Raw data preview")
        st.dataframe(raw_df.head(20), use_container_width=True)

        st.subheader("Feature correlation with diagnosis")
        corr_df = raw_df.copy()
        corr_df["diagnosis"] = (corr_df["diagnosis"] == "M").astype(int)
        corr = corr_df.corr(numeric_only=True)["diagnosis"].drop("diagnosis").sort_values()
        st.bar_chart(corr)
    else:
        st.info("data.csv not found — dataset exploration is unavailable, but predictions still work.")

    st.subheader("About the model")
    st.markdown(
        """
- **Algorithm:** Logistic Regression (scikit-learn)
- **Preprocessing:** All 30 numeric features standardized with `StandardScaler`
- **Train/test split:** 70% / 30%, `random_state=42`
- **Target:** `diagnosis` — Malignant (M) encoded as 1, Benign (B) encoded as 0
        """
    )


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

if page == "Single Prediction":
    render_single_prediction()
elif page == "Batch Prediction (CSV)":
    render_batch_prediction()
else:
    render_info_page()