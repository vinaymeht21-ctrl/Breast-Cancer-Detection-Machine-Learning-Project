# 🩺 Breast Cancer Diagnosis Predictor

An interactive **Streamlit web app** that predicts whether a breast tumor is **Benign** or **Malignant** using a Logistic Regression model trained on the classic [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) dataset.

> ⚠️ **Disclaimer:** This project is an educational demo only. It is not a medical device and must not be used for real diagnostic decisions.

---

## ✨ Features

- **Single Patient Prediction** — 30 sliders grouped into *Mean / Standard Error / Worst* tabs, with a live prediction, probability gauge, and comparison against dataset averages
- **Batch Prediction (CSV)** — upload a CSV of multiple patients, get predictions with malignant probabilities, and download the results
- **Dataset & Model Info** — data overview, diagnosis distribution, feature correlations, and model details
- **Quick actions** — load a random sample patient or reset to dataset average values

## 🧠 Model

| Item | Detail |
|------|--------|
| Algorithm | Logistic Regression (`scikit-learn`, `max_iter=5000`) |
| Preprocessing | `StandardScaler` on all 30 numeric features |
| Train/test split | 70% / 30%, `random_state=42` |
| Target encoding | `M` → 1 (Malignant), `B` → 0 (Benign) |
| Held-out test accuracy | **98.25%** |

The 30 features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, describing characteristics of the cell nuclei (e.g. radius, texture, perimeter, area, smoothness, compactness, concavity, etc.).

## 📁 Project Structure

```
├── app.py              # Streamlit web application
├── train_model.py      # Trains the model and saves artifacts
├── data.csv            # Breast Cancer Wisconsin dataset (569 rows)
├── model.pkl           # Trained LogisticRegression (generated)
├── scaler.pkl          # Fitted StandardScaler (generated)
├── feature_meta.json   # Feature names, stats & accuracy (generated)
└── requirements.txt    # Python dependencies
```

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (one-time)

This produces `model.pkl`, `scaler.pkl`, and `feature_meta.json`:

```bash
python train_model.py
```

### 3. Run the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (default `http://localhost:8501`).

## 📊 Using the App

**Single prediction:** adjust the sliders (or click *🎲 Load random sample patient* in the sidebar) and read the prediction, confidence, and risk gauge.

**Batch prediction:** upload a CSV containing the 30 measurement columns — a template is available for download inside the app. Extra columns like `id` or `diagnosis` are ignored automatically.

**Expected CSV columns:** `radius_mean`, `texture_mean`, `perimeter_mean`, ..., `fractal_dimension_worst` (all 30 features in `feature_meta.json`).

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [scikit-learn](https://scikit-learn.org/) — model & scaling
- [Plotly](https://plotly.com/python/) — probability gauge
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling
- [Joblib](https://joblib.readthedocs.io/) — model persistence
