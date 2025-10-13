# app.py
import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# -----------------------------
# Konfigurasi umum
# -----------------------------
st.set_page_config(page_title="Pima Diabetes - MLOps GSLC", layout="wide")

DEFAULT_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
DEFAULT_TARGET = "Outcome"


# -----------------------------
# Utilitas caching
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_resource(show_spinner=False)
def train_pipeline(
    df: pd.DataFrame,
    features: Tuple[str, ...],
    target: str,
    test_size: float,
    random_state: int,
    C: float,
    class_weight: str | None
):
    X = df[list(features)]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=C,
            class_weight=class_weight,
            solver="lbfgs"
        ))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = None
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "report": classification_report(y_test, y_pred, zero_division=0, output_dict=False),
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred,
        "y_proba": y_proba
    }
    return pipe, metrics


# -----------------------------
# Komponen visual
# -----------------------------
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig


def plot_correlation(df: pd.DataFrame, cols: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[cols].corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation (numerical features)")
    return fig


def plot_distributions(df: pd.DataFrame, cols: list[str]) -> plt.Figure:
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()
    for i, c in enumerate(cols):
        sns.histplot(df[c], kde=True, ax=axes[i])
        axes[i].set_title(c)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return fig


# -----------------------------
# Aplikasi
# -----------------------------
st.title("Prediksi Diabetes (Pima) • GSLC MLOps")

with st.sidebar:
    st.header("Data Source")
    use_uploaded = st.toggle("Gunakan file upload", value=False)
    uploaded_file = None
    if use_uploaded:
        uploaded_file = st.file_uploader("Upload CSV (schema seperti diabetes.csv)", type=["csv"])

    st.header("Training Parameters")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
    C = st.slider("LogReg C (inverse regularization)", 0.01, 10.0, 1.0, 0.01)
    balanced = st.checkbox("class_weight='balanced'", value=True)
    class_weight = "balanced" if balanced else None

    st.header("Threshold")
    threshold = st.slider("Decision threshold (inference)", 0.05, 0.95, 0.50, 0.01)

# 1) Load data
df = None
if use_uploaded and uploaded_file is not None:
    df = load_uploaded_csv(uploaded_file)
else:
    # Mencoba memuat diabetes.csv dari repo
    if os.path.exists("diabetes.csv"):
        df = load_csv("diabetes.csv")
    else:
        st.info("Letakkan `diabetes.csv` di root repo, atau aktifkan 'Gunakan file upload' pada sidebar.")
        st.stop()

# Validasi minimal kolom
missing_cols = [c for c in DEFAULT_FEATURES + [DEFAULT_TARGET] if c not in df.columns]
if missing_cols:
    st.error(f"Kolom berikut tidak ditemukan di dataset: {missing_cols}")
    st.stop()

# 2) Tabs utama
tab_eda, tab_train, tab_infer = st.tabs(["Dataset & EDA", "Training", "Inference"])

with tab_eda:
    st.subheader("Ringkasan Dataset")
    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.write("Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.write("Deskripsi Statistik")
        st.dataframe(df[DEFAULT_FEATURES + [DEFAULT_TARGET]].describe().T, use_container_width=True)

    with c2:
        st.write("Info kolom")
        info_df = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "nulls": df.isna().sum(),
            "non_null": df.notna().sum()
        })
        st.dataframe(info_df, use_container_width=True, height=400)

    st.markdown("---")
    st.subheader("Distribusi Fitur Numerik")
    fig_dist = plot_distributions(df, DEFAULT_FEATURES)
    st.pyplot(fig_dist, clear_figure=True)

    st.subheader("Korelasi")
    fig_corr = plot_correlation(df, DEFAULT_FEATURES + [DEFAULT_TARGET])
    st.pyplot(fig_corr, clear_figure=True)

with tab_train:
    st.subheader("Training Logistic Regression")
    st.caption("Pipeline: StandardScaler → LogisticRegression")

    pipe, metrics = train_pipeline(
        df=df,
        features=tuple(DEFAULT_FEATURES),
        target=DEFAULT_TARGET,
        test_size=test_size,
        random_state=random_state,
        C=C,
        class_weight=class_weight
    )

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    colB.metric("Precision", f"{metrics['precision']:.3f}")
    colC.metric("Recall", f"{metrics['recall']:.3f}")
    colD.metric("F1-score", f"{metrics['f1']:.3f}")

    st.markdown("**Classification Report**")
    st.code(metrics["report"])

    st.markdown("**Confusion Matrix**")
    fig_cm = plot_confusion_matrix(metrics["y_test"], metrics["y_pred"])
    st.pyplot(fig_cm, clear_figure=True)

with tab_infer:
    st.subheader("Prediksi Individual")
    st.caption("Isi nilai fitur di bawah. Prediksi akan menggunakan pipeline terlatih di tab Training.")

    cols = st.columns(4)
    inputs: Dict[str, float] = {}

    inputs["Pregnancies"] = cols[0].number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    inputs["Glucose"] = cols[1].number_input("Glucose", min_value=0, max_value=300, value=120, step=1)
    inputs["BloodPressure"] = cols[2].number_input("BloodPressure", min_value=0, max_value=200, value=70, step=1)
    inputs["SkinThickness"] = cols[3].number_input("SkinThickness", min_value=0, max_value=99, value=20, step=1)

    cols2 = st.columns(4)
    inputs["Insulin"] = cols2[0].number_input("Insulin", min_value=0, max_value=900, value=80, step=1)
    inputs["BMI"] = cols2[1].number_input("BMI", min_value=0.0, max_value=80.0, value=28.0, step=0.1)
    inputs["DiabetesPedigreeFunction"] = cols2[2].number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    inputs["Age"] = cols2[3].number_input("Age", min_value=1, max_value=120, value=33, step=1)

    if st.button("Prediksi"):
        X_new = pd.DataFrame([inputs], columns=DEFAULT_FEATURES)
        # Menggunakan pipeline yang sudah ditraining di tab Training
        proba = None
        try:
            proba = pipe.predict_proba(X_new)[:, 1][0]
        except Exception:
            # Jika model tidak mendukung predict_proba
            proba = float(pipe.decision_function(X_new)[0])
            # Skala ke 0..1 secara sederhana (tidak ideal), tapi kasus LogReg mendukung predict_proba.
            proba = 1 / (1 + np.exp(-proba))

        pred = int(proba >= threshold)
        label = "Positif (1)" if pred == 1 else "Negatif (0)"

        m1, m2 = st.columns(2)
        m1.metric("Label", label)
        m2.metric("Probabilitas Positif", f"{proba:.3f}")

        st.info(f"Threshold saat ini: {threshold:.2f}. Ubah pada sidebar jika diperlukan.")
