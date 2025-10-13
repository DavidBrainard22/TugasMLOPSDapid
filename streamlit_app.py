import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =======================
# PAGE CONFIGURATION
# =======================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =======================
# CUSTOM STYLE
# =======================
st.markdown("""
<style>
    /* Background Gradient */
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
    }
    
    /* Header Styling */
    .main-header {
        font-size: 36px;
        font-weight: 900;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 30px;
        padding: 30px;
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(30, 64, 175, 0.15);
        border: 3px solid #e2e8f0;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #dc2626 0%, #ea580c 50%, #16a34a 100%);
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #dc2626 0%, #ea580c 100%);
        border-radius: 12px;
        padding: 30px;
        color: white;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        border: none;
    }

    .kpi-value {
        font-size: 40px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .kpi-label {
        font-size: 18px;
        opacity: 0.95;
        font-weight: 600;
    }

    /* Status Indicator */
    .status-card {
        background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        border: 2px solid #e2e8f0;
        border-left: 5px solid #dc2626;
    }
    
    /* Button Styling */
    .stButton button {
        background: linear-gradient(135deg, #dc2626 0%, #ea580c 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 15px 30px;
        font-weight: 700;
        font-size: 18px;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(220, 38, 38, 0.4);
    }

    /* Metric styling */
    .metric-container {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }

    /* Section headers */
    .section-header {
        font-size: 26px;
        font-weight: 700;
        color: #1e293b;
        margin: 25px 0 20px 0;
        padding: 20px;
        background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        border-left: 6px solid #dc2626;
    }
</style>
""", unsafe_allow_html=True)

# =======================
# LOAD AND PREPARE DATA
# =======================
@st.cache_data
def load_data():
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    df = pd.DataFrame(X, columns=feature_names)
    df['Outcome'] = y
    
    # Adjust feature scales to match realistic ranges for diabetes
    df['Pregnancies'] = (df['Pregnancies'] * 2 + 3).astype(int)
    df['Glucose'] = (df['Glucose'] * 50 + 100).astype(int)
    df['BloodPressure'] = (df['BloodPressure'] * 20 + 70).astype(int)
    df['SkinThickness'] = (df['SkinThickness'] * 15 + 20).astype(int)
    df['Insulin'] = (df['Insulin'] * 100 + 80).astype(int)
    df['BMI'] = (df['BMI'] * 10 + 25).round(1)
    df['DiabetesPedigreeFunction'] = (df['DiabetesPedigreeFunction'] * 0.5 + 0.3).round(3)
    df['Age'] = (df['Age'] * 20 + 25).astype(int)
    
    return df

# =======================
# MODEL TRAINING
# =======================
@st.cache_resource
def train_model(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = VotingClassifier(
        estimators=[
            ('SVC', SVC(kernel='linear', C=0.1, gamma=10, probability=True, random_state=42)),
            ('LR', LogisticRegression(C=0.1, random_state=42)),
            ('RF', RandomForestClassifier(n_estimators=100, random_state=42))
        ],
        voting='soft'
    )
    
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test_scaled, y_test

# =======================
# DASHBOARD HEADER
# =======================
st.markdown("""
<div class="main-header">
    <div>🩺 Diabetes Prediction System</div>
    <div>Advanced Machine Learning for Early Diabetes Detection</div>
</div>
""", unsafe_allow_html=True)

# =======================
# KPI AND STATUS CARD
# =======================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="status-card">
        <div style="font-size: 18px;">System Status</div>
        <div style="font-size: 22px; font-weight: 700; color: #16a34a;">🟢 Active & Optimal</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="status-card">
        <div style="font-size: 18px;">Model Version</div>
        <div style="font-size: 22px; font-weight: 700; color: #dc2626;">v3.0.1</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="status-card">
        <div style="font-size: 18px;">Data Size</div>
        <div style="font-size: 22px; font-weight: 700; color: #7c3aed;">1,000 Records</div>
    </div>
    """, unsafe_allow_html=True)

# =======================
# PREDICTION SECTION
# =======================
st.markdown('<div class="section-header">💡 Prediction Panel</div>', unsafe_allow_html=True)

# Input Fields
pregnancies = st.slider("Pregnancies", 0, 20, 5)
glucose = st.slider("Glucose Level", 50, 200, 120)
blood_pressure = st.slider("Blood Pressure", 40, 120, 70)
skin_thickness = st.slider("Skin Thickness", 10, 60, 30)
insulin = st.slider("Insulin Level", 0, 300, 80)
bmi = st.slider("BMI", 15.0, 40.0, 25.5)
diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 18, 80, 33)

# Predict on button click
if st.button("Run Diabetes Risk Assessment"):
    df = load_data()
    model, scaler, X_test_scaled, y_test = train_model(df)
    
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    input_scaled = scaler.transform(input_data)
    
    prediction_proba = model.predict_proba(input_scaled)[0]
    diabetes_prob = prediction_proba[1] * 100

    st.markdown(f"**Diabetes Probability**: {diabetes_prob:.2f}%")
    
    if diabetes_prob >= 70:
        risk_status = "High Risk"
        risk_color = "#dc2626"
    elif diabetes_prob >= 40:
        risk_status = "Moderate Risk"
        risk_color = "#ea580c"
    else:
        risk_status = "Low Risk"
        risk_color = "#16a34a"
    
    st.markdown(f"""
    <div style="background: {risk_color}; padding: 15px; color: white; border-radius: 10px;">
        <div style="font-size: 24px; font-weight: 700;">{risk_status}</div>
        <div style="font-size: 16px;">Risk Assessment Completed</div>
    </div>
    """, unsafe_allow_html=True)
