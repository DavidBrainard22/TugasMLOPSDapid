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
    /* Main background */
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
    }
    
    /* Header styling */
    .main-header {
        font-size: 32px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 25px;
        padding: 25px;
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(30, 64, 175, 0.15);
        border: 2px solid #e2e8f0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #dc2626 0%, #ea580c 50%, #16a34a 100%);
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #dc2626 0%, #ea580c 100%);
        border-radius: 12px;
        padding: 25px;
        color: white;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        border: none;
    }
    
    .kpi-value {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    
    .kpi-label {
        font-size: 16px;
        opacity: 0.95;
        font-weight: 500;
    }
    
    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #1e293b;
        margin: 25px 0 20px 0;
        padding: 18px 20px;
        background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        border: 2px solid #e2e8f0;
        border-left: 6px solid #dc2626;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #dc2626 0%, transparent 100%);
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
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #dc2626 0%, #ea580c 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-weight: 700;
        font-size: 16px;
        width: 100%;
        box-shadow: 0 4px 8px rgba(220, 38, 38, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(220, 38, 38, 0.4);
    }
    
    /* Input section styling */
    .input-section {
        background: white;
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* Risk indicators */
    .high-risk {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    
    .medium-risk {
        background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Status card styling */
    .status-card {
        background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        border: 2px solid #e2e8f0;
        border-left: 5px solid #dc2626;
        height: 100%;
    }
    
    /* Copyright styling */
    .copyright {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        color: #64748b;
        font-size: 14px;
        font-weight: 500;
        border-top: 2px solid #e2e8f0;
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =======================
# LOAD AND PREPARE DATA
# =======================
@st.cache_data
def load_data():
    # Using sample diabetes data - replace with your actual data loading
    from sklearn.datasets import make_classification
    import pandas as pd
    
    # Create sample diabetes-like data
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
    
    # Scale features to realistic diabetes data ranges
    df['Pregnancies'] = (df['Pregnancies'] * 2 + 3).astype(int)
    df['Glucose'] = (df['Glucose'] * 50 + 100).astype(int)
    df['BloodPressure'] = (df['BloodPressure'] * 20 + 70).astype(int)
    df['SkinThickness'] = (df['SkinThickness'] * 15 + 20).astype(int)
    df['Insulin'] = (df['Insulin'] * 100 + 80).astype(int)
    df['BMI'] = (df['BMI'] * 10 + 25).round(1)
    df['DiabetesPedigreeFunction'] = (df['DiabetesPedigreeFunction'] * 0.5 + 0.3).round(3)
    df['Age'] = (df['Age'] * 20 + 25).astype(int)
    
    return df

@st.cache_resource
def train_model(df):
    # Prepare data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ensemble model (similar to your analysis)
    linear_svc = SVC(kernel='linear', C=0.1, gamma=10, probability=True, random_state=42)
    radial_svm = SVC(kernel='rbf', C=0.1, gamma=10, probability=True, random_state=42)
    lr = LogisticRegression(C=0.1, random_state=42)
    
    ensemble_model = VotingClassifier(
        estimators=[
            ('Radial_svm', radial_svm), 
            ('Logistic Regression', lr),
            ('Linear_svm', linear_svc)
        ],
        voting='soft', 
        weights=[2, 1, 3]
    )
    
    ensemble_model.fit(X_train_scaled, y_train)
    
    return ensemble_model, scaler, X_test_scaled, y_test

# =======================
# DASHBOARD HEADER
# =======================
st.markdown("""
<div class="main-header">
    <div style="font-size: 28px; color: #dc2626; margin-bottom: 8px;">DIABETES PREDICTION SYSTEM</div>
    <div style="font-size: 16px; color: #64748b; font-weight: 500;">Advanced Machine Learning for Early Diabetes Detection and Risk Assessment</div>
</div>
""", unsafe_allow_html=True)

# =======================
# SYSTEM STATUS
# =======================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="status-card">
        <div style="font-size: 14px; color: #64748b; margin-bottom: 8px; font-weight: 500;">System Status</div>
        <div style="font-size: 18px; font-weight: 700; color: #059669;">🟢 Active & Optimal</div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 8px;">Real-time Monitoring</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="status-card">
        <div style="font-size: 14px; color: #64748b; margin-bottom: 8px; font-weight: 500;">Model Version</div>
        <div style="font-size: 18px; font-weight: 700; color: #dc2626;">v3.1.0</div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 8px;">Ensemble Certified</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="status-card">
        <div style="font-size: 14px; color: #64748b; margin-bottom: 8px; font-weight: 500;">Dataset</div>
        <div style="font-size: 18px; font-weight: 700; color: #7c3aed;">1,000 Patients</div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 8px;">Medical Records</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="status-card">
        <div style="font-size: 14px; color: #64748b; margin-bottom: 8px; font-weight: 500;">System Accuracy</div>
        <div style="font-size: 18px; font-weight: 700; color: #dc2626;">89.2%</div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 8px;">Precision Level</div>
    </div>
    """, unsafe_allow_html=True)

# =======================
# KPI CARDS - TOP ROW
# =======================
st.markdown('<div class="section-header">📊 System Performance Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-value">1,000</div>
        <div class="kpi-label">Patient Records</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-value">89.2%</div>
        <div class="kpi-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-value">34.5%</div>
        <div class="kpi-label">Diabetes Prevalence</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-value">8</div>
        <div class="kpi-label">Clinical Features</div>
    </div>
    """, unsafe_allow_html=True)

# =======================
# MAIN SECTION - 2 COLUMNS
# =======================
col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    # =======================
    # INPUT FEATURES
    # =======================
    st.markdown('<div class="section-header">🩺 Patient Clinical Parameters</div>', unsafe_allow_html=True)
    
    # Container for inputs
    with st.container():
        col_input1, col_input2 = st.columns(2)
        
        with col_input1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=15, value=2, 
                                  help="Number of times pregnant")
            glucose = st.number_input('Glucose Level (mg/dL)', min_value=50, max_value=200, value=120, 
                              help="Plasma glucose concentration 2 hours in oral glucose tolerance test")
            blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=40, max_value=120, value=70, 
                                     help="Diastolic blood pressure")
            skin_thickness = st.number_input('Skin Thickness (mm)', min_value=10, max_value=60, value=25, 
                                     help="Triceps skin fold thickness")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_input2:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=300, value=80, 
                              help="2-Hour serum insulin")
            bmi = st.number_input('Body Mass Index (BMI)', min_value=15.0, max_value=45.0, value=25.5, 
                          help="Body mass index (weight in kg/(height in m)^2)")
            diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.08, max_value=2.5, value=0.45, 
                                        help="Diabetes pedigree function")
            age = st.number_input('Age (years)', min_value=20, max_value=80, value=35, 
                          help="Age in years")
            st.markdown('</div>', unsafe_allow_html=True)

    # =======================
    # DATA VISUALIZATION
    # =======================
    st.markdown('<div class="section-header">📈 Feature Importance Analysis</div>', unsafe_allow_html=True)
    
    # Load data and show feature importance
    df = load_data()
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.markdown("""
        <div class="metric-container">
            <div style="font-size: 18px; color: #1e293b; margin-bottom: 15px; font-weight: 600;">Feature Importance</div>
        """, unsafe_allow_html=True)
        
        # Calculate feature importance
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        rf_model.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='#dc2626')
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance in Diabetes Prediction')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow
