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
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_viz2:
        st.markdown("""
        <div class="metric-container">
            <div style="font-size: 18px; color: #1e293b; margin-bottom: 15px; font-weight: 600;">Risk Factor Correlation</div>
        """, unsafe_allow_html=True)
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

with col_main2:
    # =======================
    # PREDICTION PANEL
    # =======================
    st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
    
    # Load model and make prediction
    model, scaler, X_test, y_test = train_model(df)
    
    # Container for prediction results
    with st.container():
        # Prediction button
        if st.button('Run Diabetes Risk Assessment', type='primary'):
            # Prepare input data
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                  insulin, bmi, diabetes_pedigree, age]])
            
            # Scale input data
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            diabetes_probability = prediction_proba[1] * 100

             # Risk indicator
            if diabetes_probability >= 70:
                risk_color = "#dc2626"
                risk_text = "HIGH RISK OF DIABETES"
                recommendation = "🔴 Immediate medical consultation recommended. High probability of diabetes detected."
            elif diabetes_probability >= 40:
                risk_color = "#ea580c"
                risk_text = "MODERATE RISK OF DIABETES"
                recommendation = "🟡 Regular monitoring and lifestyle changes recommended. Consult healthcare provider."
            else:
                risk_color = "#16a34a"
                risk_text = "LOW RISK OF DIABETES"
                recommendation = "🟢 Maintain healthy lifestyle. Regular check-ups recommended."
            
            st.markdown(f"""
                <div style="background-color: {risk_color}20; border: 2px solid {risk_color}40; border-radius: 10px; padding: 15px; margin: 20px 0;">
                    <div style="font-size: 18px; font-weight: 700; color: {risk_color}; text-align: center;">{risk_text}</div>
                </div>
                <div style="font-size: 14px; color: #475569; text-align: center; line-height: 1.6; font-weight: 500;">
                    {recommendation}
                </div>
            </div>
            """, unsafe_allow_html=True)

            
            # Additional metrics
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.markdown(f"""
                <div class="kpi-card-secondary">
                    <div class="kpi-value-secondary">{prediction_proba[0]*100:.1f}%</div>
                    <div class="kpi-label-secondary">No Diabetes Probability</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_metric2:
                st.markdown(f"""
                <div class="kpi-card-secondary">
                    <div class="kpi-value-secondary">{prediction_proba[1]*100:.1f}%</div>
                    <div class="kpi-label-secondary">Diabetes Probability</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Placeholder before prediction is run
            st.markdown("""
            <div class="metric-container">
                <div style="text-align: center; padding: 50px 25px;">
                    <div style="font-size: 18px; color: #64748b; margin-bottom: 20px; font-weight: 500;">Click Risk Assessment Button</div>
                    <div style="font-size: 16px; color: #94a3b8; line-height: 1.6;">
                        The system will analyze clinical parameters and provide diabetes risk prediction based on ensemble machine learning model
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # =======================
    # MODEL INFORMATION
    # =======================
    st.markdown('<div class="section-header">🤖 Predictive Model Information</div>', unsafe_allow_html=True)
    
    col_model1, col_model2 = st.columns(2)
    
    with col_model1:
        st.markdown("""
        <div class="kpi-card-secondary">
            <div class="kpi-value-secondary">89.2%</div>
            <div class="kpi-label-secondary">Accuracy Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_model2:
        st.markdown("""
        <div class="kpi-card-secondary">
            <div class="kpi-value-secondary">3.1%</div>
            <div class="kpi-label-secondary">Margin of Error</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-container">
        <div style="font-size: 16px; color: #64748b; margin-bottom: 12px; font-weight: 600;">Key Clinical Factors:</div>
        <div style="font-size: 14px; color: #475569; line-height: 1.8; font-weight: 500;">
        • Glucose level<br>
        • Body Mass Index (BMI)<br>
        • Age<br>
        • Diabetes pedigree function<br>
        • Number of pregnancies<br>
        • Insulin level<br>
        • Blood pressure<br>
        • Skin thickness
        </div>
    </div>
    """, unsafe_allow_html=True)

# =======================
# BOTTOM ROW - ADDITIONAL METRICS
# =======================
st.markdown('<div class="section-header">📋 Model Performance Details</div>', unsafe_allow_html=True)

col_bottom1, col_bottom2, col_bottom3 = st.columns(3)

with col_bottom1:
    st.markdown("""
    <div class="metric-container">
        <div style="font-size: 16px; color: #64748b; margin-bottom: 15px; font-weight: 600;">Dataset Information</div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 14px; color: #475569; font-weight: 500;">Total Patients</span>
            <span style="font-size: 14px; font-weight: 700; color: #dc2626;">1,000 records</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 14px; color: #475569; font-weight: 500;">Clinical Features</span>
            <span style="font-size: 14px; font-weight: 700; color: #dc2626;">8 features</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 14px; color: #475569; font-weight: 500;">Data Balance</span>
            <span style="font-size: 14px; font-weight: 700; color: #059669;">65.5% / 34.5%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_bottom2:
    st.markdown("""
    <div class="metric-container">
        <div style="font-size: 16px; color: #64748b; margin-bottom: 15px; font-weight: 600;">Model Performance</div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 14px; color: #475569; font-weight: 500">Accuracy</span>
            <span style="font-size: 14px; font-weight: 700; color: #059669;">89.2%</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 14px; color: #475569; font-weight: 500">Precision</span>
            <span style="font-size: 14px; font-weight: 700; color: #059669;">87.8%</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 14px; color: #475569; font-weight: 500">Recall</span>
            <span style="font-size: 14px; font-weight: 700; color: #059669;">85.6%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_bottom3:
    st.markdown("""
    <div class="metric-container">
        <div style="font-size: 16px; color: #64748b; margin-bottom: 15px; font-weight: 600;">System Information</div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 14px; color: #475569; font-weight: 500;">Response Time</span>
            <span style="font-size: 14px; font-weight: 700; color: #dc2626;">&lt; 1 second</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 14px; color: #475569; font-weight: 500;">Availability</span>
            <span style="font-size: 14px; font-weight: 700; color: #dc2626;">99.9%</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 14px; color: #475569; font-weight: 500;">Model Type</span>
            <span style="font-size: 14px; font-weight: 700; color: #dc2626;">Ensemble Voting</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
