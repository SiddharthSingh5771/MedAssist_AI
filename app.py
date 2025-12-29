import os
import pickle
import streamlit as st
import plotly.graph_objects as go # New library for graphs
from streamlit_option_menu import option_menu

# --- Page Configuration ---
st.set_page_config(
    page_title="Unified Health Diagnostic",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# --- 🎨 DARK MODE HIGH CONTRAST CSS ---
st.markdown("""
    <style>
        .stApp { background-color: #0e1117; color: #ffffff; }
        [data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #333; }
        .css-1r6slb0, .stForm { background-color: #1f2937; border: 1px solid #374151; padding: 25px; border-radius: 12px; }
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] { background-color: #374151; color: #ffffff !important; border: 1px solid #4b5563; border-radius: 6px; }
        .stMarkdown, p, h1, h2, h3, label { color: #e5e7eb !important; }
        ul[data-testid="stSelectboxVirtualDropdown"] li { background-color: #1f2937; color: white; }
        div.stButton > button { background: linear-gradient(45deg, #2563eb, #3b82f6); color: white; border: none; font-weight: bold; font-size: 18px; padding: 12px 24px; border-radius: 8px; transition: all 0.3s ease; }
        div.stButton > button:hover { transform: scale(1.02); box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39); }
        
        /* Custom Card Style */
        .st-card {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- 🚀 HERO HEADER HELPER ---
def hero_header(title, subtitle, icon="🏥", color="#3b82f6"):
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {color}22, transparent); padding: 20px; border-radius: 10px; border-left: 5px solid {color}; margin-bottom: 20px;">
        <h1 style="color: white; margin:0; font-size: 40px;">{icon} {title}</h1>
        <p style="color: #d1d5db; font-size: 18px; margin:0;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# --- Model Loading ---
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model_path = os.path.join(working_dir, 'saved_models', 'diabetes_model.sav')
heart_model_path = os.path.join(working_dir, 'saved_models', 'heart_model.sav')

diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))
heart_disease_model = pickle.load(open(heart_model_path, 'rb'))

# --- Sidebar ---
# --- Sidebar ---
with st.sidebar:
    st.image("assets/medassist_logo.svg", width=100)
    st.title("MedAssist AI")
    st.caption("🤖 Advanced Health Diagnostics")
    
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Select Disease Predictor",
        options=['Diabetes Prediction', 'Heart Disease Prediction'],
        icons=['droplet-fill', 'heart-pulse-fill'],
        menu_icon='activity',  # Changed from 'cast' to 'activity' for medical relevance
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#3b82f6", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#1f2937",
                "color": "#e5e7eb" # Match global text color
            },
            "nav-link-selected": {"background-color": "#2563eb", "color": "white", "font-weight": "600"},
            "menu-title": {"color": "#ffffff", "font-size": "18px", "font-weight": "bold", "margin-bottom": "10px"}
        }
    )
    
    st.markdown("---")
    
    # About Section
    with st.expander("ℹ️ About", expanded=False):
        st.markdown('''
        **MedAssist AI** uses machine learning to predict health risks.
        
        - 🧠 **Diabetes Model**: 85% Accuracy
        - 💓 **Heart Model**: 88% Accuracy
        
        *Built with Streamlit & Python.*
        ''')

# --- 📊 HELPER FUNCTION FOR GAUGE CHART ---
def create_gauge_chart(probability):
    # Determine color based on risk
    if probability < 30:
        color = "green"
    elif probability < 70:
        color = "yellow"
    else:
        color = "red"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Probability (%)", 'font': {'size': 24, 'color': "white"}},
        number = {'font': {'color': "white"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [30, 70], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Arial"})
    return fig


# --- Diabetes Prediction Page ---
if selected == 'Diabetes Prediction':
    
    hero_header("Diabetes Risk Assessment", "Predict the likelihood of diabetes using clinical metrics.", icon="🩸")

    # Group 1: Demographics & Gender
    st.markdown('<div class="st-card">', unsafe_allow_html=True)
    st.markdown("#### 👤 Patient Profile")
    gender = st.radio("Gender", ["Female", "Male"], index=None, horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.form("diabetes_form"):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("### � Demographics & indices")
            Pregnancies = 0 
            if gender == "Female":
                Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1, value=0)
            elif gender == "Male":
                st.info("ℹ️ Pregnancy input disabled for Male patients.")
            else:
                st.warning("Please select a Gender above to enable demographics.")
            
            Age = st.number_input('Age (Years)', min_value=0, max_value=120, step=1, value=0)
            BMI = st.number_input('BMI Index', min_value=0.0, max_value=70.0, step=0.1, format="%.1f", value=0.0)
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.01, format="%.3f", value=0.0)

        with col2:
            st.markdown("### 🧪 Clinical Vitals")
            Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=500, step=1, value=0)
            BloodPressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=200, step=1, value=0)
            SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, step=1, value=0)
            Insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900, step=1, value=0)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 ANALYZE RISK", type="primary")

        if submitted:
            if gender is None:
                st.error("⚠️ Please select the Patient Gender.")
            elif Age == 0 or Glucose == 0 or BMI == 0:
                st.error("⚠️ Age, Glucose, and BMI cannot be 0.")
            else:
                user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
                
                # Get Probability instead of just class
                prediction_prob = diabetes_model.predict_proba([user_input])[0][1] * 100
                prediction_class = diabetes_model.predict([user_input])[0]

                # Create Layout: Text on Left, Graph on Right
                st.markdown("---")
                st.markdown("### 🏥 Analysis Results")
                
                res_col1, res_col2 = st.columns([2, 1])
                
                with res_col1:
                    if prediction_class == 1:
                        st.markdown(f"""
                        <div style="background-color: rgba(153, 27, 27, 0.4); border: 2px solid #ef4444; border-radius: 10px; padding: 20px;">
                            <h2 style="color: #fca5a5; margin:0;">⚠️ POSITIVE (High Risk)</h2>
                            <p style="color: white; font-size: 18px;">
                                The AI model estimates a <strong>{prediction_prob:.1f}% probability</strong> of diabetes.
                            </p>
                            <p style="color: #e5e7eb;">Recommended Actions:</p>
                            <ul style="color: #e5e7eb;">
                                <li>Consult an Endocrinologist immediately.</li>
                                <li>Schedule an HbA1c test.</li>
                                <li>Monitor blood sugar levels daily.</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: rgba(22, 101, 52, 0.4); border: 2px solid #22c55e; border-radius: 10px; padding: 20px;">
                            <h2 style="color: #86efac; margin:0;">✅ NEGATIVE (Low Risk)</h2>
                            <p style="color: white; font-size: 18px;">
                                The AI model estimates only a <strong>{prediction_prob:.1f}% probability</strong> of diabetes.
                            </p>
                            <p style="color: #e5e7eb;">Maintain a healthy lifestyle to keep risk low.</p>
                        </div>
                        """, unsafe_allow_html=True)

                with res_col2:
                    # Display the Gauge Chart
                    fig = create_gauge_chart(prediction_prob)
                    st.plotly_chart(fig, use_container_width=True)


# --- Heart Disease Prediction Page ---
if selected == 'Heart Disease Prediction':

    hero_header("Heart Disease Assessment", "Evaluate cardiovascular health metrics.", icon="💓", color="#ef4444")

    st.markdown('<div class="st-card">', unsafe_allow_html=True)
    st.markdown("#### 👤 Patient Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, step=1, value=0)
    with col2:
        sex_option = st.selectbox('Sex', ['Male', 'Female'], index=None, placeholder="Select...")
    with col3:
        cp_option = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'], index=None, placeholder="Select...")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.form("heart_form"):
        st.markdown("### 🧪 Clinical Vitals & Tests")
        col4, col5, col6 = st.columns(3)
        with col4:
            trestbps = st.number_input('Resting BP (mm Hg)', min_value=0, max_value=250, step=1, value=0)
            chol = st.number_input('Cholesterol (mg/dl)', min_value=0, max_value=600, step=1, value=0)
        with col5:
            fbs_option = st.selectbox('Fasting Blood Sugar > 120 mg/dl?', ['False', 'True'], index=None, placeholder="Select...")
            thalach = st.number_input('Max Heart Rate', min_value=0, max_value=250, step=1, value=0)
        with col6:
            restecg_option = st.selectbox('Resting ECG', ['Normal', 'ST-T Wave Abnormality', 'LV Hypertrophy'], index=None, placeholder="Select...")
            exang_option = st.selectbox('Exercise Induced Angina?', ['No', 'Yes'], index=None, placeholder="Select...")

        st.markdown("---")
        st.markdown("### 🏃 Stress Test Results")
        col7, col8, col9 = st.columns(3)
        with col7:
            oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=10.0, step=0.1, value=0.0)
        with col8:
            slope = st.number_input('Slope of Peak ST', min_value=0, max_value=2, step=1, value=0)
            ca = st.number_input('Major Vessels (0-3)', min_value=0, max_value=3, step=1, value=0)
        with col9:
            thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversable Defect'], index=None, placeholder="Select...")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted_heart = st.form_submit_button("💓 ANALYZE HEART HEALTH", type="primary")

        if submitted_heart:
            if None in [sex_option, cp_option, fbs_option, restecg_option, exang_option, thal]:
                st.error("⚠️ Please select an option for all dropdown fields.")
            elif age == 0:
                st.error("⚠️ Age cannot be 0.")
            else:
                sex = 1 if sex_option == 'Male' else 0
                cp = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}[cp_option]
                fbs = 1 if fbs_option == 'True' else 0
                restecg = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'LV Hypertrophy': 2}[restecg_option]
                exang = 1 if exang_option == 'Yes' else 0
                thal_val = {'Normal': 0, 'Fixed Defect': 1, 'Reversable Defect': 2}[thal]

                user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal_val]
                
                # Get Probability
                prediction_prob = heart_disease_model.predict_proba([user_input])[0][1] * 100
                prediction_class = heart_disease_model.predict([user_input])[0]

                st.markdown("---")
                st.markdown("### 🏥 Analysis Results")
                res_col1, res_col2 = st.columns([2, 1])

                with res_col1:
                    if prediction_class == 1:
                        st.markdown(f"""
                        <div style="background-color: rgba(153, 27, 27, 0.4); border: 2px solid #ef4444; border-radius: 10px; padding: 20px;">
                            <h2 style="color: #fca5a5; margin:0;">⚠️ HEART DISEASE DETECTED</h2>
                            <p style="color: white; font-size: 18px;">
                                The AI model estimates a <strong>{prediction_prob:.1f}% probability</strong> of heart disease.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: rgba(22, 101, 52, 0.4); border: 2px solid #22c55e; border-radius: 10px; padding: 20px;">
                            <h2 style="color: #86efac; margin:0;">✅ HEART IS HEALTHY</h2>
                            <p style="color: white; font-size: 18px;">
                                The AI model estimates only a <strong>{prediction_prob:.1f}% probability</strong> of heart disease.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                with res_col2:
                    fig = create_gauge_chart(prediction_prob)
                    st.plotly_chart(fig, use_container_width=True)