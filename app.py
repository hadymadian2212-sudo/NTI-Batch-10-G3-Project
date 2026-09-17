import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Telco Customer Churn Predictor", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp {
        background-color: #0c3336;
    }
    h1, h2, h3, h4, h5, h6, span {
        color: #f6efe2 !important;
    }
    div[data-testid="stForm"] {
        background-color: #f6efe2;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    div[data-testid="stForm"] p, div[data-testid="stForm"] label, div[data-testid="stForm"] div {
        color: #0c3336 !important;
    }
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #c29b40;
        color: white !important;
        font-weight: bold;
        border-radius: 12px;
        border: none;
        width: 100%;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #a88532;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>Workspace: Telco Customer Churn Predictor</h2>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

with st.form("churn_prediction_form"):
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("#### 1. Customer Profile")
        
        st.markdown("**Demographic Status**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.radio("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        st.markdown("<hr style='border-top: 1px solid #d4cbb8;'>", unsafe_allow_html=True)
        
        partner = st.radio("Partner (Yes/No)", ["Yes", "No"], horizontal=True)
        
        st.markdown("<hr style='border-top: 1px solid #d4cbb8;'>", unsafe_allow_html=True)
        
        st.markdown("**Plan Selected & Billing**")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless_billing = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)

    with col2:
        st.markdown("#### 2. Usage & History")
        
        st.markdown("**Service Access**")
        c2_1, c2_2 = st.columns(2)
        with c2_1:
            internet_service = st.selectbox("Internet (DSL/Fiber)", ["DSL", "Fiber optic", "No"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        with c2_2:
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            online_security = st.selectbox("Security Package", ["No", "Yes", "No internet service"])
            
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        
        st.markdown("<hr style='border-top: 1px solid #d4cbb8;'>", unsafe_allow_html=True)
        
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        
        charge_col1, charge_col2 = st.columns(2)
        with charge_col1:
            monthly_charges = st.number_input("Est. Monthly Charges", min_value=0.0, value=70.0, step=0.5)
        with charge_col2:
            total_charges = st.number_input("Total Charges", min_value=0.0, value=850.0, step=1.0)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        submit_button = st.form_submit_button(label="🚀 Rocket Predict Churn")

if submit_button:
    input_df = pd.DataFrame([{
        'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 
        'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service, 
        'MultipleLines': multiple_lines, 'InternetService': internet_service, 
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup, 
        'DeviceProtection': device_protection, 'TechSupport': tech_support, 
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies, 
        'Contract': contract, 'PaperlessBilling': paperless_billing, 
        'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }])
    
    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0]
    stay_prob = proba[0] * 100
    churn_prob = proba[1] * 100
    
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>Prediction Dashboard</h3>", unsafe_allow_html=True)
    
    res_col1, res_col2, res_col3 = st.columns([1, 2, 2])
    
    with res_col2:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = stay_prob,
            number = {'suffix': "%", 'font': {'size': 40, 'color': '#f6efe2'}},
            title = {'text': "Stay Probability", 'font': {'size': 20, 'color': '#f6efe2'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#f6efe2"},
                'bar': {'color': "#c29b40"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': "#0c3336"},
                    {'range': [50, 100], 'color': "#14595e"}
                ]
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f6efe2"}, height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col3:
        fig_bar = go.Figure(data=[
            go.Bar(name='Stay', x=['Status'], y=[stay_prob], marker_color='#c29b40'),
            go.Bar(name='Churn', x=['Status'], y=[churn_prob], marker_color='#0c3336')
        ])
        fig_bar.update_layout(
            title="Probability % (Churn vs. Stay)",
            barmode='group',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(246, 239, 226, 0.1)",
            font={'color': "#f6efe2"},
            height=300
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<h4 style='text-align: center; margin-top: 20px;'>Final Decision</h4>", unsafe_allow_html=True)
    
    decision_col1, decision_col2, decision_col3 = st.columns([1, 2, 1])
    
    with decision_col2:
        if prediction == 1:
            st.error("⚠️ **High Risk:** The model predicts that this customer is likely to **CHURN**.")
        else:
            st.success("✅ **Low Risk:** The model predicts that this customer is likely to **STAY**.")