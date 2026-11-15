import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
    /* 100% Aggressive removal of +/- buttons */
    input[type="number"] {
        -moz-appearance: textfield !important;
        appearance: textfield !important;
    }
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button {
        -webkit-appearance: none !important;
        appearance: none !important;
        margin: 0 !important;
        display: none !important;
    }

    /* Hide ALL placeholder text completely */
    input::placeholder {
        color: transparent !important;
        opacity: 0 !important;
    }

    /* Olive Dark Theme Background */
    .stApp {
        background-color: #1a1c14 !important;
        color: #e0e0e0 !important;
    }
    
    /* Input Fields Styling */
    .stNumberInput > div > div > input, 
    .stSelectbox > div > div > div > select {
        background-color: #2a2e22 !important;
        color: #ffffff !important;
        border: 1px solid #4a5238 !important;
        border-radius: 6px !important;
        padding: 10px !important;
    }
    
    /* Labels Styling */
    .stNumberInput label, .stSelectbox label {
        color: #a3b18a !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Button Styling */
    .stButton > button {
        background-color: #6b8e23 !important;
        color: #ffffff !important;
        font-weight: bold !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 12px !important;
        width: 100% !important;
        margin-top: 10px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background-color: #556b2f !important;
        box-shadow: 0 4px 8px rgba(107, 142, 35, 0.4) !important;
    }

    /* Result Boxes */
    .result-box {
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: bold;
        margin-top: 2rem;
        letter-spacing: 0.5px;
    }
    .safe {
        background-color: #2a3b1e;
        color: #a3b18a;
        border: 1px solid #4a5d23;
    }
    .fraud {
        background-color: #3b1e1e;
        color: #d98c8c;
        border: 1px solid #5d2323;
    }
    
    h1 { color: #ffffff !important; text-align: center; margin-bottom: 1rem; }
    p { color: #888888 !important; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Fraud Detection System")
st.caption("Enter transaction details to verify its authenticity.")

@st.cache_resource
def load_model():
    return joblib.load('Base_model.pkl')

model = load_model()

col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Time Step", min_value=1, value=1, step=1)
    t_type = st.selectbox("Transaction Type", [0, 1, 2, 3, 4], format_func=lambda x: ['Cash In', 'Cash Out', 'Debit', 'Payment', 'Transfer'][x])
    amount = st.number_input("Amount", min_value=0.0, value=0.0, step=100.0)
    old_bal_org = st.number_input("Sender's Old Balance", min_value=0.0, value=0.0, step=1000.0)

with col2:
    new_bal_org = st.number_input("Sender's New Balance", min_value=0.0, value=0.0, step=1000.0)
    old_bal_dest = st.number_input("Receiver's Old Balance", min_value=0.0, value=0.0, step=1000.0)
    new_bal_dest = st.number_input("Receiver's New Balance", min_value=0.0, value=0.0, step=1000.0)
    flagged = st.selectbox("Bank Flagged? (Yes/No)", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

if st.button("Check Transaction"):
    data = pd.DataFrame({
        'step': [step], 
        'type': [t_type], 
        'amount': [amount], 
        'oldbalanceOrg': [old_bal_org],
        'newbalanceOrig': [new_bal_org], 
        'oldbalanceDest': [old_bal_dest], 
        'newbalanceDest': [new_bal_dest],
        'isFlaggedFraud': [flagged]
    })

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1] * 100

    if pred == 1:
        st.markdown(f'<div class="result-box fraud">⚠️ FRAUD DETECTED! (Confidence: {prob:.2f}%)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box safe">✅ SAFE TRANSACTION (Confidence: {100 - prob:.2f}%)</div>', unsafe_allow_html=True)