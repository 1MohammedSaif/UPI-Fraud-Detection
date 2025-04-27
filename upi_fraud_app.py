import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Set page config
st.set_page_config(page_title="UPI Fraud Detection", layout="wide")

# Title and description
st.title("UPI Fraud Detection System")
st.markdown("""
This application helps detect fraudulent UPI transactions using a pre-trained XGBoost model.
""")

# Load the pre-trained model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('upi_fraud_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        st.error("Could not load the pre-trained model. Please make sure 'upi_fraud_model.pkl' and 'scaler.pkl' exist in the directory.")
        return None, None

model, scaler = load_model()

# Load data for options
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Copy of Sample_DATA.csv")
        return df
    except:
        st.error("Please make sure the data file 'Copy of Sample_DATA.csv' is in the same directory")
        return None

df = load_data()

if df is not None and model is not None:
    # Create input form
    st.header("Fraud Prediction")
    st.subheader("Enter Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Transaction Amount", min_value=0.0)
        transaction_frequency = st.number_input("Transaction Frequency", min_value=0)
        transaction_amount_deviation = st.number_input("Transaction Amount Deviation")
    
    with col2:
        transaction_type = st.selectbox("Transaction Type", df['Transaction_Type'].unique())
        payment_gateway = st.selectbox("Payment Gateway", df['Payment_Gateway'].unique())
        device_os = st.selectbox("Device OS", df['Device_OS'].unique())
    
    if st.button("Predict Fraud"):
        # Prepare input data
        input_data = pd.DataFrame({
            'amount': [amount],
            'Transaction_Frequency': [transaction_frequency],
            'Transaction_Amount_Deviation': [transaction_amount_deviation],
            'Transaction_Type': [transaction_type],
            'Payment_Gateway': [payment_gateway],
            'Device_OS': [device_os]
        })
        
        # Encode categorical variables
        for col in input_data.select_dtypes(include=['object']).columns:
            input_data[col] = LabelEncoder().fit_transform(input_data[col])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Display results
        if prediction == 1:
            st.error(f"⚠️ This transaction is likely FRAUDULENT (Probability: {probability:.2%})")
        else:
            st.success(f"✅ This transaction appears to be LEGITIMATE (Probability of fraud: {probability:.2%})")
        
        # Show feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': input_data.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
        ax.set_title("Feature Importance in Fraud Detection")
        st.pyplot(fig)
        
        # Show model details
        st.subheader("Model Information")
        st.write("Model Type: XGBoost Classifier")
        st.write("Features Used:", list(input_data.columns))
        
        # Example of a suspicious transaction
        st.subheader("Example of a Suspicious Transaction")
        st.write("""
        A transaction with these characteristics would be considered suspicious:
        - High amount (e.g., ₹50,000+)
        - First-time transaction (Transaction Frequency = 1)
        - High deviation from normal amounts
        - Unusual transaction type
        - Uncommon payment gateway
        - New device
        """) 