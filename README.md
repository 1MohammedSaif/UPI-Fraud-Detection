# UPI Fraud Detection System

A machine learning-based system for detecting fraudulent UPI transactions using Streamlit and XGBoost.

## Overview

This project implements a fraud detection system for UPI (Unified Payments Interface) transactions. It uses machine learning to analyze transaction patterns and identify potentially fraudulent activities.

## Features

- Real-time fraud prediction for UPI transactions
- Interactive web interface using Streamlit
- Feature importance visualization
- Probability-based fraud scoring
- Support for various transaction types and payment gateways

## Prerequisites

- Python 3.7+
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/1MohammedSaif/UPI-Fraud-Detection.git
cd UPI-Fraud-Detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Data and Model Files

The following files are required to run the application:
- `upi_fraud_model.pkl` - Pre-trained XGBoost model
- `scaler.pkl` - Feature scaler
- `Copy of Sample_DATA.csv` - Sample transaction data

These files are included in the repository.

## Usage

1. Start the Streamlit application:
```bash
streamlit run upi_fraud_app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter transaction details in the web interface:
   - Transaction Amount
   - Transaction Frequency
   - Transaction Amount Deviation
   - Transaction Type
   - Payment Gateway
   - Device OS

4. Click "Predict Fraud" to get the fraud prediction and probability score

## Project Structure

- `upi_fraud_app.py` - Main Streamlit application
- `UPI_FRAUD_DETECTION_PROJECT.ipynb` - Jupyter notebook with data analysis and model development
- `UPI_Fraud_Detection_Report.md` - Detailed project report
- `requirements.txt` - Python package dependencies

## Model Details

The system uses an XGBoost classifier trained on historical transaction data. Key features include:
- Transaction amount
- Transaction frequency
- Amount deviation from normal patterns
- Transaction type
- Payment gateway
- Device information

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

Mohammed Saif Shaikh
Email: ms.sk.3609@gmail.com