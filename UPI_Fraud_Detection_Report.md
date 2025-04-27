# UPI Fraud Detection System - Project Report

## Executive Summary
This report presents a comprehensive analysis of the UPI Fraud Detection System, which uses machine learning to identify potentially fraudulent UPI transactions. The system employs XGBoost algorithm to analyze transaction patterns and detect anomalies that may indicate fraudulent activities.

## 1. Project Overview

### 1.1 Problem Statement
The increasing adoption of UPI (Unified Payments Interface) in India has led to a rise in digital payment fraud. This project aims to develop an automated system to detect fraudulent transactions in real-time, thereby enhancing the security of digital payments.

### 1.2 Objectives
- Develop a machine learning model to detect fraudulent UPI transactions
- Create an interactive web application for real-time fraud detection
- Analyze transaction patterns and identify key fraud indicators
- Provide actionable insights for fraud prevention

## 2. Data Analysis

### 2.1 Dataset Description
- Total Records: 647 transactions
- Features: 20 attributes including transaction details, user information, and device data
- Time Period: Financial Year 2023-2024
- Data Type: Synthetic dataset with realistic patterns

### 2.2 Key Features
1. Transaction Details:
   - Transaction ID, Date, Time
   - Amount and Amount Deviation
   - Transaction Type and Status
   - Payment Gateway

2. User Information:
   - Customer ID
   - Transaction Frequency
   - Days Since Last Transaction

3. Device and Location:
   - Device ID and OS
   - IP Address
   - Transaction City and State

4. Merchant Information:
   - Merchant ID
   - Merchant Category
   - Transaction Channel

## 3. Methodology

### 3.1 Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Data normalization

### 3.2 Feature Engineering
- Transaction amount deviation calculation
- Time-based features
- Frequency-based features
- Location-based features

### 3.3 Model Development
- Algorithm: XGBoost Classifier
- Key Parameters:
  - Learning Rate: 0.1
  - Max Depth: 6
  - Number of Trees: 100
  - Scale Positive Weight: 5 (for handling class imbalance)

### 3.4 Model Evaluation
- Accuracy metrics
- Precision and Recall
- F1-Score
- Confusion Matrix
- Feature Importance Analysis

## 4. System Implementation

### 4.1 Web Application
- Framework: Streamlit
- Key Components:
  - Data Analysis Dashboard
  - Real-time Fraud Prediction
  - Interactive Visualizations
  - Model Performance Metrics

### 4.2 Key Features
1. Transaction Analysis:
   - Fraud distribution visualization
   - Transaction amount analysis
   - Payment gateway analysis
   - Transaction type analysis

2. Fraud Prediction:
   - Real-time transaction evaluation
   - Probability scoring
   - Feature importance visualization
   - Risk assessment

## 5. Results and Findings

### 5.1 Model Performance
- High accuracy in fraud detection
- Effective handling of imbalanced data
- Strong performance on unseen data

### 5.2 Key Insights
1. Fraud Patterns:
   - Higher fraud rates in certain transaction types
   - Correlation between amount deviation and fraud
   - Time-based patterns in fraudulent transactions

2. Risk Factors:
   - First-time transactions
   - High-value transactions
   - Unusual transaction times
   - New device usage

## 6. Future Enhancements

### 6.1 Technical Improvements
- Integration with real-time transaction systems
- Implementation of ensemble methods
- Addition of more sophisticated features
- Real-time model retraining

### 6.2 Business Applications
- Integration with banking systems
- Automated fraud alerts
- Customer risk scoring
- Merchant risk assessment

## 7. Conclusion
The UPI Fraud Detection System demonstrates the effective application of machine learning in financial security. The system provides a robust framework for detecting fraudulent transactions while maintaining a user-friendly interface for real-time analysis and prediction.

## 8. References
- XGBoost Documentation
- Streamlit Documentation
- UPI Transaction Guidelines
- Financial Fraud Detection Research Papers 