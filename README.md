ANN-Based Fraud Detection System
Overview

This project implements an end-to-end fraud detection system using an Artificial Neural Network (ANN). The model is trained on transaction-level data to identify fraudulent financial transactions. The system handles extreme class imbalance using class weights and applies consistent preprocessing during training and prediction to ensure reliable real-world performance.

The project is designed to be production-ready and can be integrated into a web application or API for real-time fraud detection.

Problem Statement

Financial fraud detection is a critical challenge due to highly imbalanced datasets where fraudulent transactions represent a very small percentage of total transactions. Traditional accuracy-based models fail in such scenarios. This project focuses on detecting fraud effectively by prioritizing recall and using probability-based decision thresholds.

Dataset Description

The dataset contains transactional, temporal, and balance-based features:

step: Time step of the transaction (1 step = 1 hour)

type: Type of transaction (PAYMENT, TRANSFER, CASH_OUT, DEBIT)

amount: Transaction amount

oldbalanceOrg: Sender balance before the transaction

newbalanceOrig: Sender balance after the transaction

oldbalanceDest: Receiver balance before the transaction

newbalanceDest: Receiver balance after the transaction

isFraud: Target variable indicating fraud (1) or non-fraud (0)

Identifier columns such as sender and receiver IDs are excluded from modeling as they do not provide predictive value.

Key Features

Artificial Neural Network (ANN) for binary classification

Class imbalance handling using class weights

Label encoding for categorical features

Feature scaling using StandardScaler

Probability-based fraud prediction

Threshold tuning for improved recall

Production-ready prediction pipeline

Machine Learning Pipeline
1. Data Preprocessing

Removed identifier columns

Label encoded the transaction type feature

Scaled numerical features using StandardScaler

Ensured consistent preprocessing for training and prediction

2. Class Imbalance Handling

Used class weights during training to penalize fraud misclassification more heavily

Focused evaluation on recall, precision, and F1-score instead of accuracy

3. Model Architecture

Multi-layer Artificial Neural Network

ReLU activation in hidden layers

Sigmoid activation in output layer for probability estimation

Binary cross-entropy loss function

4. Model Evaluation

Evaluated on unseen test data

Used confusion matrix, recall, precision, and F1-score

Avoided accuracy as the primary metric due to class imbalance

Prediction Logic

The model outputs a fraud probability between 0 and 1. A configurable threshold is applied to convert this probability into a final fraud or non-fraud decision.

Prediction flow:

Raw transaction input

Label encoding of transaction type

Feature scaling using saved scaler

ANN probability prediction

Threshold-based classification

Saved Artifacts

The following objects are saved for deployment and reproducibility:

model.h5 – Trained ANN model

scaler.pkl – Fitted StandardScaler for numerical features

type_encoder.pkl – Label encoder for transaction type

Decision threshold value

Saving preprocessing objects ensures consistency between training and inference.

Project Structure
├── model.h5
├── scaler.pkl
├── type_encoder.pkl
├── fraud_detection.py
├── README.md
└── requirements.txt

Technologies Used

Python

TensorFlow / Keras

Scikit-learn

NumPy

Pandas

Joblib

How to Run the Project

Clone the repository

Install dependencies using requirements.txt

Load the trained model, scaler, and encoder

Pass raw transaction data to the prediction function

Receive fraud probability and classification result

Results and Observations

The model successfully distinguishes fraudulent and non-fraudulent transactions

High fraud recall achieved through class weighting

Normal transactions receive very low fraud probabilities

Fraud-like transactions produce significantly higher probabilities
