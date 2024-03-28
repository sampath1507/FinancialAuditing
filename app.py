import streamlit as st
import pickle
import pandas as pd
import numpy as np
import google.generativeai as genai

# Create a Gemini API client
genai.configure(api_key="AIzaSyAedK1gcNbPJ83hIFTb2_silc2vcOTcJM8")
model = genai.GenerativeModel('gemini-pro')

# Load the training dataset
train_data = pd.read_csv("audit_data.csv")

# Load the testing dataset
test_data = pd.read_csv("trial.csv")

# Feature descriptions
feature_descriptions = """
PARA_A: Audit Parameter A in the report
PARA_B: Audit Parameter B in the report
TOTAL: Sum of Audit Parameters A and B
Sector_score: Score of the organization in its respective sector
numbers: Rating of the organization
Loss_score: Loss value of the organization
Money_value: Money value of the organization
History_score: Historical score of the organization
District: Type of district the organization belongs to
Risk: Whether the organization is fraudulent or not (1 for high risk, 0 for low risk)
"""

# Prompt the Gemini model with the dataset information, feature descriptions, and feature selection process
dataset_info = f"""
Training Dataset Information: {train_data.info()}
Training Dataset Head: {train_data.head().to_string()}
Testing Dataset Information: {test_data.info()}
Testing Dataset Head: {test_data.head().to_string()}
Feature Descriptions: {feature_descriptions}
Feature Selection Process: After performing hyperparameter tuning, a subset of the most important features was selected for the audit risk prediction model. The selected features are: ['Audit_Risk', 'Inherent_Risk', 'Score', 'TOTAL', 'Risk_D', 'Score_MV', 'CONTROL_RISK', 'Risk_B', 'PARA_B', 'Money_Value', 'PARA_A', 'RiSk_E', 'Score_A', 'Risk_A']. The summarized report should focus on these selected features and their impact on the predicted risk.
"""

def generate_audit_report(input_data, predicted_risk):
    # feature_values = []
    # for feature in ['Audit_Risk', 'Inherent_Risk', 'Score', 'TOTAL', 'Risk_D', 'Score_MV', 'CONTROL_RISK', 'Risk_B', 'PARA_B', 'Money_Value', 'PARA_A', 'RiSk_E', 'Score_A', 'Risk_A']:
    #     feature_values.append(f"{feature}: {input_data[feature]}")
    feature_values = input_data.iloc[0].to_numpy().reshape(1, -1)
    risk_label = "High" if predicted_risk == 1 else "Low"
    prompt = f"""{dataset_info}
You are an experienced financial auditor. Your task is to analyze the following feature values and the predicted audit risk, and provide a summarized report explaining the reasons for the predicted risk level based on the dataset information, feature descriptions, and feature selection process provided.
Feature Values: {', '.join(feature_values[0])}
Predicted Risk: {predicted_risk} (1 for high risk, 0 for low risk)
Summarized Report:
"""
    generation_config = genai.types.GenerationConfig(
        candidate_count=1,
        stop_sequences=['space'],
        max_output_tokens=400,
        temperature=0
    )
    response = model.predict(prompt, generation_config=generation_config)
    report = response.generated_text.strip()
    return report

# Load the trained model
model_path = "decision_tree_model.pickle"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define function to make predictions
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit interface with custom styling
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        color: #336699;
        text-align: center;
        margin-top: 50px;
        margin-bottom: 30px;
    }
    .sidebar .sidebar-content {
        font-size: 18px;
        background-color: #f0f0f0;
        color: #333333;
    }
    .block-container {
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        background-color: #ffffff;
    }
    .prediction {
        font-size: 24px;
        color: #228B22;
        text-align: center;
    }
    .button {
        background-color: #336699;
        color: white;
        font-size: 18px;
        border-radius: 5px;
        padding: 10px 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1 class='title'>Audit Risk Prediction</h1>", unsafe_allow_html=True)

# Define input fields
st.sidebar.header('Input Parameters')
audit_risk = st.sidebar.number_input('Audit Risk', min_value=0.0, max_value=100000.0, value=0.0, format="%.2f")
inherent_risk = st.sidebar.number_input('Inherent Risk', min_value=0.0, max_value=100000.0, value=0.0, format="%.2f")
score = st.sidebar.number_input('Score', min_value=0.0, max_value=100.0, value=0.0, format="%.2f")
total = st.sidebar.number_input('Total', min_value=0.0, max_value=100000.0, value=0.0, format="%.2f")
risk_d = st.sidebar.number_input('Risk_D', min_value=0.0, max_value=100000.0, value=0.0, format="%.2f")
control_risk = st.sidebar.number_input('Control Risk', min_value=0.0, max_value=100000.0, value=0.0, format="%.2f")
money_value = st.sidebar.number_input('Money Value', min_value=0.0, max_value=100000.0, value=0.0, format="%.2f")
para_b = st.sidebar.number_input('PARA_B', min_value=0.0, max_value=100000.0, value=0.0, format="%.2f")
risk_e = st.sidebar.number_input('RiSk_E', min_value=0.0, max_value=100000.0, value=0.0, format="%.2f")
para_a = st.sidebar.number_input('PARA_A', min_value=0.0, max_value=100000.0, value=0.0, format="%.2f")
risk_a = st.sidebar.number_input('Risk_A', min_value=0.0, max_value=100000.0, value=0.0, format="%.2f")
score_mv = st.sidebar.number_input('Score_MV', min_value=0.0, max_value=100.0, value=0.0, format="%.2f")
risk_b = st.sidebar.number_input('Risk_B', min_value=0.0, max_value=100000.0, value=0.0, format="%.2f")
score_a = st.sidebar.number_input('Score_A', min_value=0.0, max_value=100.0, value=0.0, format="%.2f")
score_b = st.sidebar.number_input('Score_B', min_value=0.0, max_value=100.0, value=0.0, format="%.2f")

# Store input data in a DataFrame
input_data = pd.DataFrame({
    'Audit_Risk': [audit_risk],
    'Inherent_Risk': [inherent_risk],
    'Score': [score],
    'TOTAL': [total],
    'Risk_D': [risk_d],
    'CONTROL_RISK': [control_risk],
    'Money_Value': [money_value],
    'PARA_B': [para_b],
    'RiSk_E': [risk_e],
    'PARA_A': [para_a],
    'Risk_A': [risk_a],
    'Score_MV': [score_mv],
    'Risk_B': [risk_b],
    'Score_A': [score_a],
    'Score_B': [score_b]
})

# Predict the risk
if st.sidebar.button('Predict'):
    prediction = predict(model, input_data)
    report = generate_audit_report(input_data.iloc[0], prediction[0])
    st.markdown(f'<p class="prediction">Predicted Risk: <b>{prediction[0]}</b></p>', unsafe_allow_html=True)
    st.markdown(f'<div class="block-container">{report}</div>', unsafe_allow_html=True)