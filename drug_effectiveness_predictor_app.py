import streamlit as st
import numpy as np
import joblib
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os

st.set_page_config(page_title="Drug Effectiveness Predictor", layout="wide")

# Load the XGBoost model
@st.cache_resource
def load_model():
    xgb_model_path = 'xgboost_model.joblib'
    if not os.path.exists(xgb_model_path):
        st.error(f"Model file not found: {xgb_model_path}")
        return None
    return joblib.load(xgb_model_path)

xgb_model = load_model()

# Load the dataset
@st.cache_data
def load_data():
    file_path = "Drug.csv"
    if not os.path.exists(file_path):
        st.error(f"Data file not found: {file_path}")
        return None
    return pd.read_csv(file_path)

Drug = load_data()

# Function to preprocess input
def preprocess_input(drug_name, condition):
    features = np.zeros(773)  # Initialize with zeros
    features[0] = len(drug_name)
    features[1] = len(condition)
    features[2:] = np.random.rand(771)
    return features.reshape(1, -1)

# Function to predict drug effectiveness and ease of use
def predict_drug_effectiveness_and_ease(drug_name, condition):
    input_features = preprocess_input(drug_name, condition)
    predictions = xgb_model.predict(input_features)
    effectiveness = predictions[0][0]
    ease_of_use = predictions[0][1]
    return effectiveness, ease_of_use

# Streamlit app

st.title('Drug Effectiveness Predictor')

st.markdown("""
This application predicts the effectiveness of a drug based on its name and the condition it treats.
*Please note:* These predictions are based on a simplified model and may not be fully accurate.
""")

# User input
col1, col2 = st.columns(2)
with col1:
    drug_name = st.text_input('Enter the drug name:')
with col2:
    condition_name = st.text_input('Enter the condition:')

if st.button('Predict'):
    if not drug_name or not condition_name:
        st.warning("Please enter both drug name and condition.")
    elif Drug is None or xgb_model is None:
        st.error("Cannot make predictions due to missing data or model.")
    else:
        # Check if drug and condition exist in the dataset
        if drug_name not in Drug['Drug'].values or condition_name not in Drug['Condition'].values:
            st.error("Drug or condition not found in the database. Please check your input.")
        else:
            input_array = preprocess_input(drug_name, condition_name)
            #st.write(f"Input array shape: {input_array.shape}")  # Debug information
            
            effectiveness, ease_of_use = predict_drug_effectiveness_and_ease(drug_name, condition_name)
            
            st.subheader('Prediction Results')
            
            # Create a figure with two subplots
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Effectiveness plot
            ax1.bar(['Effectiveness'], [effectiveness], color='royalblue')
            ax1.set_ylabel('Score')
            ax1.set_title('Effectiveness Prediction')
            ax1.set_ylim(0, 5)
            
            
            # Improve aesthetics
            for ax in [ax1]:
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
            
            st.write(f'*Effectiveness:* {effectiveness:.2f}')
            
            st.subheader('Interpretation')
            st.write("Effectiveness scores typically range from 0 to 5.")
            st.write("Higher scores indicate better effectiveness.")
            
            if effectiveness < 1.5:
                st.write("- The drug may have low effectiveness for the given condition.")
            elif effectiveness < 3.5:
                st.write("- The drug may have moderate effectiveness for the given condition.")
            else:
                st.write("- The drug may have high effectiveness for the given condition.")
            st.write("*Please consult with a healthcare professional for medical advice.*")

# Add some information about the limitations of the model
st.sidebar.header('Note')
st.sidebar.write("""
This is a simplified model and its predictions should be taken with caution. 
The model uses limited information and random values for many features, 
which may affect the accuracy of the predictions.
""")