import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    Drug = pd.read_csv("Drug.csv")
    Drug = Drug.drop_duplicates()
    Drug['Reviews'] = Drug['Reviews'].str.replace(' Reviews', '', regex=False)
    Drug['Reviews'] = Drug['Reviews'].astype(int)
    return Drug

# Load or create preprocessors
@st.cache_resource
def load_preprocessors(Drug):
    try:
        label_encoders = joblib.load('label_encoders.joblib')
        scaler = joblib.load('scaler.joblib')
    except FileNotFoundError:
        label_encoders = {}
        categorical_features = ['Condition', 'Drug', 'Indication']
        for feature in categorical_features:
            label_en = LabelEncoder()
            Drug[feature] = label_en.fit_transform(Drug[feature])
            label_encoders[feature] = label_en
        
        scaler = StandardScaler()
        Drug[['Reviews']] = scaler.fit_transform(Drug[['Reviews']])
        
        joblib.dump(label_encoders, 'label_encoders.joblib')
        joblib.dump(scaler, 'scaler.joblib')
    
    return label_encoders, scaler

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf_model_effective.joblib')
        return model
    except Exception as e:
        st.warning(f"Failed to load model: {str(e)}")
        return None

# Preprocess input
def preprocess_input(drug_name, condition_name, Drug, label_encoders, scaler):
    try:
        # Create a DataFrame with all features, initialized to 0
        input_df = pd.DataFrame(0, index=[0], columns=Drug.columns)
        
        # Set the known values
        input_df['Drug'] = label_encoders['Drug'].transform([drug_name])[0]
        input_df['Condition'] = label_encoders['Condition'].transform([condition_name])[0]
        input_df['Reviews'] = scaler.transform([[0]])[0][0]
        
        # Ensure all columns from the training data are present
        for col in Drug.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match the order used during training
        input_df = input_df[Drug.columns]
        
        return input_df.values.reshape(1, -1)
    except KeyError as e:
        st.error(f"Missing key: {str(e)}. Please check your input data.")
        return None
    except ValueError:
        st.error("Drug or condition not found in the database. Please check your input.")
        return None

# Predict function
def predict(model, input_array):
    try:
        return model.predict(input_array)[0]
    except ValueError as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Streamlit app
st.title('Drug Effectiveness Prediction')

# Load data and model
Drug = load_data()
label_encoders, scaler = load_preprocessors(Drug)
model = load_model()

# User input
drug_name = st.text_input('Drug Name')
condition_name = st.text_input('Condition Name')

if st.button('Predict'):
    if not drug_name or not condition_name:
        st.warning("Please enter both drug name and condition.")
    else:
        # Check if drug and condition exist in the dataset
        if drug_name not in Drug['Drug'].values or condition_name not in Drug['Condition'].values:
            st.error("Drug or condition not found in the database. Please check your input.")
        else:
            input_array = preprocess_input(drug_name, condition_name, Drug, label_encoders, scaler)
            if input_array is not None:
                st.write(f"Input array shape: {input_array.shape}")  # Debug information
                prediction = predict(model, input_array)
                
                # Display result
                if prediction is not None:
                    st.subheader('Prediction Result')
                    st.write(f'Random Forest (Effective): {prediction:.2f}')

                    # Visualization
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(['Random Forest (Effective)'], [prediction], color='blue')

                    ax.set_ylabel('Score')
                    ax.set_title('Effectiveness Prediction')

                    st.pyplot(fig)

                    st.subheader('Interpretation')
                    st.write("Effectiveness scores typically range from 0 to 10.")
                    st.write("Higher scores indicate better effectiveness.")