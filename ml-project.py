import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and encoder
model = joblib.load('rf_model.pkl')
le = joblib.load('label_encoder.pkl')

# Load feature names (your reduced 20 features)
feature_names = ['skin_rash', 'chills', 'joint_pain', 'stomach_pain', 'vomiting','fatigue', 'weight_loss', 'high_fever', 'sweating', 'headache','nausea', 'loss_of_appetite', 'abdominal_pain', 'diarrhoea','yellowing_of_eyes', 'malaise', 'chest_pain', 'dizziness','irritability', 'muscle_pain']

st.title("Disease Prediction App")
st.markdown("Select symptoms below and get a disease prediction.")

# Create a checkbox for each feature
input_features = []
for feat in feature_names:
    val = st.checkbox(feat.replace("_", " ").title())
    input_features.append(1 if val else 0)

if st.button("Predict"):
    prediction = model.predict([input_features])[0]
    disease = le.inverse_transform([prediction])[0]
    st.success(f"Predicted Disease: **{disease}**")
