import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load model
model = pickle.load(open('gbin_model.pkl', 'rb'))

# Title
st.title('Insurance Price Prediction App')

# User Inputs
age = st.number_input('Age', min_value=1, max_value=100, value=25)
gender = st.selectbox('Gender', ('male', 'female'))
bmi = st.number_input('BMI', min_value=10.0, max_value=80.0, value=30.0)
smoker = st.selectbox('Smoker', ('yes', 'no'))
children = st.number_input('Children', min_value=0, max_value=10, value=2)
region = st.selectbox('Region', ('southwest', 'southeast', 'northwest', 'northeast'))

# Encoding
Smoker = 1 if smoker == 'yes' else 0
sex_male = 1 if gender == 'male' else 0
sex_female = 1 if gender == 'female' else 0

region_dict = {
    'southwest': 0,
    'northwest': 1,
    'northeast': 2,
    'southeast': 3
}
Region = region_dict[region]

# Create DataFrame
input_features = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'Smoker': [Smoker],
    'sex_female': [sex_female],
    'sex_male': [sex_male],
    'Region': [Region]
})

# Scaling (same logic, just cleaner)
scaler = StandardScaler()
input_features[['age', 'bmi']] = scaler.fit_transform(input_features[['age', 'bmi']])

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_features)[0]
    output = round(np.exp(prediction), 2)
    st.success(f'Price Prediction: ${output}')