import streamlit as st
import pickle
import numpy as np

# Load the model and encoders
with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

model = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# Adding background image using HTML and CSS
video_html = """
<style>
.stApp {
    background: url("https://media.istockphoto.com/id/858560952/photo/dollar-banknotes-background.jpg?s=612x612&w=0&k=20&c=ghEdwyNnfmWHIDoCnw74HgFxBJ908XLKnfwrCT90Tlg=") no-repeat center center fixed;
    background-size: cover;
}
.title-box {
    background-color: rgba(0, 0, 0, 0.6); /* semi-transparent black background */
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    color: white;
}
.title-box h1 {
    color: white; /* Ensure title text is white */
    margin: 0;
}
.result-box {
    background-color: white; /* white background for the result box */
    padding: 20px;
    border-radius: 10px;
    color: black; /* text color in the result box */
    text-align: center;
    margin-top:10px;
}
.custom-label {
    color: white; /* White text color for custom labels */
    font-weight: bold;
    margin-bottom: 2px; /* Reduced margin between label and input */
    display: block;
}
.custom-text {
    color: white; /* White text color for general text */
    margin-bottom: 10px;
}
</style>
"""
st.markdown(video_html, unsafe_allow_html=True)

st.markdown('<div class="title-box"><h1>Salary Prediction App For IT Employees</h1></div>', unsafe_allow_html=True)

# Custom text and labels with white color
st.markdown('<span class="custom-text">Provide your information to predict your salary:</span>', unsafe_allow_html=True)

# Custom labels and inputs
st.markdown('<span class="custom-label">Country:</span>', unsafe_allow_html=True)
country = st.selectbox("", le_country.classes_)

st.markdown('<span class="custom-label">Education Level:</span>', unsafe_allow_html=True)
education = st.selectbox("", le_education.classes_)

st.markdown('<span class="custom-label">Years of Experience:</span>', unsafe_allow_html=True)
experience = st.slider("", 0, 50, 1)

if st.button("Predict"):
    country_encoded = le_country.transform([country])
    education_encoded = le_education.transform([education])
    features = np.array([[country_encoded[0], education_encoded[0], experience]])
    
    prediction = model.predict(features)
    output = round(prediction[0], 2)
    
    st.markdown(f'<div class="result-box">Estimated Salary: <strong>${output}</strong></div>', unsafe_allow_html=True)

st.markdown('<span class="custom-text">Adjust the input parameters to get a new prediction.</span>', unsafe_allow_html=True)
