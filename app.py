import streamlit as st
import pandas as pd
import pickle

# Title
st.title("❤️ Heart Disease Prediction App")

# Load trained model
try:
    with open("Heart_disease_prediction_model.pkl", "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
except:
    model = None
    st.warning("⚠️ Model file not found. Please train and save your model as Heart_disease_prediction_model.pkl")

# User input fields
age = st.number_input("Age", 29, 77, 30)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
chest_pain_type = st.selectbox("Chest Pain Type (1–4)", [ 1, 2, 3,4])
resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", 94, 180, 120)
serum_cholesterol_mg_per_dl = st.number_input("Serum Cholesterol (mg/dl)", 126, 546, 200)
fasting_blood_sugar_gt_120_mg_per_dl = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ekg_results = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
max_heart_rate_achieved = st.number_input("Max Heart Rate Achieved", 50, 250, 150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak_eq_st_depression = st.number_input("Oldpeak (ST Depression)", -2.0, 6.5, 1.0, step=0.1)
slope_of_peak_exercise_st_segment = st.selectbox("Slope of ST Segment (1–3)", [ 1, 2,3])
num_major_vessels = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0–2)", [0, 1, 2])

# Prediction button
if st.button("🔍 Predict"):
    if model:
        input_data = pd.DataFrame([[
            thal, resting_blood_pressure, chest_pain_type, num_major_vessels,
            fasting_blood_sugar_gt_120_mg_per_dl, resting_ekg_results,
            serum_cholesterol_mg_per_dl, oldpeak_eq_st_depression,
            sex, age, max_heart_rate_achieved, exercise_induced_angina
        ]], columns=[
            "thal", "resting_blood_pressure", "chest_pain_type", "num_major_vessels",
            "fasting_blood_sugar_gt_120_mg_per_dl", "resting_ekg_results",
            "serum_cholesterol_mg_per_dl", "oldpeak_eq_st_depression",
            "sex", "age", "max_heart_rate_achieved", "exercise_induced_angina"
        ])

        prediction = model.predict(input_data)[0]

        st.subheader("🩺 Prediction Result:")
        if prediction == 1:
            st.error("⚠️ The patient is **likely** to have Heart Disease.")
        else:
            st.success("✅ The patient is **unlikely** to have Heart Disease.")
    else:
        st.error("❌ Model not available. Please check your model file.")


