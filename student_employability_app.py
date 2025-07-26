# student_employability_app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
try:
    model = joblib.load("employability_predictor.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("❌ Error: Model or scaler file not found. Please train the model first.")
    st.stop()

st.set_page_config(page_title="🎓 Student Employability Predictor", layout="centered")

st.title("🎓 Student Employability Predictor")
st.markdown("Fill out the form below to predict whether a student is employable.")

# Input form
with st.form("prediction_form"):
    gender = st.radio("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    general_appearance = st.slider("General Appearance (1-5)", 1, 5, 3)
    gpa = st.number_input("GPA (0.0 - 4.0)", min_value=0.0, max_value=4.0, step=0.01)
    speaking = st.slider("Manner of Speaking (1-5)", 1, 5, 3)
    physical = st.slider("Physical Condition (1-5)", 1, 5, 3)
    alertness = st.slider("Mental Alertness (1-5)", 1, 5, 3)
    confidence = st.slider("Self-Confidence (1-5)", 1, 5, 3)
    ideas = st.slider("Ability to Present Ideas (1-5)", 1, 5, 3)
    communication = st.slider("Communication Skills (1-5)", 1, 5, 3)
    performance = st.slider("Student Performance Rating (1-5)", 1, 5, 3)
    no_skills = st.radio("Has No Skills?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    year_grad = st.number_input("Year of Graduation", min_value=2000, max_value=2030, value=2025)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([{
        'GENDER': gender,
        'GENERAL_APPEARANCE': general_appearance,
        'GENERAL_POINT_AVERAGE': gpa,
        'MANNER_OF_SPEAKING': speaking,
        'PHYSICAL_CONDITION': physical,
        'MENTAL_ALERTNESS': alertness,
        'SELF-CONFIDENCE': confidence,
        'ABILITY_TO_PRESENT_IDEAS': ideas,
        'COMMUNICATION_SKILLS': communication,
        'STUDENT_PERFORMANCE_RATING': performance,
        'NO_SKILLS': no_skills,
        'Year_of_Graduate': year_grad
    }])

    # Scale input and predict
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0]

    if prediction == 1:
        st.success("✅ The student is predicted to be **Employable**!")
    else:
        st.warning("⚠️ The student is predicted to be **Less Employable**.")

    st.markdown(f"**Probability of Employable:** {proba[1]*100:.2f}%")
    st.markdown(f"**Probability of Less Employable:** {proba[0]*100:.2f}%")
