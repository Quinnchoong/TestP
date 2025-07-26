# student_employability_predict.py
import pandas as pd
import joblib

# Load the model and scaler
try:
    model = joblib.load("employability_predictor.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("❌ Error: Model or scaler file not found.")
    exit()

# Define feature columns
feature_columns = [
    'GENDER', 'GENERAL_APPEARANCE', 'GENERAL_POINT_AVERAGE',
    'MANNER_OF_SPEAKING', 'PHYSICAL_CONDITION', 'MENTAL_ALERTNESS',
    'SELF-CONFIDENCE', 'ABILITY_TO_PRESENT_IDEAS', 'COMMUNICATION_SKILLS',
    'STUDENT_PERFORMANCE_RATING', 'NO_SKILLS', 'Year_of_Graduate'
]

# Collect user input
print("🎓 Student Employability Predictor (Console Version)\n")

inputs = {}
inputs['GENDER'] = int(input("Gender (0=Female, 1=Male): "))
inputs['GENERAL_APPEARANCE'] = int(input("Appearance (1-5): "))
inputs['GENERAL_POINT_AVERAGE'] = float(input("GPA (0.0-4.0): "))
inputs['MANNER_OF_SPEAKING'] = int(input("Speaking (1-5): "))
inputs['PHYSICAL_CONDITION'] = int(input("Physical (1-5): "))
inputs['MENTAL_ALERTNESS'] = int(input("Alertness (1-5): "))
inputs['SELF-CONFIDENCE'] = int(input("Confidence (1-5): "))
inputs['ABILITY_TO_PRESENT_IDEAS'] = int(input("Ideas (1-5): "))
inputs['COMMUNICATION_SKILLS'] = int(input("Communication (1-5): "))
inputs['STUDENT_PERFORMANCE_RATING'] = int(input("Performance (1-5): "))
inputs['NO_SKILLS'] = int(input("Has No Skills? (0=No, 1=Yes): "))
inputs['Year_of_Graduate'] = int(input("Graduation Year (e.g., 2022): "))

# Prepare dataframe
input_df = pd.DataFrame([inputs])[feature_columns]

# Scale and predict
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]
proba = model.predict_proba(scaled_input)[0]

# Show results
if prediction == 1:
    print("\n✅ Prediction: The student is **Employable**!")
else:
    print("\n⚠️ Prediction: The student is **Less Employable**.")

print(f"Probability of Employable: {proba[1]*100:.2f}%")
print(f"Probability of Less Employable: {proba[0]*100:.2f}%")
