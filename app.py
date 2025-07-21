import streamlit as st
import joblib as jb
import pandas as pd

# App title and description
st.set_page_config(page_title="Cancer Prediction App", layout="centered")
st.title("🩺 Cancer Prediction App")
st.write("Enter the tumor characteristics to predict if it's **benign** or **malignant**.")

# Load trained model
model = jb.load('Cancer_rediction_model.pkl')

# Input features
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 
    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 
    'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 
    'symmetry_worst', 'fractal_dimension_worst'
]

# Get user input
user_input = {}
st.header("🔢 Input Tumor Measurements")

with st.form("input_form"):
    for feature in features:
        user_input[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.01, format="%.2f")

    submit = st.form_submit_button("🚀 Predict")

# Prepare input for prediction
input_df = pd.DataFrame([user_input])

if submit:
    # Show input summary
    st.subheader("📊 Entered Data")
    st.dataframe(input_df)

    # Make prediction
    prediction = model.predict(input_df)[0]  # 0 = Benign, 1 = Malignant (assuming)
    result = "🔴 Malignant" if prediction == 1 else "🟢 Benign"

    st.divider()
    st.subheader("🧬 Prediction Result")
    st.success(f"The tumor is predicted to be: **{result}**")
else:
    st.info("Fill in the details above and click Predict to see the result.")
