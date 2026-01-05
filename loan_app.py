import streamlit as st
import pickle
import numpy as np

st.title("🏦 Loan Approval Prediction with Multiple Models")

models = {
    "Logistic Regression": pickle.load(open("loan_model1.pkl", "rb")),
    "KNN": pickle.load(open("loan_model2.pkl", "rb")),
    "Decision Tree": pickle.load(open("loan_model3.pkl", "rb")),
    "Random Forest": pickle.load(open("loan_model4.pkl", "rb"))
}

selected_model_name = st.selectbox("Select Model", list(models.keys()))
model = models[selected_model_name]

experience = st.number_input("Experience (Years)", min_value=0, max_value=50, value=1)
income = st.number_input("Annual Income ($)", min_value=0, max_value=1000000, value=50000)
cc_avg = st.number_input("Average Credit Card Spending (CCAvg)", min_value=0.0, max_value=50000.0, value=2000.0)
education = st.selectbox("Education", ["Undergrad", "Graduate", "Advanced/Professional"])
mortgage = st.number_input("Mortgage Value ($)", min_value=0, max_value=1000000, value=0)
cd_account = st.selectbox("CD Account", ["Yes", "No"])

input_data = np.array([[ 
    experience,
    income,
    cc_avg,
    1 if education == "Graduate" else (2 if education == "Advanced/Professional" else 0),
    mortgage,
    1 if cd_account == "Yes" else 0
]])

if st.button("Check Loan Status"):
    prediction = model.predict(input_data)[0]
    st.write(f"Model used: **{selected_model_name}**")
    if prediction == 1:
        st.success("✅ Congratulations! Your loan is APPROVED.")
    else:
        st.error("❌ Sorry, your loan application is REJECTED.")