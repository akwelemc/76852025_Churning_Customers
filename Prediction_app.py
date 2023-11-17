# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import pickle

# Load the saved model and scaler
model = load_model('assignment3_save.h5')

with open('standard_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoder.pkl', 'rb') as scaler_file:
    encoder = pickle.load(scaler_file)


st.title("Customer Churn Prediction App")

st.sidebar.header("User Input Features")

    # Create input fields for the relevant features
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
tenure = st.sidebar.slider("Tenure", min_value=0, max_value=100, value=50)
multiple_lines = st.sidebar.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
online_security = st.sidebar.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
device_protection = st.sidebar.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
tech_support = st.sidebar.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
contract = st.sidebar.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ['No', 'Yes'])
payment_method = st.sidebar.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.sidebar.slider("Monthly Charges", min_value=0, max_value=150, value=75)
total_charges = st.sidebar.slider("Total Charges", min_value=0, max_value=5000, value=2500)

if st.sidebar.button("Submit"): 
    # user_input = pd.DataFrame({
    #     'gender': gender,
    #     'tenure': tenure,
    #     'MultipleLines': multiple_lines,
    #     'OnlineSecurity': online_security,
    #     'DeviceProtection': device_protection,
    #     'TechSupport': tech_support,
    #     'Contract': contract,
    #     'PaperlessBilling': paperless_billing,
    #     'PaymentMethod': payment_method,
    #     'MonthlyCharges': monthly_charges,
    #     'TotalCharges': total_charges   
    # })

    user_input = [gender, tenure, multiple_lines, online_security,
                  device_protection, tech_support, contract, paperless_billing,
                  payment_method, monthly_charges, total_charges]


    for i in range(len(user_input)):
        if (i != len(user_input) - 1 and i != len(user_input) - 2 and i != 1):
            user_input[i] = encoder.fit_transform([user_input[i]])

    scaled_data = scaler.fit_transform([user_input])

    #processed_input_df = pd.DataFrame(scaled_data, index=[0], columns=['gender', 'tenure', 'MultipleLines', 'OnlineSecurity', 'DeviceProtection', 'TechSupport', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])

    prediction = model.predict(scaled_data)


    # Display the prediction
    st.subheader("Prediction")
    if prediction > 0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is likely to stay.")

