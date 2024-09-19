import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

model = xgb.Booster({'nthread': 4})  # init model
model.load_model('R2_0.32_MAE_800.json')  # load data



# Define months of interest
months = ['September', 'October', 'November', 'December', 'January', 'February', 'March']

# Define marketing channels
channels = ['PaidSocial', 'PaidSearch', 'DirectMail', 'Undirected']
#MonthsRunning	MonthNumber	WebUsersMay	WebUsersJune	WebUsersJuly	ContactMay	ContactJune	ContactJuly	PaidSocial	PaidSearch	DirectMail	Undirected(radio_outofhome_print)

# Function to make predictions based on the input DataFrame
def make_predictions(input_df):
    # Ensure input data is in the correct shape
    input_features = input_df.T.values  # Shape (months, channels), needs to be (channels, months)
    
    # Predict using both models for each month (each row in input_features)
    total_contact_preds = model.predict(input_features)  # One prediction per month
    web_user_preds = model.predict(input_features)  # One prediction per month
    
    # Create a DataFrame for predictions with months as columns
    predictions = pd.DataFrame({
        month: [np.round(total_contact_preds[i], 2), np.round(web_user_preds[i], 2)] 
        for i, month in enumerate(months)
    }, index=['TotalContact', 'WebUsers'])
    
    return predictions

# Initialize the app
st.title('Marketing Spend Prediction')

# Define the structure of the input editable DataFrame
input_data = pd.DataFrame({
    month: [0] * len(channels) for month in months  # Initial input as zeroes for all months
}, index=channels)  # Channels as row labels

# Display the editable input DataFrame
st.subheader("Input Marketing Spend for Each Channel (Rows) and Month (Columns)")
input_df_viz = st.data_editor(input_data, use_container_width=True)

# Generate predictions when inputs are changed
if st.button('Generate Predictions'):
    input_data_calc = input_df_viz
    input_data_calc.append(['MonthsRunning',5,6,7,8,9,10,11])
    st.dataframe(input_data_calc)
    # Call the prediction function
    predictions = make_predictions(input_data_calc)
    
    # Display the output as a non-editable DataFrame
    st.subheader("Predicted TotalContact and WebUsers (Rows) for Each Month (Columns)")
    st.dataframe(predictions, use_container_width=True)
