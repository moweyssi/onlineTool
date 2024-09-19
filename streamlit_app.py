import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# Load the trained XGBoost model
model = xgb.Booster({'nthread': 4})  # init model
model.load_model('R2_0.32_MAE_800.json')  # load the pre-trained model

# Define months of interest
months = ['September', 'October', 'November', 'December', 'January', 'February', 'March']

# Define marketing channels (renaming "Undirected" for display)
channels = ['PaidSocial', 'PaidSearch', 'DirectMail', 'Undirected(radio_outofhome_print)']

# Pre-defined values for "under-the-hood" columns
web_users_may = 1000
web_users_june = 1100
web_users_july = 1050
contact_may = 50
contact_june = 60
contact_july = 55

# The model needs columns in the following order: 
# ['MonthsRunning', 'MonthNumber', 'WebUsersMay', 'WebUsersJune', 'WebUsersJuly', 'ContactMay', 
#  'ContactJune', 'ContactJuly', 'PaidSocial', 'PaidSearch', 'DirectMail', 'Undirected']

# Function to make predictions based on the input DataFrame
def make_predictions(input_df):
    # Add the predefined columns to the input dataframe
    input_df['WebUsersMay'] = web_users_may
    input_df['WebUsersJune'] = web_users_june
    input_df['WebUsersJuly'] = web_users_july
    input_df['ContactMay'] = contact_may
    input_df['ContactJune'] = contact_june
    input_df['ContactJuly'] = contact_july
    
    # Add 'MonthsRunning' and 'MonthNumber'
    input_df['MonthsRunning'] = np.arange(1, len(months) + 1)  # Example: 1, 2, 3, ...
    input_df['MonthNumber'] = np.arange(5, 5 + len(months))     # Example: 5, 6, 7, ... (starting from May)
    
    # Reorder columns to match the model's requirements
    columns_order = ['MonthsRunning', 'MonthNumber', 'WebUsersMay', 'WebUsersJune', 'WebUsersJuly', 
                     'ContactMay', 'ContactJune', 'ContactJuly', 'PaidSocial', 'PaidSearch', 
                     'DirectMail', 'Undirected(radio_outofhome_print)']
    
    
    # Reorder the dataframe columns
    input_df = input_df[columns_order]
    
    # Convert DataFrame to DMatrix for XGBoost prediction
    dmatrix = xgb.DMatrix(input_df)
    
    # Predict using the XGBoost model
    predictions = model.predict(dmatrix)
    
    # Extract predictions and organize them
    total_contact_preds = [np.round(pred[0], 2) for pred in predictions]  # First value of the tuple is TotalContact
    web_user_preds = [np.round(pred[1], 2) for pred in predictions]       # Second value of the tuple is WebUsers
    
    # Create a DataFrame for predictions with months as columns
    predictions_df = pd.DataFrame({
        month: [total_contact_preds[i], web_user_preds[i]] 
        for i, month in enumerate(months)
    }, index=['TotalContact', 'WebUsers'])
    
    return predictions_df

# Initialize the app
st.title('Marketing Spend Prediction')

# Define the structure of the input editable DataFrame
input_data = pd.DataFrame({
    month: [0] * len(channels) for month in months  # Initial input as zeroes for all months
}, index=channels)  # Channels as row labels

# Display the editable input DataFrame
st.subheader("Input Marketing Spend for Each Channel (Rows) and Month (Columns)")

input_df = st.data_editor(input_data, use_container_width=True)
total_df = pd.DataFrame({
    "Total          ":input_df.sum(numeric_only=True, axis=0)
})
st.dataframe(total_df.T, use_container_width=True)
# Generate predictions when inputs are changed
if st.button('Generate Predictions'):
    # Transpose the input to align with model's expected input format
    input_df = input_df.T
    
    # Call the prediction function
    predictions = make_predictions(input_df)
    
    # Display the output as a non-editable DataFrame
    st.subheader("Predicted TotalContact and WebUsers (Rows) for Each Month (Columns)")
    st.dataframe(predictions, use_container_width=True)
