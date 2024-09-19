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

# Updated predefined values for "under-the-hood" columns
web_users_may = 345
web_users_june = 1116
web_users_july = 1061
contact_may = 34
contact_june = 110
contact_july = 133

# The model needs columns in the following order: 
# ['MonthsRunning', 'MonthNumber', 'WebUsersMay', 'WebUsersJune', 'WebUsersJuly', 'ContactMay', 
#  'ContactJune', 'ContactJuly', 'PaidSocial', 'PaidSearch', 'DirectMail', 'Undirected']

# Pre-populated data (in the transposed format)
prepopulated_data = {
    'September': [3850.33, 6882.34, 0, 3938.75],
    'October': [3850.33, 6882.34, 22500, 3938.75],
    'November': [3850.33, 6882.34, 22500, 3938.75],
    'December': [3850.33, 2000, 0, 3938.75],
    'January': [3850.33, 2000, 0, 3938.75],
    'February': [3850.33, 2000, 0, 0],
    'March': [0, 0, 0, 0]
}

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

# Function to add totals row dynamically
def add_totals_row(df):
    # Calculate the sum for each month (column-wise sum)
    totals = df.sum(axis=0)
    totals.name = 'Total'  # Set name of totals row
    
    # Append the totals row to the input DataFrame
    df_with_totals = df.append(totals)
    
    return df_with_totals

# Initialize the app
st.title('Marketing Spend Prediction')

# Pre-populated DataFrame with input data (transposed format)
input_data = pd.DataFrame(prepopulated_data, index=channels)

# Display the editable input DataFrame with a dynamically updated totals row
st.subheader("Input Marketing Spend for Each Channel (Rows) and Month (Columns)")

# Allow the user to edit the DataFrame
edited_df = st.data_editor(input_data, use_container_width=True)

# Add the totals row dynamically based on the user's edits
edited_df_with_totals = add_totals_row(edited_df)

# Display the table with the totals row
st.dataframe(edited_df_with_totals, use_container_width=True)

# Generate predictions when inputs are changed
if st.button('Generate Predictions'):
    # Remove the totals row before sending to the prediction function
    input_df = edited_df_with_totals.drop('Total', axis=0)
    
    # Transpose the input to align with model's expected input format
    input_df = input_df.T
    
    # Call the prediction function
    predictions = make_predictions(input_df)
    
    # Display the output as a non-editable DataFrame
    st.subheader("Predicted TotalContact and WebUsers (Rows) for Each Month (Columns)")
    st.dataframe(predictions, use_container_width=True)
