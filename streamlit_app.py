import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression  # Dummy model for example

# Load or define the regression model
# For demo purposes, using simple linear regression models
# In reality, you would load a pre-trained model, e.g., with joblib or pickle.
model_total_contact = LinearRegression()
model_web_users = LinearRegression()

# Dummy training data (this would be replaced by your real model and dataset)
X_train = np.random.rand(10, 4)  # Random marketing spends as features
y_train_contact = np.random.rand(10) * 1000  # Random targets for TotalContact
y_train_web = np.random.rand(10) * 500  # Random targets for WebUsers

# Fit the models (replace with your pre-trained model)
model_total_contact.fit(X_train, y_train_contact)
model_web_users.fit(X_train, y_train_web)

# Define months of interest
months = ['September', 'October', 'November', 'December', 'January', 'February', 'March']

# Define marketing channels
channels = ['PaidSocial', 'PaidSearch', 'DirectMail', 'Undirected']

# Function to make predictions based on the input DataFrame
def make_predictions(input_df):
    # Transpose the input DataFrame to get the correct shape (features should be columns, not rows)
    input_features = input_df.T.values  # Transposing to get (7 months, 4 channels)
    
    # Predict using both models
    total_contact_preds = model_total_contact.predict(input_features.T)  # Transpose to align features
    web_user_preds = model_web_users.predict(input_features.T)  # Transpose to align features
    
    # Create a DataFrame for predictions with months as columns
    predictions = pd.DataFrame({
        'TotalContact': np.round(total_contact_preds, 2),
        'WebUsers': np.round(web_user_preds, 2)
    }, index=months).T  # Transpose to get predictions as rows and months as columns
    
    return predictions

# Initialize the app
st.title('Marketing Spend Prediction')

# Define the structure of the input editable DataFrame
input_data = pd.DataFrame({
    month: [0] * len(channels) for month in months  # Initial input as zeroes for all months
}, index=channels)  # Channels as row labels

# Display the editable input DataFrame
st.subheader("Input Marketing Spend for Each Channel (Rows) and Month (Columns)")
input_df = st.data_editor(input_data, use_container_width=True)

# Generate predictions when inputs are changed
if st.button('Generate Predictions'):
    # Call the prediction function
    predictions = make_predictions(input_df)
    
    # Display the output as a non-editable DataFrame
    st.subheader("Predicted TotalContact and WebUsers (Rows) for Each Month (Columns)")
    st.dataframe(predictions, use_container_width=True)
