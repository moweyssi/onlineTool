import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go

# Load the trained XGBoost model
model = xgb.Booster({'nthread': 4})  # init model
model.load_model('R2_0.41_MAE_646.json')  # load the pre-trained model

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
    total_contact_preds = [np.round(pred[0], 0) for pred in predictions]  # First value of the tuple is TotalContact
    web_user_preds = [np.round(pred[1], 0) for pred in predictions]       # Second value of the tuple is WebUsers
    
    # Create a DataFrame for predictions with months as columns
    predictions_df = pd.DataFrame({
        month: [total_contact_preds[i], web_user_preds[i]] 
        for i, month in enumerate(months)
    }, index=['TotalContact', 'WebUsers'])
    
    return predictions_df

# Initialize the app
st.title('Marketing Spend Prediction')

# Pre-populated DataFrame with input data (transposed format)
input_data = pd.DataFrame(prepopulated_data, index=channels)

# Display the editable input DataFrame
st.subheader("Input Marketing Spend for Each Channel (Rows) and Month (Columns)")
input_df = st.data_editor(input_data, use_container_width=True)

# Generate predictions when inputs are changed
if st.button('Generate Predictions'):
    # Transpose the input to align with model's expected input format
    input_df = input_df.T
    
    # Call the prediction function
    predictions = make_predictions(input_df)
    
    # Display the output as a non-editable DataFrame
    st.subheader("Predicted TotalContact and WebUsers (Rows) for Each Month (Columns)")
    st.dataframe(predictions, use_container_width=True)




# Assuming the feature importances are calculated like this
xg_importances = model.get_score(importance_type='weight')  # Get feature importances from XGBoost

# Get feature names
feature_names = ['MonthsRunning', 'MonthNumber', 'WebUsersMay', 'WebUsersJune', 'WebUsersJuly', 
                 'ContactMay', 'ContactJune', 'ContactJuly', 'PaidSocial', 'PaidSearch', 
                 'DirectMail', 'Undirected(radio_outofhome_print)']

# Convert to DataFrame for easier manipulation
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': [xg_importances.get(f, 0) for f in feature_names]  # Using 0 if feature is not in the importance dict
})

# Sort feature importances by descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Add explanation about feature importances
with st.expander("See explanation"):
    st.write("""
        **Feature Importance Explanation:**

        Feature importance reflects how much each feature contributed to the model's predictions. 
        Features with higher importance values have more influence on the predictions made by the model.

        The bar chart below shows the relative importance of each feature in the XGBoost model. 
        These values can help identify the key factors driving the model's behavior.
    """)

    # Create a Plotly bar chart for feature importances
    fig = go.Figure(data=[go.Bar(
        x=feature_importance_df['Feature'],
        y=feature_importance_df['Importance'],
        marker_color='indianred'
    )])

    fig.update_layout(
        title="Feature Importances",
        xaxis_title="Features",
        yaxis_title="Importance",
        xaxis_tickangle=-45,
        template="plotly_white",
        height=600
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)