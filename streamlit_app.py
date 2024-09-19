import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# Load the trained RandomForest model
with open("rfr_newmodel.pkl", "rb") as file:
    model = pickle.load(file)

# Define months of interest
months = ['September', 'October', 'November', 'December', 'January', 'February', 'March']

# Define marketing channels (renaming "Undirected" for display)
channels = ['PaidSocial', 'PaidSearch', 'DirectMail', 'Radio', 'OutOfHome', 'Print']

# Pre-populated data for marketing spend
prepopulated_data = {
    'September': [3850, 6882, 0, 3938, 0, 0],
    'October': [3850, 6882, 22500, 3938, 0, 0],
    'November': [3850, 6882, 22500, 3938, 0, 0],
    'December': [3850, 2000, 0, 3938, 0, 0],
    'January': [3850, 2000, 0, 3938, 0, 0],
    'February': [3850, 2000, 0, 0, 0, 0],
    'March': [0, 0, 0, 0, 0, 0]
}

# Function to make predictions based on sequential input
def make_predictions(df):
    df=df
    predictions = []
    
    for month in df.index:
        i = df.index.get_loc(month)  # Get the numerical index of the current month
        if i == 0:
            # First row is already fully filled, just predict
            prediction = model.predict(df.loc[[month]])[0]
            predictions.append(prediction)
        else:
            # Fill in 'MonthBefore' data with the predictions from the previous month
            if predictions:  # Ensure there's a previous prediction to use
                df.loc[month, 'WebUsersMonthBefore'] = predictions[-1][0]
                df.loc[month, 'ContactsMonthBefore'] = predictions[-1][1]

            # Ensure columns are being accessed safely
            for col in ['PaidSocial', 'PaidSearch', 'DirectMail', 'Radio', 'OutOfHome', 'Print']:
                month_col = f'{col}MonthBefore'
                if month_col in df.columns:
                    previous_month = df.index[i - 1]  # Get the previous month
                    df.loc[month, month_col] = df.loc[previous_month, col]  # Fill with the previous month's spend
            
            # Make prediction for the current month
            prediction = model.predict(df.loc[[month]])[0]
            predictions.append(prediction)

    # Convert predictions into a DataFrame with columns for 'WebUsers' and 'Contacts'
    predictions_df = pd.DataFrame(predictions, columns=['WebUsers', 'Contacts'], index=df.index)
    
    return predictions_df.T.style.format("{:.0f}")

# Initialize the app
st.title('Marketing Spend Prediction')

# Pre-populated DataFrame with input data (transposed format)
input_data = pd.DataFrame(prepopulated_data, index=channels).T

# Define the structure of the DataFrame required for the model
def create_initial_dataframe(input_data):
    df = pd.DataFrame({
        'MonthsRunning': [5, 6, 7, 8, 9, 10, 11],
        'MonthNumber': [9, 10, 11, 12, 1, 2, 3],
        'PaidSocial': input_data['PaidSocial'],
        'PaidSearch': input_data['PaidSearch'],
        'DirectMail': input_data['DirectMail'],
        'Radio': input_data['Radio'],
        'OutOfHome': input_data['OutOfHome'],
        'Print': input_data['Print'],
        # Fill with the first row's full data; others will be filled sequentially
        'WebUsersMonthBefore': [2589] + [np.nan] * 6,
        'ContactsMonthBefore': [22] + [np.nan] * 6,
        'PaidSocialMonthBefore': [5008.15] + [np.nan] * 6,
        'PaidSearchMonthBefore': [6017.84] + [np.nan] * 6,
        'DirectMailMonthBefore': [0] + [np.nan] * 6,
        'RadioMonthBefore': [3938.75] + [np.nan] * 6,
        'OutOfHomeMonthBefore': [0] + [np.nan] * 6,
        'PrintMonthBefore': [0] + [np.nan] * 6
    })
    return df


# Display the editable input DataFrame (transposed)
st.subheader("Input Marketing Spend for Each Channel (Rows) and Month (Columns)")
input_df = st.data_editor(input_data.T, use_container_width=True)
# Create initial dataframe with missing values
initial_df = create_initial_dataframe(input_df.T)
# Generate predictions when inputs are changed
if st.button('Generate Predictions'):
    # Call the prediction function
    predictions = make_predictions(initial_df)
    
    # Display the output as a non-editable DataFrame
    st.subheader("Predicted WebUsers and Contacts for Each Month")
    st.dataframe(predictions, use_container_width=True)

# Feature importance visualization (if your RandomForest model supports it)
if hasattr(model, 'feature_importances_'):
    rf_importances = model.feature_importances_

    # Get feature names
    feature_names = ['MonthsRunning', 'MonthNumber', 'PaidSocial', 'PaidSearch', 'DirectMail',
                     'Radio', 'OutOfHome', 'Print', 'WebUsersMonthBefore', 'ContactsMonthBefore',
                     'PaidSocialMonthBefore', 'PaidSearchMonthBefore', 'DirectMailMonthBefore',
                     'RadioMonthBefore', 'OutOfHomeMonthBefore', 'PrintMonthBefore']

    # Convert to DataFrame for easier manipulation
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_importances
    })

    # Sort feature importances by descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Add explanation about feature importances
    with st.expander("See explanation"):
        st.write("""
            **Feature Importance Explanation:**

            Feature importance reflects how much each feature contributed to the model's predictions. 
            Features with higher importance values have more influence on the predictions made by the model.
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
