import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #10EDFB, #F206E5);
    }
    .dataframe-table {
        background: white;
        color: black;
        border-radius: 10px;
        padding: 10px;
    }
    .stMarkdown {
        margin-bottom: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def train_state_model():
    try:
        df = pd.read_csv('cyber_crime_data.csv').rename(str.strip, axis='columns')
        st.write("Data loaded successfully.")

        # Prepare data (assuming 'State/UT', '2016', '2017', and '2018' are columns in the dataset)
        X = df[['2016', '2017']].values
        y = df['2018'].values

        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Save the model
        with open('statemodel.pkl', 'wb') as f:
            pickle.dump(model, f)
        st.write("Model trained and saved successfully.")
    except Exception as e:
        st.error(f"An error occurred while training the model: {e}")

def load_state_model():
    try:
        with open('statemodel.pkl', 'rb') as f:
            model = pickle.load(f)
        st.write("Model loaded successfully.")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first.")
        return None
    except EOFError:
        st.error("Model file is empty or corrupted. Please train the model again.")
        return None

def predict_state(model):
    st.header('State Prediction')
    
    # Read data
    try:
        df = pd.read_csv('cyber_crime_data.csv').rename(str.strip, axis='columns')
    except FileNotFoundError:
        st.error("CSV file not found.")
        df = pd.DataFrame()

    if not df.empty:
        selected_state = st.selectbox('Select State/UT', df['State/UT'].unique() if 'State/UT' in df else [])

        filtered_df = df[df['State/UT'] == selected_state] if not df.empty else pd.DataFrame()

        if not filtered_df.empty:
            filtered_df = filtered_df.drop(columns=['Mid-Year Projected Population (in Lakhs) (2018)+', 'Rate of Total Cyber Crimes (2018)++'])

            st.header('Filtered Cyber Crime Data')
            st.write(filtered_df)

            X_new = filtered_df[['2016', '2017']].values

            predictions = model.predict(X_new)
            filtered_df['Predicted 2018'] = predictions

            increase_2016_to_2017 = filtered_df['2017'].iloc[0] - filtered_df['2016'].iloc[0]
            increase_2017_to_2018 = predictions[0] - filtered_df['2017'].iloc[0]
            average_increase = (increase_2016_to_2017 + increase_2017_to_2018) / 2
            predicted_crime_rate_2024 = predictions[0] + (average_increase * 6)

            st.write(f"Predicted Cyber Crime Rate for 2024: {predicted_crime_rate_2024:.2f}")

            plt.figure(figsize=(10, 6))
            plt.bar(['2016', '2017', 'Predicted 2018'], [filtered_df['2016'].iloc[0], filtered_df['2017'].iloc[0], predictions[0]], color=['blue', 'orange', 'green'])
            plt.title('Cyber Crime Rates for {}'.format(selected_state))
            plt.xlabel('Year')
            plt.ylabel('Rate of Total Cyber Crimes')
            st.pyplot(plt)
        else:
            st.write("No data available for the selected state.")
    else:
        st.write("No data available.")

# Check if the model file exists and has content before trying to load it
if not os.path.exists('statemodel.pkl') or os.path.getsize('statemodel.pkl') == 0:
    st.write("Model file not found or is empty. Training the model...")
    train_state_model()

# Load the trained model
model = load_state_model()

# Only predict if the model is successfully loaded
if model:
    predict_state(model)
