import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pickle

# Streamlit configuration
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

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('districtwise_cyber_crimes.csv', encoding='ISO-8859-1').rename(str.strip, axis='columns')

def train_district_model(df):
    # Prepare data for training (aggregate data if necessary)
    X = df[['YEAR']].values
    models = {}
    columns_to_check = [
        'tampering_computer_source_documents', 'computer_related_offences', 'cyber_terrorism', 
        'pub_of_sexually_exp_act_elec_form', 'intrcptn_montr_decryp_info', 
        'unauthorized_access_to_protected_computer_system', 'abetment_to_commit_offences', 
        'attempt_to_commit_offences', 'other_sections_of_it_act', 'abetment_of_suicide_online', 
        'cyber_stalking_bullying_of_women_or_children', 'data_theft', 'fraud', 'cheating', 
        'forgery', 'defamation_or_morphing', 'fake_profile', 'counterfeiting', 
        'cyber_blackmailing_threatening', 'fake_news_on_social_media', 'other_offences', 
        'gambling_act', 'lotteries_act', 'copy_right_act', 'trade_marks_act', 'other_sll_crimes', 
        'total_cyber_crimes'
    ]

    for col in columns_to_check:
        y = df[col].values
        model = LinearRegression()
        model.fit(X, y)
        models[col] = model

    # Save all models in a single pickle file
    with open('districtmodel.pkl', 'wb') as f:
        pickle.dump(models, f)
    st.success("District model trained and saved successfully.")

def load_district_model():
    with open('districtmodel.pkl', 'rb') as f:
        models = pickle.load(f)
    return models

def predict_district(models):
    st.header('District Prediction')

    df = load_data()

    if not df.empty:
        year_options = df['YEAR'].unique()
        selected_year = st.selectbox('Select Year', year_options)

        state_options = df['state_name'].unique()
        selected_state = st.selectbox('Select State', state_options)

        district_options = df[df['state_name'] == selected_state]['district_name'].unique()
        selected_district = st.selectbox('Select District', district_options)

        filtered_df = df[(df['YEAR'] == selected_year) & (df['state_name'] == selected_state) & (df['district_name'] == selected_district)]

        st.header('Filtered Cyber Crime Data')
        st.dataframe(filtered_df)

        columns_to_check = [
            'tampering_computer_source_documents', 'computer_related_offences', 'cyber_terrorism', 
            'pub_of_sexually_exp_act_elec_form', 'intrcptn_montr_decryp_info', 
            'unauthorized_access_to_protected_computer_system', 'abetment_to_commit_offences', 
            'attempt_to_commit_offences', 'other_sections_of_it_act', 'abetment_of_suicide_online', 
            'cyber_stalking_bullying_of_women_or_children', 'data_theft', 'fraud', 'cheating', 
            'forgery', 'defamation_or_morphing', 'fake_profile', 'counterfeiting', 
            'cyber_blackmailing_threatening', 'fake_news_on_social_media', 'other_offences', 
            'gambling_act', 'lotteries_act', 'copy_right_act', 'trade_marks_act', 'other_sll_crimes', 
            'total_cyber_crimes'
        ]

        st.header('Cyber Crimes in Selected District')
        for col in columns_to_check:
            crime_value = filtered_df.iloc[0][col]
            if crime_value > 0:
                st.write(f"{col.replace('_', ' ').title()}: {crime_value}")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='YEAR', y=col, data=df[df['district_name'] == selected_district], ax=ax)
                plt.xlabel('Year')
                plt.ylabel('Number of Crimes')
                plt.title(f'{col.replace("_", " ").title()} Over the Years in {selected_district}')
                st.pyplot(fig)

        st.header('Prediction for Cyber Crimes in 2024 (as percentages)')
        crime_predictions = {}
        total_crimes_2024 = 0

        for col in columns_to_check:
            model = models[col]
            predicted_crime_2024 = model.predict([[2024]])
            crime_predictions[col] = predicted_crime_2024[0]
            total_crimes_2024 += predicted_crime_2024[0]

        for col, prediction in crime_predictions.items():
            if col != 'total_cyber_crimes':
                percentage = (prediction / total_crimes_2024) * 100
                st.write(f"Predicted percentage of {col.replace('_', ' ').title()} in 2024: {percentage:.2f}%")
    else:
        st.write("No data available.")

# Load data
df = load_data()

# Train the model if not already trained
train_district_model(df)

# Load the trained model
models = load_district_model()

# Call predict_district within your main app logic where appropriate
# For example:
predict_district(models)
