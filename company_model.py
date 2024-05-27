import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import joblib
import pickle
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #10EDFB, #F206E5);
    color: white;
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
""", unsafe_allow_html=True)

st.title("Cyber Crime in Companies Analysis")

def load_company_model():
    with open('companymodel.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict_company(model):
    st.header('Company Prediction')
    try:
        df = pd.read_csv('company_cyber_crime_data.csv', encoding='ISO-8859-1').rename(str.strip, axis='columns')
        df['Records'] = df['Records'].str.replace(' ', '').str.replace(',', '').astype(float)
    except (UnicodeDecodeError, FileNotFoundError, ValueError) as e:
        st.error(f"Error: {e}")
        df = pd.DataFrame()

    if not df.empty:
        selected_company = st.selectbox('Select Company', df['Company'].unique(), key='company_dropdown')
        filtered_df = df[df['Company'] == selected_company]

        if not filtered_df.empty:
            st.header('Filtered Cyber Crime Data')
            st.markdown(filtered_df.to_html(classes='dataframe-table'), unsafe_allow_html=True)

            plt.figure(figsize=(10, 6))
            sns.set_style("whitegrid", {'axes.grid': False})
            ax = sns.barplot(x='Year', y='Records', hue='Method', data=filtered_df, palette='viridis')

            ax.set_facecolor("none")
            ax.spines['bottom'].set_linewidth(0)
            ax.spines['left'].set_linewidth(0)
            ax.spines['top'].set_color(None)
            ax.spines['right'].set_color(None)
            ax.figure.patch.set_alpha(0)
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(2)
                spine.set_visible(True)

            label_color = '#0d47a1'
            plt.title(f'Year-wise Cyber Crime Records for {selected_company}', color=label_color)
            plt.xlabel('Year', color=label_color)
            plt.ylabel('Number of Records', color=label_color)

            ax.tick_params(colors=label_color, which='both')

            legend = plt.legend(title='Method')
            plt.setp(legend.get_texts(), color=label_color)
            plt.setp(legend.get_title(), color=label_color)

            st.pyplot(plt)

            if 'Year' in filtered_df.columns and 'Records' in filtered_df.columns:
                X = filtered_df[['Year']]
                y = filtered_df['Records']

                model = LinearRegression()
                model.fit(X, y)

                prediction_2024 = model.predict([[2024]])[0]

                st.subheader('Prediction')
                st.write(f"Predicted number of records for {selected_company} in 2024: {prediction_2024:.2f}")


                # Assuming 'model' is your trained machine learning model
                with open('companymodel.pkl', 'wb') as f:
                    pickle.dump(model, f)
                st.write("Model saved successfully.")
            else:
                st.write("Data is not suitable for machine learning.")
        else:
            st.write("No data available for the selected company.")
    else:
        st.write("No data available.")


