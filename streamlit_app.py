import streamlit as st
from company_model import load_company_model, predict_company
from statemodel import load_state_model, predict_state
from districtmodel import load_district_model, predict_district
import time
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
    .justified-text {
        text-align: justify;
    }
    </style>
      
    """,
    unsafe_allow_html=True
)

# Function to display the welcome page
def display_welcome():
    st.title('Cyber Crime Filter Gateway')
    st.write("Developed by Meera Solgama..")
    st.write("Proudly representing L. D. College of Engineering")
    st.write("Obtained a degree from Gujarat Technological University, known for academic excellence.")
    st.write("""
### Abstract
This project introduces a Streamlit web portal for analyzing and predicting cybercrime trends in 2024. Users can filter data by company, state, and district. The portal uses machine learning to forecast future trends, aiding proactive threat management.

With a user-friendly interface, the portal is valuable for cybersecurity professionals, law enforcement, and policymakers. It combines data visualization and predictive analytics to enhance cyber resilience and mitigate digital threats.
""")

    

# Function to load and predict using the appropriate model based on the option selected
def main():
    # Sidebar with options
    option = st.sidebar.selectbox('Select Option:', ('select','State', 'District', 'Company'), index=0)

    # Display the welcome page only when no option is selected
    if option == 'select':
        display_welcome()
    else:
        # Display a loading indicator
        with st.spinner('Loading models...'):
            # Load and predict using the appropriate model based on the option selected
            if option == 'Company':
                model = load_company_model()
                st.write("Company model loaded successfully.")
                predict_company(model)
            elif option == 'State':
                model = load_state_model()
                st.write("State model loaded successfully.")
                predict_state(model)
            elif option == 'District':
                model = load_district_model()
                st.write("District model loaded successfully.")
                predict_district(model)

# Run the app
if __name__ == '__main__':
    main()
