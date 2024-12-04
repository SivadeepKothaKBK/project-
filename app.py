import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set Streamlit page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for Stylish UI
st.markdown("""
    <style>
        /* Background styling */
        .stApp {
            background-color: #f4f4f4;
            color: #333333;
        }
        /* Stylish header */
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
        }
        .title span:nth-child(1) {
            color: #4285F4;
        }
        .title span:nth-child(2) {
            color: #EA4335;
        }
        .title span:nth-child(3) {
            color: #FBBC05;
        }
        .title span:nth-child(4) {
            color: #34A853;
        }
        .title span:nth-child(5) {
            color: #4285F4;
        }
        .title span:nth-child(6) {
            color: #EA4335;
        }
        .title span:nth-child(7) {
            color: #FBBC05;
        }
        .title span:nth-child(8) {
            color: #34A853;
        }
        .title span:nth-child(9) {
            color: #4285F4;
        }
        .title span:nth-child(10) {
            color: #EA4335;
        }
        .title span:nth-child(11) {
            color: #FBBC05;
        }
        .title span:nth-child(12) {
            color: #34A853;
        }
        .title span:nth-child(13) {
            color: #4285F4;
        }
        .title span:nth-child(14) {
            color: #EA4335;
        }
        .title span:nth-child(15) {
            color: #FBBC05;
        }
        .title span:nth-child(16) {
            color: #34A853;
        }
        .title span:nth-child(17) {
            color: #4285F4;
        }
        .title span:nth-child(18) {
            color: #EA4335;
        }
        .title span:nth-child(19) {
            color: #FBBC05;
        }
        .title span:nth-child(20) {
            color: #34A853;
        }
        .subheader {
            font-size: 18px;
            text-align: center;
            color: #333333;
            margin-top: -20px;
            margin-bottom: 40px;
        }
        /* Custom button styling */
        div.stButton > button {
            background-color: #2E8BC0;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px 20px;
        }
        div.stButton > button:hover {
            background-color: #145DA0;
            color: #f4f4f4;
        }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown(
    """
    <div class="title">
        <span>F</span><span>r</span><span>a</span><span>u</span><span>d</span> <span>D</span><span>e</span><span>t</span><span>e</span><span>c</span><span>t</span><span>i</span><span>o</span><span>n</span> <span>S</span><span>y</span><span>s</span><span>t</span><span>e</span><span>m</span>
    </div>
    <div class="subheader">Predict fraud or no fraud for any transaction</div>
    """, unsafe_allow_html=True
)

# Create Two Pages Using Buttons
st.markdown("## Choose a Page")
page = st.radio(
    label="Navigate to a Page",
    options=["Fraud Prediction on a Transaction", "How it Works"],
    index=0,
    key="page_selector",
)

if page == "Fraud Prediction on a Transaction":
    # Fraud Prediction Section
    st.write("### Enter transaction details to predict whether it is fraud or not.")
    
    # Input fields for features
    distance_from_home = st.number_input("Distance from Home (e.g., 5.0)", value=0.0)
    distance_from_last_transaction = st.number_input("Distance from Last Transaction (e.g., 2.0)", value=0.0)
    ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price (e.g., 1.2)", value=0.0)
    repeat_retailer = st.selectbox("Repeat Retailer (1 for Yes, 0 for No)", [0, 1])
    used_chip = st.selectbox("Used Chip (1 for Yes, 0 for No)", [0, 1])
    used_pin_number = st.selectbox("Used PIN Number (1 for Yes, 0 for No)", [0, 1])
    online_order = st.selectbox("Online Order (1 for Yes, 0 for No)", [0, 1])
    
    # Button to predict
    if st.button("Predict"):
        # Create an array for the inputs
        input_features = np.array([[distance_from_home, distance_from_last_transaction,
                                    ratio_to_median_purchase_price, repeat_retailer,
                                    used_chip, used_pin_number, online_order]])
        
        # Scale the input features
        input_features_scaled = scaler.transform(input_features)
    
        # Make the prediction
        prediction = model.predict(input_features_scaled)
    
        # Display the result
        if prediction[0] == 1:
            st.error("The transaction is likely FRAUD!")
        else:
            st.success("The transaction is NOT fraud.")

elif page == "How it Works":
    # How it Works Section
    st.write("### How the Fraud Detection Model Works")
    st.markdown("""
        - **Purpose**: This system is designed to predict whether a transaction is fraudulent or not.
        - **Features Used**:
          - Distance from Home: Measures how far the transaction occurred from the user's home.
          - Distance from Last Transaction: Measures how far the transaction occurred from the last recorded transaction.
          - Ratio to Median Purchase Price: Compares the transaction amount to typical spending patterns.
          - Repeat Retailer: Checks if the transaction is from a frequently visited retailer.
          - Used Chip: Indicates if the card's chip was used for the transaction.
          - Used PIN Number: Indicates if the transaction was authorized using a PIN.
          - Online Order: Indicates if the transaction was made online.
        - **Process**:
          1. The model uses these features as inputs.
          2. Preprocessed data is passed into a trained machine learning model.
          3. The model outputs whether the transaction is likely fraud or not.
        - **Best Model**: The system dynamically uses the best-performing model based on evaluation metrics.
    """)


