# import streamlit as st
# import requests

# # Streamlit UI Setup with improved layout
# st.set_page_config(page_title="Gold Price Prediction", layout="centered")

# st.markdown(
#     """
#     <h1 style='text-align: center; color: gold;'>Gold Price Prediction</h1>
#     <p style='text-align: center;'>Enter the required values to predict the gold price.</p>
#     """,
#     unsafe_allow_html=True,
# )

# # Creating a form for better UI structure
# with st.form("prediction_form"):
#     st.subheader("Input Features")
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         open_price = st.number_input("Open Price", min_value=0.0, format="%.2f")
#         volume = st.number_input("Volume", min_value=0.0, format="%.2f")
#     with col2:
#         high = st.number_input("High Price", min_value=0.0, format="%.2f")
#         high_low = st.number_input("High-Low", min_value=0.0, format="%.2f")
#     with col3:
#         low = st.number_input("Low Price", min_value=0.0, format="%.2f")
#         open_close = st.number_input("Open-Close", min_value=0.0, format="%.2f")
    
#     # Submit button
#     submitted = st.form_submit_button("Predict")

# if submitted:
#     input_data = {
#         "open": open_price,
#         "high": high,
#         "low": low,
#         "volume": volume,
#         "highLow": high_low,
#         "openClose": open_close
#     }
#     try:
#         response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
#         if response.status_code == 200:
#             prediction = response.json().get("predicted_price", "Error")
#             st.markdown(
#                 f"""
#                 <div style='text-align: center; padding: 20px; background-color: #f0f0f5; border-radius: 10px;'>
#                     <h2 style='color: #006400;'>Predicted Gold Price: Rs. {prediction:.2f}</h2>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
#         else:
#             st.error("Failed to get prediction. Please try again.")
#     except requests.exceptions.RequestException:
#         st.error("Error connecting to the backend API.")

import streamlit as st
import requests

# 🚀 Replace this with your actual FastAPI URL on Render
API_URL = "https://your-fastapi-app.onrender.com"

# Streamlit UI Setup with improved layout
st.set_page_config(page_title="Gold Price Prediction", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: gold;'>Gold Price Prediction</h1>
    <p style='text-align: center;'>Enter the required values to predict the gold price.</p>
    """,
    unsafe_allow_html=True,
)

# API Health Check Button
if st.button("Test API Connection"):
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            st.success("✅ API is reachable!")
        else:
            st.error(f"⚠️ API returned an error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"🚨 Could not connect to API: {e}")

# Creating a form for better UI structure
with st.form("prediction_form"):
    st.subheader("Input Features")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        open_price = st.number_input("Open Price", min_value=0.0, format="%.2f")
        volume = st.number_input("Volume", min_value=0.0, format="%.2f")
    with col2:
        high = st.number_input("High Price", min_value=0.0, format="%.2f")
        high_low = st.number_input("High-Low", min_value=0.0, format="%.2f")
    with col3:
        low = st.number_input("Low Price", min_value=0.0, format="%.2f")
        open_close = st.number_input("Open-Close", min_value=0.0, format="%.2f")
    
    # Submit button
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        "open": open_price,
        "high": high,
        "low": low,
        "volume": volume,
        "highLow": high_low,
        "openClose": open_close
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=input_data)
        if response.status_code == 200:
            prediction = response.json().get("predicted_price", "Error")
            st.markdown(
                f"""
                <div style='text-align: center; padding: 20px; background-color: #f0f0f5; border-radius: 10px;'>
                    <h2 style='color: #006400;'>Predicted Gold Price: Rs. {prediction:.2f}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error(f"⚠️ Prediction failed: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"🚨 API Connection Error: {e}")
