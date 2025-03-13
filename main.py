# from fastapi import FastAPI
# import pickle
# import numpy as np
# from pydantic import BaseModel

# # Load the trained model
# model_path = "best_model.pkl"
# with open(model_path, "rb") as f:
#     model = pickle.load(f)

# # Initialize FastAPI app
# app = FastAPI()

# # Define input data structure
# class GoldPriceInput(BaseModel):
#     open: float
#     high: float
#     low: float
#     volume: float
#     highLow: float
#     openClose: float

# @app.get("/")
# def home():
#     return {"message": "Gold Price Prediction API"}

# @app.post("/predict")
# def predict_price(data: GoldPriceInput):
#     # Convert input data to numpy array
#     features = np.array([[
#         data.open, data.high, data.low, data.volume, data.highLow, data.openClose
#     ]])
    
#     # Make prediction
#     predicted_price = model.predict(features)[0]
    
#     return {"predicted_price": float(predicted_price)}

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for Streamlit connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your Streamlit app URL for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model_path = "best_model.pkl"

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevents crashes if model file is missing

# Define input data structure
class GoldPriceInput(BaseModel):
    open: float
    high: float
    low: float
    volume: float
    highLow: float
    openClose: float

@app.get("/")
def home():
    return {"message": "Gold Price Prediction API is running!"}

@app.post("/predict")
def predict_price(data: GoldPriceInput):
    # Check if the model is loaded correctly
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found or failed to load")

    # Convert input data to numpy array
    features = np.array([[
        data.open, data.high, data.low, data.volume, data.highLow, data.openClose
    ]])
    
    # Make prediction
    predicted_price = model.predict(features)[0]
    
    return {
        "input_data": data.dict(),
        "predicted_price": round(float(predicted_price), 2)  # Round for better readability
    }
