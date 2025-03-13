from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

# Load the trained model
model_path = "best_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

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
    return {"message": "Gold Price Prediction API"}

@app.post("/predict")
def predict_price(data: GoldPriceInput):
    # Convert input data to numpy array
    features = np.array([[
        data.open, data.high, data.low, data.volume, data.highLow, data.openClose
    ]])
    
    # Make prediction
    predicted_price = model.predict(features)[0]
    
    return {"predicted_price": float(predicted_price)}

