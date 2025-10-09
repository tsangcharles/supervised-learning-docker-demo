from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API for predicting Titanic passenger survival",
    version="1.0.0"
)

# Global variables to hold model and encoder
model = None
label_encoder = None

class PassengerData(BaseModel):
    """Input data schema for prediction"""
    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    sex: str = Field(..., description="Sex (male or female)")
    age: float = Field(..., ge=0, le=100, description="Age in years")
    sibsp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, description="Passenger fare")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pclass": 3,
                "sex": "male",
                "age": 22.0,
                "sibsp": 1,
                "parch": 0,
                "fare": 7.25
            }
        }

class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    survived: int = Field(..., description="0 = Did not survive, 1 = Survived")
    probability: float = Field(..., description="Probability of survival")
    prediction_text: str = Field(..., description="Human-readable prediction")

@app.on_event("startup")
async def load_model():
    """Load the trained model and label encoder on startup"""
    global model, label_encoder
    
    model_path = '/app/models/titanic_model.pkl'
    encoder_path = '/app/models/label_encoder.pkl'
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}. Please train the model first.")
    
    if not os.path.exists(encoder_path):
        raise RuntimeError(f"Label encoder not found at {encoder_path}. Please train the model first.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    print("Model and label encoder loaded successfully!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Titanic Survival Prediction API",
        "endpoints": {
            "POST /predict": "Make a prediction",
            "GET /health": "Check API health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(passenger: PassengerData):
    """Predict survival for a passenger"""
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Encode sex
        sex_encoded = label_encoder.transform([passenger.sex.lower()])[0]
        
        # Prepare features in the correct order
        features = np.array([[
            passenger.pclass,
            sex_encoded,
            passenger.age,
            passenger.sibsp,
            passenger.parch,
            passenger.fare
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]  # Probability of survival
        
        # Create human-readable response
        prediction_text = "Survived" if prediction == 1 else "Did not survive"
        
        return PredictionResponse(
            survived=int(prediction),
            probability=float(probability),
            prediction_text=prediction_text
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(passengers: list[PassengerData]):
    """Predict survival for multiple passengers"""
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    for passenger in passengers:
        try:
            sex_encoded = label_encoder.transform([passenger.sex.lower()])[0]
            features = np.array([[
                passenger.pclass,
                sex_encoded,
                passenger.age,
                passenger.sibsp,
                passenger.parch,
                passenger.fare
            ]])
            
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            prediction_text = "Survived" if prediction == 1 else "Did not survive"
            
            predictions.append({
                "survived": int(prediction),
                "probability": float(probability),
                "prediction_text": prediction_text
            })
        except Exception as e:
            predictions.append({
                "error": str(e)
            })
    
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

