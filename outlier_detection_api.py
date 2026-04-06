from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
import pickle

app = FastAPI()

with open("outlier_detection_model.pkl", "rb") as f:
    model = pickle.load(f)

class OutlierDetectionRequest(BaseModel):
    ride_time: float = Field(..., description="Ride time in seconds")
    trip_distance: float = Field(..., description="Trip distance in miles")

class OutlierDetectionResponse(BaseModel):
    outlier: bool = Field(..., description="True if the ride is classified as an outlier")

@app.get("/detect-outliers", response_model=OutlierDetectionResponse)
def detect_outliers(
    ride_time: float = Query(..., description="Ride time in seconds"),
    trip_distance: float = Query(..., description="Trip distance in miles")
):
    input_df = pd.DataFrame([{
        "ride_time": ride_time,
        "trip_distance": trip_distance
    }])

    try:
        preds = model.predict(input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    outlier = bool(preds)
    return OutlierDetectionResponse(outlier=outlier)