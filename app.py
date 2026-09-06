from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import time


# -----------------------------------
# Load trained model
# -----------------------------------

model = joblib.load("churn_model.pkl")


# -----------------------------------
# Create FastAPI applications
# -----------------------------------

app = FastAPI()


# -----------------------------------
# Input schema
# -----------------------------------

class CustomerData(BaseModel):
    age: float
    monthly_usage: float
    support_calls: float


# -----------------------------------
# Prediction endpoint
# -----------------------------------

# @app.post("/predict")
# def predict(data: CustomerData):

#     X = [[
#         data.age,
#         data.monthly_usage,
#         data.support_calls
#     ]]

#     prediction = model.predict(X)[0]

#     probability = model.predict_proba(X)[0][1]

#     return {
#         "prediction": int(prediction),
#         "churn_probability": float(probability)
#     }

@app.post("/predict")
def predict(data: CustomerData):

    start_time = time.time()

    X = [[
        data.age,
        data.monthly_usage,
        data.support_calls
    ]]

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0][1]

    response_time = time.time() - start_time

    print("----- Prediction Monitoring -----")
    print("Age:", data.age)
    print("Usage:", data.monthly_usage)
    print("Support calls:", data.support_calls)
    print("Prediction:", prediction)
    print("Probability:", probability)
    print("Response time:", response_time)

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
        "response_time": response_time
    }