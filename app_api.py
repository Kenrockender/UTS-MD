import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
clf_path = os.path.join(BASE_DIR, 'models', 'clf_model.pkl')
reg_path = os.path.join(BASE_DIR, 'models', 'reg_model.pkl')

with open(clf_path, 'rb') as f:
    clf_model = pickle.load(f)
with open(reg_path, 'rb') as f:
    reg_model = pickle.load(f)


class StudentInput(BaseModel):
    gender: str
    branch: str
    cgpa: float
    tenth_percentage: float
    twelfth_percentage: float
    backlogs: int
    study_hours_per_day: float
    attendance_percentage: float
    projects_completed: int
    internships_completed: int
    coding_skill_rating: int
    communication_skill_rating: int
    aptitude_skill_rating: int
    hackathons_participated: int
    certifications_count: int
    sleep_hours: float
    stress_level: int
    part_time_job: str
    family_income_level: str
    city_tier: str
    internet_access: str
    extracurricular_involvement: str


@app.get("/")
def root():
    return {"message": "Placement Prediction API is running", "docs": "/docs"}


@app.post("/predict/classification")
def predict_classification(data: StudentInput):
    df = pd.DataFrame([data.dict()])
    prediction = clf_model.predict(df)[0]
    probability = clf_model.predict_proba(df)[0].tolist()
    label = "Placed" if prediction == 1 else "Not Placed"
    return {
        "placement_status": label,
        "placed_probability": round(probability[1], 4),
        "not_placed_probability": round(probability[0], 4)
    }


@app.post("/predict/regression")
def predict_regression(data: StudentInput):
    df = pd.DataFrame([data.dict()])
    salary = reg_model.predict(df)[0]
    return {"salary_lpa": round(float(salary), 2)}
