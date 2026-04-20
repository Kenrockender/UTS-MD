import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Placement Prediction API",
    description="Predicts student placement status and salary. DTSC6012001 — Dataset A",
    version="1.0.0"
)

# Load models
with open('models/clf_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)
with open('models/reg_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)


# Input schema
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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Male", "branch": "CSE", "cgpa": 8.5,
                    "tenth_percentage": 85.0, "twelfth_percentage": 82.0,
                    "backlogs": 0, "study_hours_per_day": 6.0,
                    "attendance_percentage": 90.0, "projects_completed": 5,
                    "internships_completed": 2, "coding_skill_rating": 4,
                    "communication_skill_rating": 4, "aptitude_skill_rating": 4,
                    "hackathons_participated": 3, "certifications_count": 4,
                    "sleep_hours": 7.0, "stress_level": 4, "part_time_job": "No",
                    "family_income_level": "Medium", "city_tier": "Tier 1",
                    "internet_access": "Yes", "extracurricular_involvement": "High"
                },
                {
                    "gender": "Female", "branch": "IT", "cgpa": 6.5,
                    "tenth_percentage": 70.0, "twelfth_percentage": 65.0,
                    "backlogs": 1, "study_hours_per_day": 3.0,
                    "attendance_percentage": 75.0, "projects_completed": 1,
                    "internships_completed": 0, "coding_skill_rating": 2,
                    "communication_skill_rating": 3, "aptitude_skill_rating": 2,
                    "hackathons_participated": 0, "certifications_count": 1,
                    "sleep_hours": 6.0, "stress_level": 7, "part_time_job": "Yes",
                    "family_income_level": "Low", "city_tier": "Tier 3",
                    "internet_access": "No", "extracurricular_involvement": "Low"
                },
                {
                    "gender": "Male", "branch": "ECE", "cgpa": 9.2,
                    "tenth_percentage": 90.0, "twelfth_percentage": 88.0,
                    "backlogs": 0, "study_hours_per_day": 8.0,
                    "attendance_percentage": 95.0, "projects_completed": 7,
                    "internships_completed": 3, "coding_skill_rating": 5,
                    "communication_skill_rating": 4, "aptitude_skill_rating": 5,
                    "hackathons_participated": 5, "certifications_count": 6,
                    "sleep_hours": 8.0, "stress_level": 3, "part_time_job": "No",
                    "family_income_level": "High", "city_tier": "Tier 2",
                    "internet_access": "Yes", "extracurricular_involvement": "High"
                }
            ]
        }
    }


# Endpoints
@app.get("/")
def root():
    return {"message": "Placement Prediction API is running", "docs": "/docs"}


@app.post("/predict/classification")
def predict_classification(data: StudentInput):
    df = pd.DataFrame([data.model_dump()])
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
    df = pd.DataFrame([data.model_dump()])
    salary = reg_model.predict(df)[0]
    return {"salary_lpa": round(float(salary), 2)}
