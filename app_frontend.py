import requests
import pandas as pd
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Placement API Client", layout="wide")
st.title("Placement Prediction - API Client")

with st.sidebar:
    st.header("Student Profile")

    gender    = st.selectbox("Gender",  ["Male", "Female"])
    branch    = st.selectbox("Branch",  ["CSE", "ECE", "EEE", "ME", "CE", "IT"])
    city      = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    internet  = st.selectbox("Internet Access", ["Yes", "No"])
    family    = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
    part_time = st.selectbox("Part Time Job", ["No", "Yes"])
    extra     = st.selectbox("Extracurricular Involvement", ["Low", "Medium", "High"])

    cgpa     = st.slider("CGPA", 4.0, 10.0, 7.5, 0.1)
    tenth    = st.slider("10th Percentage", 40.0, 100.0, 75.0)
    twelfth  = st.slider("12th Percentage", 40.0, 100.0, 75.0)
    backlogs = st.number_input("Backlogs", 0, 20, 0)
    attend   = st.slider("Attendance %", 40.0, 100.0, 80.0)

    coding      = st.slider("Coding Skill (1-10)", 1, 10, 5)
    comm        = st.slider("Communication (1-10)", 1, 10, 5)
    apt         = st.slider("Aptitude (1-10)", 1, 10, 5)

    projects    = st.number_input("Projects",       0, 20, 3)
    internships = st.number_input("Internships",    0, 10, 1)
    hackathons  = st.number_input("Hackathons",     0, 20, 2)
    certs       = st.number_input("Certifications", 0, 20, 2)

    study  = st.slider("Study Hours/Day",    0.0, 12.0, 4.0, 0.5)
    sleep  = st.slider("Sleep Hours",        3.0, 10.0, 7.0, 0.5)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

payload = {
    "gender": gender, "branch": branch, "cgpa": cgpa,
    "tenth_percentage": tenth, "twelfth_percentage": twelfth,
    "backlogs": backlogs, "study_hours_per_day": study,
    "attendance_percentage": attend, "projects_completed": projects,
    "internships_completed": internships, "coding_skill_rating": coding,
    "communication_skill_rating": comm, "aptitude_skill_rating": apt,
    "hackathons_participated": hackathons, "certifications_count": certs,
    "sleep_hours": sleep, "stress_level": stress, "part_time_job": part_time,
    "family_income_level": family, "city_tier": city,
    "internet_access": internet, "extracurricular_involvement": extra
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Classification")
    if st.button("Predict Placement Status"):
        res = requests.post(f"{API_URL}/predict/classification", json=payload)
        result = res.json()
        label = result["placement_status"]
        st.success(f"Result: **{label}**")
        st.metric("Placed Probability",     f"{result['placed_probability']*100:.1f}%")
        st.metric("Not Placed Probability", f"{result['not_placed_probability']*100:.1f}%")

with col2:
    st.subheader("Regression")
    if st.button("Predict Salary (LPA)"):
        res = requests.post(f"{API_URL}/predict/regression", json=payload)
        result = res.json()
        st.success(f"Estimated Salary: **Rs {result['salary_lpa']} LPA**")

st.divider()
st.subheader("Payload Sent to API")
st.json(payload)
