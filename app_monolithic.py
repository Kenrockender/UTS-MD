"""
Task 3 -- Monolithic Streamlit App
Run: streamlit run app_monolithic.py
"""

import pickle
import pandas as pd
import streamlit as st

# -- Page config (MUST be first Streamlit call) --------------------------------
st.set_page_config(page_title="Student Placement Predictor", layout="wide")

# -- Load models ---------------------------------------------------------------
@st.cache_resource
def load_models():
    with open('models/clf_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('models/reg_model.pkl', 'rb') as f:
        reg = pickle.load(f)
    return clf, reg

clf_model, reg_model = load_models()

st.title("Student Placement & Salary Predictor")
st.caption("Model Deployment Exam -- DTSC6012001 | Dataset A")

# -- Sidebar inputs ------------------------------------------------------------
with st.sidebar:
    st.header("Student Profile")

    st.subheader("Personal Info")
    gender  = st.selectbox("Gender",  ["Male", "Female"])
    branch  = st.selectbox("Branch",  ["CSE", "ECE", "EEE", "ME", "CE", "IT"])
    city    = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    internet = st.selectbox("Internet Access", ["Yes", "No"])
    family  = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
    part_time = st.selectbox("Part Time Job", ["No", "Yes"])
    extra   = st.selectbox("Extracurricular Involvement", ["Low", "Medium", "High"])

    st.subheader("Academic Performance")
    cgpa    = st.slider("CGPA", 4.0, 10.0, 7.5, 0.1)
    tenth   = st.slider("10th Percentage", 40.0, 100.0, 75.0, 0.5)
    twelfth = st.slider("12th Percentage", 40.0, 100.0, 75.0, 0.5)
    backlogs = st.number_input("Backlogs", 0, 20, 0)
    attend  = st.slider("Attendance %", 40.0, 100.0, 80.0, 0.5)

    st.subheader("Skills & Activities")
    coding  = st.slider("Coding Skill (1-10)",        1, 10, 5)
    comm    = st.slider("Communication Skill (1-10)", 1, 10, 5)
    apt     = st.slider("Aptitude Skill (1-10)",      1, 10, 5)
    projects    = st.number_input("Projects Completed",   0, 20, 3)
    internships = st.number_input("Internships Completed", 0, 10, 1)
    hackathons  = st.number_input("Hackathons Participated", 0, 20, 2)
    certs       = st.number_input("Certifications Count",    0, 20, 2)

    st.subheader("Lifestyle")
    study   = st.slider("Study Hours / Day", 0.0, 12.0, 4.0, 0.5)
    sleep   = st.slider("Sleep Hours",       3.0, 10.0, 7.0, 0.5)
    stress  = st.slider("Stress Level (1-10)", 1, 10, 5)

# -- Build input DataFrame -----------------------------------------------------
input_data = pd.DataFrame([{
    'gender': gender, 'branch': branch, 'cgpa': cgpa,
    'tenth_percentage': tenth, 'twelfth_percentage': twelfth,
    'backlogs': backlogs, 'study_hours_per_day': study,
    'attendance_percentage': attend, 'projects_completed': projects,
    'internships_completed': internships, 'coding_skill_rating': coding,
    'communication_skill_rating': comm, 'aptitude_skill_rating': apt,
    'hackathons_participated': hackathons, 'certifications_count': certs,
    'sleep_hours': sleep, 'stress_level': stress, 'part_time_job': part_time,
    'family_income_level': family, 'city_tier': city,
    'internet_access': internet, 'extracurricular_involvement': extra
}])

# -- Predictions ---------------------------------------------------------------
placement_raw = clf_model.predict(input_data)[0]
placement_prob = clf_model.predict_proba(input_data)[0]
placement_label = "Placed" if placement_raw == 1 else "Not Placed"
confidence = max(placement_prob) * 100

salary = reg_model.predict(input_data)[0]

# -- Display results -----------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Placement Prediction", placement_label)
col2.metric("Confidence", f"{confidence:.1f}%")
col3.metric("Estimated Salary", f"Rs {salary:.2f} LPA")

st.divider()

# -- Data Visualizations -------------------------------------------------------
viz1, viz2 = st.columns(2)

with viz1:
    st.subheader("Placement Probability")
    prob_df = pd.DataFrame({
        'Status': ['Not Placed', 'Placed'],
        'Probability': [placement_prob[0] * 100, placement_prob[1] * 100]
    })
    st.bar_chart(prob_df.set_index('Status'), color='#4CAF50')

with viz2:
    st.subheader("Skill Overview")
    skill_df = pd.DataFrame({
        'Skill': ['Coding', 'Communication', 'Aptitude', 'CGPA (scaled)',
                  'Attendance (scaled)', 'Study Hours (scaled)'],
        'Score': [coding, comm, apt, cgpa, attend / 10, study * 10 / 12]
    })
    st.bar_chart(skill_df.set_index('Skill'), color='#2196F3')

st.divider()
st.subheader("Input Summary")
st.dataframe(input_data, use_container_width=True)
