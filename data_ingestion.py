"""
Data Ingestion Module
Reads raw CSV files, merges, validates, and saves to ingested/ folder.
"""

from pathlib import Path
import pandas as pd

# Base directory
BASE_DIR = Path(__file__).parent

# Define folders
RAW_DIR = BASE_DIR
INGESTED_DIR = BASE_DIR / "ingested"

# Define files
FEATURES_FILE = RAW_DIR / "A.csv"
TARGETS_FILE = RAW_DIR / "A_targets.csv"
OUTPUT_FILE = INGESTED_DIR / "student_data.csv"

# Column definitions
CAT_COLS = [
    'gender', 'branch', 'part_time_job', 'family_income_level',
    'city_tier', 'internet_access', 'extracurricular_involvement'
]
NUM_COLS = [
    'cgpa', 'tenth_percentage', 'twelfth_percentage', 'backlogs',
    'study_hours_per_day', 'attendance_percentage', 'projects_completed',
    'internships_completed', 'coding_skill_rating', 'communication_skill_rating',
    'aptitude_skill_rating', 'hackathons_participated', 'certifications_count',
    'sleep_hours', 'stress_level'
]


def ingest_data():
    # Ensure output folder exists
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    # Read raw data
    features = pd.read_csv(FEATURES_FILE)
    targets = pd.read_csv(TARGETS_FILE)

    # Merge features and targets on Student_ID
    df = features.merge(targets, on='Student_ID')
    df.drop(columns=['Student_ID'], inplace=True)

    # Handle missing values
    df['extracurricular_involvement'] = df['extracurricular_involvement'].fillna('Medium')

    # Basic validation
    assert not df.empty, "Dataset is empty"

    # Save ingested data
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Data ingested from {FEATURES_FILE} & {TARGETS_FILE} → {OUTPUT_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    ingest_data()
