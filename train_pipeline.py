# Nama: [Isi Nama Anda]
# NIM: [Isi NIM Ganjil Anda]

import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error, r2_score

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

def load_dataset(features_path='A.csv', targets_path='A_targets.csv'):
    """Fungsi modular untuk memuat dataset"""
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    df = features.merge(targets, on='Student_ID')
    df.drop(columns=['Student_ID'], inplace=True)
    return df

def main():
    # 1. Data Ingestion
    print("Membaca dataset...")
    df = load_dataset()

    # 2. Preprocessing & Split
    le = LabelEncoder()
    df['placement_status'] = le.fit_transform(df['placement_status'])
    
    X = df.drop(columns=['placement_status', 'salary_lpa'])
    y_clf = df['placement_status']
    y_reg = df['salary_lpa']
    
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    def build_preprocessor():
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        return ColumnTransformer(transformers=[
            ('num', num_pipeline, NUM_COLS),
            ('cat', cat_pipeline, CAT_COLS)
        ])
    
    # 3. Model Training & MLflow Tracking
    mlflow.set_experiment("uts_md_deployment")
    os.makedirs('models', exist_ok=True)
    
    # Model Klasifikasi
    print("Training Classification model...")
    with mlflow.start_run(run_name="classification-lr"):
        clf_pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', LogisticRegression(max_iter=500, random_state=42))
        ])
        clf_pipeline.fit(X_train, yc_train)
        
        preds_clf = clf_pipeline.predict(X_test)
        f1 = f1_score(yc_test, preds_clf, average='weighted')
        
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("f1_weighted", f1)
        mlflow.sklearn.log_model(clf_pipeline, "clf_model")
        
        # Simpan persistence .pkl
        with open('models/clf_model.pkl', 'wb') as f:
            pickle.dump(clf_pipeline, f)
        print(f"Classification F1 Score: {f1:.4f}")

    # Model Regresi
    print("Training Regression model...")
    with mlflow.start_run(run_name="regression-lr"):
        reg_pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('regressor', LinearRegression())
        ])
        reg_pipeline.fit(X_train, yr_train)
        
        preds_reg = reg_pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(yr_test, preds_reg))
        r2 = r2_score(yr_test, preds_reg)
        
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(reg_pipeline, "reg_model")
        
        # Simpan persistence .pkl
        with open('models/reg_model.pkl', 'wb') as f:
            pickle.dump(reg_pipeline, f)
        print(f"Regression RMSE: {rmse:.4f} | R2: {r2:.4f}")
        
    print("Pipeline selesai. Model berhasil disimpan di folder 'models'.")

if __name__ == "__main__":
    main()
