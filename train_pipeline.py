import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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
    """Memuat dan menggabungkan fitur dan target berdasarkan Student_ID."""
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    df = features.merge(targets, on='Student_ID')
    df.drop(columns=['Student_ID'], inplace=True)
    return df

def build_preprocessor():
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer(transformers=[
        ('num', num_pipeline, NUM_COLS),
        ('cat', cat_pipeline, CAT_COLS)
    ])

def main():
    df = load_dataset()

    le = LabelEncoder()
    df['placement_status'] = le.fit_transform(df['placement_status'])

    X = df.drop(columns=['placement_status', 'salary_lpa'])
    y_clf = df['placement_status']
    y_reg = df['salary_lpa']

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
    )

    mlflow.set_experiment("uts_md_deployment")
    os.makedirs('models', exist_ok=True)

    clf_candidates = {
        'LogisticRegression': LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'),
        'RandomForest':       RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'DecisionTree':       DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=10),
    }

    best_clf, best_f1 = None, -1
    for name, clf in clf_candidates.items():
        with mlflow.start_run(run_name=f"clf-{name}"):
            pipeline = Pipeline([
                ('preprocessor', build_preprocessor()),
                ('classifier', clf)
            ])
            pipeline.fit(X_train, yc_train)
            f1 = f1_score(yc_test, pipeline.predict(X_test), average='weighted')

            mlflow.log_param("model", name)
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_metric("f1_weighted", f1)
            mlflow.sklearn.log_model(pipeline, f"clf_{name}")

            print(f"[CLF] {name}: F1={f1:.4f}")
            if f1 > best_f1:
                best_f1, best_clf = f1, pipeline

    with open('models/clf_model.pkl', 'wb') as f:
        pickle.dump(best_clf, f)
    print(f"Best classifier saved (F1={best_f1:.4f})")

    reg_candidates = {
        'LinearRegression': LinearRegression(),
        'RandomForest':     RandomForestRegressor(n_estimators=100, random_state=42),
        'DecisionTree':     DecisionTreeRegressor(random_state=42, max_depth=8),
    }

    best_reg, best_rmse = None, float('inf')
    for name, reg in reg_candidates.items():
        with mlflow.start_run(run_name=f"reg-{name}"):
            pipeline = Pipeline([
                ('preprocessor', build_preprocessor()),
                ('regressor', reg)
            ])
            pipeline.fit(X_train, yr_train)
            preds = pipeline.predict(X_test)
            rmse = np.sqrt(mean_squared_error(yr_test, preds))
            r2   = r2_score(yr_test, preds)

            mlflow.log_param("model", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(pipeline, f"reg_{name}")

            print(f"[REG] {name}: RMSE={rmse:.4f} R2={r2:.4f}")
            if rmse < best_rmse:
                best_rmse, best_reg = rmse, pipeline

    with open('models/reg_model.pkl', 'wb') as f:
        pickle.dump(best_reg, f)
    print(f"Best regressor saved (RMSE={best_rmse:.4f})")

if __name__ == "__main__":
    main()
