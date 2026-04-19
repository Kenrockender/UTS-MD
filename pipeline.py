"""
Task 2 — Scikit-Learn Pipeline + MLflow
Run: python pipeline.py
"""

import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from data_ingestion import ingest_data
from train import train_clf, train_reg
from evaluation import evaluate_clf, evaluate_reg

F1_THRESHOLD = 0.8


def run_pipeline():
    print("Step 1: Data Ingestion")
    ingest_data()

    # Read ingested data
    df = pd.read_csv("ingested/student_data.csv")

    # Encode classification target
    le = LabelEncoder()
    df['placement_status'] = le.fit_transform(df['placement_status'])
    with open('models/label_classes.pkl', 'wb') as f:
        pickle.dump(le.classes_, f)

    X     = df.drop(columns=['placement_status', 'salary_lpa'])
    y_clf = df['placement_status']
    y_reg = df['salary_lpa']

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
    )

    print("\nStep 2: Training Classification")
    clf_run_id = train_clf(X_train, yc_train)

    print("\nStep 3: Training Regression")
    reg_run_id = train_reg(X_train, yr_train)

    print("\nStep 4: Evaluation")
    f1 = evaluate_clf(X_test, yc_test, clf_run_id)
    rmse, r2 = evaluate_reg(X_test, yr_test, reg_run_id)

    if f1 >= F1_THRESHOLD:
        print("\n✅ Classification model approved for deployment")
    else:
        print("\n❌ Classification model rejected")

    print("\n[OK] Models saved to models/clf_model.pkl and models/reg_model.pkl")


if __name__ == '__main__':
    run_pipeline()
