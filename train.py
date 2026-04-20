import os
import pickle
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression

from data_ingestion import CAT_COLS, NUM_COLS


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUM_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS)
    ])


def train_clf(X_train, y_train):
    mlflow.set_experiment("model-deployment-exam")

    with mlflow.start_run(run_name="classification-lr") as run:
        clf_pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier',   LogisticRegression(max_iter=500, random_state=42))
        ])
        clf_pipeline.fit(X_train, y_train)

        mlflow.log_param("model",     "LogisticRegression")
        mlflow.log_param("max_iter",  500)
        mlflow.log_param("test_size", 0.2)

        mlflow.sklearn.log_model(clf_pipeline, "clf_model")

        os.makedirs('models', exist_ok=True)
        with open('models/clf_model.pkl', 'wb') as f:
            pickle.dump(clf_pipeline, f)

        print("[Classification] Model trained and saved")
        return run.info.run_id


def train_reg(X_train, y_train):
    mlflow.set_experiment("model-deployment-exam")

    with mlflow.start_run(run_name="regression-lr") as run:
        reg_pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('regressor',    LinearRegression())
        ])
        reg_pipeline.fit(X_train, y_train)

        mlflow.log_param("model",     "LinearRegression")
        mlflow.log_param("test_size", 0.2)

        mlflow.sklearn.log_model(reg_pipeline, "reg_model")

        with open('models/reg_model.pkl', 'wb') as f:
            pickle.dump(reg_pipeline, f)

        print("[Regression] Model trained and saved")
        return run.info.run_id
