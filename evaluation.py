import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, mean_squared_error, r2_score


def evaluate_clf(X_test, y_test, run_id):
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/clf_model")

    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='weighted')

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("f1_weighted", f1)

    print(f"[Classification] F1 (weighted): {f1:.4f}")
    return f1


def evaluate_reg(X_test, y_test, run_id):
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/reg_model")

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

    print(f"[Regression] RMSE: {rmse:.4f} | R²: {r2:.4f}")
    return rmse, r2
