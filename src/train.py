import time

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import RANDOM_STATE, load_wine_data


EXPERIMENT_NAME = "Wine-Cultivar-Classification"
TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "WineClassifier"

CV_SPLITS = 5


def build_model(model_name, params):
    """Create a model from a model family and parameter dictionary."""
    if model_name == "RandomForest":
        return RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )

    if model_name == "GradientBoosting":
        return GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            **params,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def get_configurations():
    """Return at least three configurations for each model family."""
    return {
        "RandomForest": [
            {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
            },
            {
                "n_estimators": 200,
                "max_depth": 5,
                "min_samples_split": 2,
            },
            {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
            },
        ],
        "GradientBoosting": [
            {
                "n_estimators": 100,
                "learning_rate": 0.10,
                "max_depth": 3,
            },
            {
                "n_estimators": 150,
                "learning_rate": 0.05,
                "max_depth": 3,
            },
            {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 2,
            },
        ],
    }


def calculate_metrics(y_true, y_pred, probabilities):
    """Calculate accuracy, macro F1, and log loss."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "log_loss": log_loss(y_true, probabilities),
    }


def train_candidates():
    """Train, evaluate, log candidates, and register the best model."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_wine_data()

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "log_loss": "neg_log_loss",
    }

    configurations = get_configurations()

    for model_name, model_configs in configurations.items():
        for config_number, params in enumerate(model_configs, start=1):
            model = build_model(model_name, params)

            with mlflow.start_run(
                run_name=f"{model_name}_config_{config_number}"
            ):
                start_time = time.perf_counter()

                cv_results = cross_validate(
                    model,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1,
                )

                elapsed_time = time.perf_counter() - start_time

                train_accuracy = cv_results["train_accuracy"].mean()
                val_accuracy = cv_results["test_accuracy"].mean()

                train_macro_f1 = cv_results["train_macro_f1"].mean()
                val_macro_f1 = cv_results["test_macro_f1"].mean()

                train_log_loss = -cv_results["train_log_loss"].mean()
                val_log_loss = -cv_results["test_log_loss"].mean()

                model.fit(X_train, y_train)

                predictions = model.predict(X_train)
                probabilities = model.predict_proba(X_train)

                train_metrics = calculate_metrics(
                    y_train,
                    predictions,
                    probabilities,
                )

                test_predictions = model.predict(X_test)
                test_probabilities = model.predict_proba(X_test)

                test_metrics = calculate_metrics(
                    y_test,
                    test_predictions,
                    test_probabilities,
                )

                mlflow.log_params(
                    {
                        "model_family": model_name,
                        "config_number": config_number,
                        **params,
                        "random_state": RANDOM_STATE,
                        "cv_folds": CV_SPLITS,
                    }
                )

                mlflow.log_metrics(
                    {
                        "cv_train_accuracy": train_accuracy,
                        "cv_val_accuracy": val_accuracy,
                        "cv_train_macro_f1": train_macro_f1,
                        "cv_val_macro_f1": val_macro_f1,
                        "cv_train_log_loss": train_log_loss,
                        "cv_val_log_loss": val_log_loss,
                        "fit_time_seconds": elapsed_time,
                        "train_accuracy": train_metrics["accuracy"],
                        "train_macro_f1": train_metrics["macro_f1"],
                        "train_log_loss": train_metrics["log_loss"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_macro_f1": test_metrics["macro_f1"],
                        "test_log_loss": test_metrics["log_loss"],
                    }
                )

                mlflow.set_tags(
                    {
                        "model_family": model_name,
                        "candidate": "true",
                        "random_seed": str(RANDOM_STATE),
                    }
                )

                signature = infer_signature(
                    X_train,
                    model.predict(X_train),
                )

                input_example = X_train.head(5)

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                )

                print(
                    f"{model_name} config {config_number}: "
                    f"validation Macro F1 = {val_macro_f1:.4f}"
                )

    print("\nAll candidate models have been logged to MLflow.")

    # ------------------------------------------------------------
    # Select the best candidate using validation Macro F1
    # ------------------------------------------------------------

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.candidate = 'true'",
        order_by=[
            "metrics.cv_val_macro_f1 DESC",
            "attributes.start_time ASC",
        ],
    )

    if not runs:
        raise RuntimeError("No candidate runs were found in MLflow.")

    best_run = runs[0]

    best_f1 = best_run.data.metrics["cv_val_macro_f1"]
    best_model_family = best_run.data.params["model_family"]
    best_config = best_run.data.params["config_number"]

    print("\nBest candidate selected:")
    print(f"Model family: {best_model_family}")
    print(f"Configuration: {best_config}")
    print(f"Validation Macro F1: {best_f1:.4f}")
    print(f"Run ID: {best_run.info.run_id}")

    # ------------------------------------------------------------
    # Register the best model
    # ------------------------------------------------------------

    model_uri = f"runs:/{best_run.info.run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    # Assign the champion alias
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered_model.version,
    )

    print("\nModel registered successfully:")
    print(f"Registered model: {REGISTERED_MODEL_NAME}")
    print(f"Version: {registered_model.version}")
    print("Alias: champion")


if __name__ == "__main__":
    train_candidates()
