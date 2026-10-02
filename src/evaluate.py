import time

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import load_wine_data


TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "WineClassifier"
MODEL_ALIAS = "champion"


def evaluate_champion():
    """Evaluate the registered champion model on the held-out test set."""

    # Connect to the same MLflow tracking database
    mlflow.set_tracking_uri(TRACKING_URI)

    # Load the original train/test split
    X_train, X_test, y_train, y_test = load_wine_data()

    # Load the model registered with the champion alias
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

    print("Loading registered champion model...")
    model = mlflow.sklearn.load_model(model_uri)

    # Warm-up prediction so model-loading overhead is not included
    model.predict(X_test)

    # Measure batch inference latency
    start_time = time.perf_counter()

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    elapsed_time = time.perf_counter() - start_time
    latency_ms = elapsed_time * 1000

    # Calculate final test metrics
    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    test_log_loss = log_loss(
        y_test,
        probabilities,
    )

    # Get the unique predicted classes
    predicted_classes = sorted(set(predictions))

    # Display results
    print("\n========================================")
    print("FINAL CHAMPION MODEL EVALUATION")
    print("========================================")

    print(f"Registered model: {REGISTERED_MODEL_NAME}")
    print(f"Model alias: {MODEL_ALIAS}")

    print("\nTest Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Log Loss: {test_log_loss:.4f}")

    print("\nInference:")
    print(f"Batch size: {len(X_test)}")
    print(f"Batch inference latency: {latency_ms:.2f} ms")
    print(f"Predicted classes: {predicted_classes}")

    # Quality gate checks
    print("\n========================================")
    print("QUALITY GATE")
    print("========================================")

    f1_pass = macro_f1 >= 0.88
    latency_pass = latency_ms <= 30
    classes_pass = set(predicted_classes).issubset({0, 1, 2})

    print(
        f"Macro F1 >= 0.88: "
        f"{'PASS' if f1_pass else 'FAIL'}"
    )

    print(
        f"Batch inference <= 30 ms: "
        f"{'PASS' if latency_pass else 'FAIL'}"
    )

    print(
        f"Classes only 0/1/2: "
        f"{'PASS' if classes_pass else 'FAIL'}"
    )

    if not (f1_pass and latency_pass and classes_pass):
        raise AssertionError("Model quality gate failed.")

    print("\nALL QUALITY GATES PASSED.")


if __name__ == "__main__":
    evaluate_champion()
