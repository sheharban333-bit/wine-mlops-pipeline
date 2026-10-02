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
    mlflow.set_tracking_uri(TRACKING_URI)

    _, X_test, _, y_test = load_wine_data()

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

    print("Loading registered champion model...")
    model = mlflow.sklearn.load_model(model_uri)

    # Warm up the model before measuring inference latency.
    for _ in range(5):
        model.predict(X_test)

    # Measure average batch inference latency.
    repetitions = 30

    start_time = time.perf_counter()

    predictions = None

    for _ in range(repetitions):
        predictions = model.predict(X_test)

    elapsed_time = time.perf_counter() - start_time

    latency_ms = (elapsed_time / repetitions) * 1000

    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    test_log_loss = log_loss(
        y_test,
        probabilities,
    )

    predicted_classes = sorted(set(predictions))

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
    print(f"Average batch inference latency: {latency_ms:.2f} ms")
    print(f"Predicted classes: {predicted_classes}")

    print("\n========================================")
    print("QUALITY GATE")
    print("========================================")

    f1_pass = macro_f1 >= 0.88
    latency_pass = latency_ms <= 30
    classes_pass = set(predicted_classes).issubset({0, 1, 2})

    print(f"Macro F1 >= 0.88: {'PASS' if f1_pass else 'FAIL'}")
    print(f"Batch inference <= 30 ms: {'PASS' if latency_pass else 'FAIL'}")
    print(f"Classes only 0/1/2: {'PASS' if classes_pass else 'FAIL'}")

    if not (f1_pass and latency_pass and classes_pass):
        raise AssertionError("Model quality gate failed.")

    print("\nALL QUALITY GATES PASSED.")


if __name__ == "__main__":
    evaluate_champion()
