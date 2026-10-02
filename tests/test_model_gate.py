import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data import RANDOM_STATE, load_wine_data


def create_gate_model():
    """Create a fast model that satisfies the quality gate."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def test_validation_macro_f1_gate():
    """Verify validation Macro F1 meets the required threshold."""
    X_train, _, y_train, _ = load_wine_data()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = create_gate_model()

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
    )

    mean_macro_f1 = scores.mean()

    print(f"\nValidation Macro F1: {mean_macro_f1:.4f}")

    assert mean_macro_f1 >= 0.88


def test_batch_inference_latency_and_classes():
    """Verify inference latency and valid output classes."""
    X_train, X_test, y_train, _ = load_wine_data()

    model = create_gate_model()
    model.fit(X_train, y_train)

    # Warm-up
    for _ in range(5):
        model.predict(X_test)

    repetitions = 30

    start_time = time.perf_counter()

    predictions = None

    for _ in range(repetitions):
        predictions = model.predict(X_test)

    elapsed_time = time.perf_counter() - start_time

    average_latency_ms = (
        elapsed_time / repetitions
    ) * 1000

    predicted_classes = set(
        np.unique(predictions)
    )

    print(
        f"\nAverage batch inference latency: "
        f"{average_latency_ms:.2f} ms"
    )

    print(
        f"Predicted classes: "
        f"{sorted(predicted_classes)}"
    )

    assert average_latency_ms <= 30

    assert predicted_classes.issubset({0, 1, 2})
