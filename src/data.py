from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20
EXPECTED_FEATURES = 13


def load_wine_data():
    """Load the Wine dataset and create a stratified train/test split."""
    wine = load_wine(as_frame=True)

    X = wine.data
    y = wine.target

    if X.isnull().any().any():
        raise ValueError("Wine dataset contains missing feature values.")

    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, got {X.shape[1]}."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test
