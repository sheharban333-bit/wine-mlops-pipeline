from src.data import EXPECTED_FEATURES, load_wine_data


def test_wine_data_split():
    """Verify the Wine dataset is loaded and split correctly."""
    X_train, X_test, y_train, y_test = load_wine_data()

    # Check number of features
    assert X_train.shape[1] == EXPECTED_FEATURES
    assert X_test.shape[1] == EXPECTED_FEATURES

    # Check 80/20 split
    assert len(X_train) == 142
    assert len(X_test) == 36

    # Check for missing values
    assert not X_train.isnull().any().any()
    assert not X_test.isnull().any().any()

    # Check that all three Wine classes are present
    assert set(y_train.unique()) == {0, 1, 2}
    assert set(y_test.unique()) == {0, 1, 2}
