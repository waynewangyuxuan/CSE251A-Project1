"""Wine dataset loader — only class 1 & 2 for binary logistic regression."""

import numpy as np
from sklearn.datasets import load_wine


def load_binary_wine(standardize=True):
    """Load Wine dataset, keep only class 0 and class 1 (59 + 71 = 130 points).

    Note: sklearn's load_wine uses 0-indexed labels, so class 0 and class 1
    correspond to classes 1 and 2 in the original UCI dataset.

    Returns:
        X: (130, 13) feature matrix
        y: (130,) labels in {0, 1}
    """
    wine = load_wine()
    X_all, y_all = wine.data, wine.target

    # Keep only class 0 and class 1 (= UCI class 1 and 2)
    mask = y_all <= 1
    X = X_all[mask].astype(np.float64)
    y = y_all[mask].astype(np.float64)

    if standardize:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0  # avoid division by zero
        X = (X - mean) / std

    return X, y


if __name__ == "__main__":
    X, y = load_binary_wine()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class 0: {(y == 0).sum()}, Class 1: {(y == 1).sum()}")
