"""
1-Nearest Neighbor Classifier
"""

import numpy as np


def knn_predict(X_prototypes, y_prototypes, X_test, batch_size=1000):
    """
    1-NN classification using prototypes.

    Args:
        X_prototypes: (M, 784) prototype features
        y_prototypes: (M,) prototype labels
        X_test: (N, 784) test features
        batch_size: Process test points in batches to manage memory

    Returns:
        y_pred: (N,) predicted labels
    """
    n_test = X_test.shape[0]
    y_pred = np.zeros(n_test, dtype=np.int32)

    # Process in batches to avoid memory issues
    for i in range(0, n_test, batch_size):
        end = min(i + batch_size, n_test)
        X_batch = X_test[i:end]

        # Compute distances: (batch_size, M)
        # Using ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        X_batch_sq = np.sum(X_batch ** 2, axis=1, keepdims=True)  # (batch, 1)
        X_proto_sq = np.sum(X_prototypes ** 2, axis=1)  # (M,)
        cross_term = X_batch @ X_prototypes.T  # (batch, M)

        distances_sq = X_batch_sq + X_proto_sq - 2 * cross_term

        # Find nearest prototype for each test point
        nearest_idx = np.argmin(distances_sq, axis=1)
        y_pred[i:end] = y_prototypes[nearest_idx]

    return y_pred


def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)


if __name__ == '__main__':
    # Simple test
    np.random.seed(42)

    # Create dummy data
    X_proto = np.random.randn(100, 784).astype(np.float32)
    y_proto = np.random.randint(0, 10, 100)
    X_test = np.random.randn(50, 784).astype(np.float32)

    y_pred = knn_predict(X_proto, y_proto, X_test)
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Predictions: {y_pred[:10]}")
