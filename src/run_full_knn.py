"""
Run Full 1-NN (Upper Bound) Experiment

This uses all 60000 training samples as "prototypes" to establish
the upper bound accuracy for prototype selection methods.
"""

import os
import sys
import time
import numpy as np

from data_loader import load_mnist
from knn_classifier import knn_predict, compute_accuracy


def main():
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist(data_dir)

    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    # Run full 1-NN (using all training data)
    print("\n" + "=" * 60)
    print("Running Full 1-NN (Upper Bound)")
    print("  M = 60000 (all training samples)")
    print("=" * 60)

    t0 = time.time()
    y_pred = knn_predict(X_train, y_train, X_test)
    time_predict = time.time() - t0

    # Compute accuracy
    accuracy = compute_accuracy(y_test, y_pred)

    # Per-class accuracy
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Prediction Time: {time_predict:.2f}s")

    print("\nPer-class Accuracy:")
    classes = np.unique(y_train)
    for c in classes:
        mask = y_test == c
        class_acc = (y_pred[mask] == y_test[mask]).mean()
        n_samples = mask.sum()
        print(f"  Class {c}: {class_acc:.4f} ({n_samples} samples)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Full 1-NN Accuracy (Upper Bound): {accuracy:.4f}")
    print(f"This is the maximum achievable accuracy for prototype selection.")
    print("=" * 60)


if __name__ == '__main__':
    main()
