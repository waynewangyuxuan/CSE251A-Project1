"""Quick test of new algorithms on MNIST."""

import sys
import os
sys.path.insert(0, '/Users/waynewang/CSE251A-Project1/src')

from data_loader import load_mnist
from experiments import run_experiments, ALGORITHMS

print("Loading MNIST...")
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
X_train, y_train, X_test, y_test = load_mnist(data_dir)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Quick test with M=500
M = 500
n_trials = 2

methods = ['variance_centroid', 'boundary_first', 'cnn', 'random']

print(f"\n{'='*60}")
print(f"Quick test: M={M}, n_trials={n_trials}")
print(f"{'='*60}")

results = {}
for method in methods:
    print(f"\nTesting {method}...")
    result = run_experiments(X_train, y_train, X_test, y_test,
                            M=M, method=method, n_trials=n_trials, verbose=True)
    results[method] = result
    print(f"  Accuracy: {result['mean']:.4f} +/- {result['std']:.4f}")
    print(f"  Time: {result['mean_time_select']:.2f}s")

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Method':<20} {'Accuracy':>12} {'Time':>10}")
print("-"*42)
for method in methods:
    r = results[method]
    print(f"{method:<20} {r['mean']:>12.4f} {r['mean_time_select']:>10.2f}s")
