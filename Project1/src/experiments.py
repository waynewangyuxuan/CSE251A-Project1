"""
Experiment Runner

Runs prototype selection experiments and computes statistics.
Supports multiple algorithms with progress tracking and detailed logging.
"""

import numpy as np
import time
import json
import os
from datetime import datetime

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars.")

from prototype_selection import (
    select_prototypes_variance_centroid,
    select_prototypes_cluster_boundary,
    select_prototypes_boundary_first,
    select_prototypes_cnn,
    select_prototypes_random
)
from knn_classifier import knn_predict, compute_accuracy


# Registry of available algorithms
ALGORITHMS = {
    'variance_centroid': select_prototypes_variance_centroid,
    'cluster_boundary': select_prototypes_cluster_boundary,
    'boundary_first': select_prototypes_boundary_first,
    'cnn': select_prototypes_cnn,
    'random': select_prototypes_random,
    # Aliases for backward compatibility
    'ours': select_prototypes_variance_centroid,
}


def run_single_experiment(X_train, y_train, X_test, y_test, M, method, random_state=None, **kwargs):
    """
    Run a single prototype selection experiment.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        M: Number of prototypes
        method: Algorithm name (see ALGORITHMS)
        random_state: Random seed
        **kwargs: Additional arguments for the algorithm

    Returns:
        dict: Detailed results including accuracy, timing, and per-class metrics
    """
    if method not in ALGORITHMS:
        raise ValueError(f"Unknown method: {method}. Available: {list(ALGORITHMS.keys())}")

    algorithm = ALGORITHMS[method]

    # Select prototypes
    t0 = time.time()
    indices = algorithm(X_train, y_train, M, random_state=random_state)
    time_select = time.time() - t0

    # Get prototypes
    X_proto = X_train[indices]
    y_proto = y_train[indices]

    # Predict
    t0 = time.time()
    y_pred = knn_predict(X_proto, y_proto, X_test)
    time_predict = time.time() - t0

    # Compute metrics
    accuracy = compute_accuracy(y_test, y_pred)

    # Per-class accuracy
    classes = np.unique(y_train)
    per_class_acc = {}
    for c in classes:
        mask = y_test == c
        if mask.sum() > 0:
            per_class_acc[int(c)] = float((y_pred[mask] == y_test[mask]).mean())

    # Prototype distribution per class
    proto_distribution = {}
    for c in classes:
        proto_distribution[int(c)] = int((y_proto == c).sum())

    return {
        'accuracy': accuracy,
        'time_select': time_select,
        'time_predict': time_predict,
        'per_class_accuracy': per_class_acc,
        'prototype_distribution': proto_distribution,
        'n_prototypes': len(indices),
    }


def run_experiments(X_train, y_train, X_test, y_test, M, method, n_trials=5,
                    verbose=True, **kwargs):
    """
    Run multiple experiments and compute statistics.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        M: Number of prototypes
        method: Algorithm name
        n_trials: Number of trials
        verbose: Print progress
        **kwargs: Additional arguments for the algorithm

    Returns:
        dict: Aggregated results with statistics
    """
    all_results = []

    iterator = range(n_trials)
    if verbose and HAS_TQDM:
        iterator = tqdm(iterator, desc=f"M={M}, {method}", leave=False)

    for trial in iterator:
        result = run_single_experiment(
            X_train, y_train, X_test, y_test,
            M, method, random_state=trial, **kwargs
        )
        all_results.append(result)

        if verbose and not HAS_TQDM:
            print(f"  Trial {trial + 1}/{n_trials}: accuracy = {result['accuracy']:.4f}")

    # Aggregate statistics
    accuracies = np.array([r['accuracy'] for r in all_results])
    times_select = np.array([r['time_select'] for r in all_results])
    times_predict = np.array([r['time_predict'] for r in all_results])

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1) if n_trials > 1 else 0.0
    ci_95 = 1.96 * std_acc / np.sqrt(n_trials) if n_trials > 1 else 0.0

    # Aggregate per-class accuracy
    per_class_mean = {}
    classes = all_results[0]['per_class_accuracy'].keys()
    for c in classes:
        class_accs = [r['per_class_accuracy'][c] for r in all_results]
        per_class_mean[c] = float(np.mean(class_accs))

    return {
        'mean': mean_acc,
        'std': std_acc,
        'ci_95': ci_95,
        'all_accuracies': accuracies.tolist(),
        'mean_time_select': float(np.mean(times_select)),
        'mean_time_predict': float(np.mean(times_predict)),
        'per_class_accuracy': per_class_mean,
        'prototype_distribution': all_results[0]['prototype_distribution'],
        'n_trials': n_trials,
        'method': method,
        'M': M,
    }


def save_results(results, output_dir, filename=None):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"

    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    return filepath


def print_results_table(all_results):
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("EXPERIMENTAL RESULTS")
    print("=" * 80)
    print(f"{'M':>8} {'Method':>18} {'Accuracy':>12} {'95% CI':>20} {'Time(s)':>10}")
    print("-" * 80)

    for (M, method), results in sorted(all_results.items()):
        mean = results['mean']
        ci = results['ci_95']
        t = results['mean_time_select']
        ci_str = f"[{mean-ci:.4f}, {mean+ci:.4f}]"
        print(f"{M:>8} {method:>18} {mean:>12.4f} {ci_str:>20} {t:>10.2f}")

    print("=" * 80)


def print_comparison_table(all_results, M_values, methods):
    """Print comparison between methods."""
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)

    # Header
    header = f"{'M':>8}"
    for method in methods:
        header += f" {method:>20}"
    print(header)
    print("-" * 100)

    for M in M_values:
        row = f"{M:>8}"
        for method in methods:
            result = all_results.get((M, method), {})
            if result:
                mean = result.get('mean', 0)
                ci = result.get('ci_95', 0)
                row += f" {mean:.4f} +/- {ci:.4f}".rjust(20)
            else:
                row += " N/A".rjust(20)
        print(row)

    print("=" * 100)
