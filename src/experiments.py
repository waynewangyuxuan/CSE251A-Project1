"""
Experiment Runner

Runs prototype selection experiments and computes statistics.
"""

import numpy as np
import time
from prototype_selection import (
    select_prototypes_variance_centroid,
    select_prototypes_random
)
from knn_classifier import knn_predict, compute_accuracy


def run_single_experiment(X_train, y_train, X_test, y_test, M, method, random_state=None):
    """
    Run a single prototype selection experiment.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        M: Number of prototypes
        method: 'ours' or 'random'
        random_state: Random seed

    Returns:
        accuracy: Classification accuracy on test set
        time_select: Time for prototype selection
        time_predict: Time for prediction
    """
    # Select prototypes
    t0 = time.time()
    if method == 'ours':
        indices = select_prototypes_variance_centroid(
            X_train, y_train, M, random_state=random_state
        )
    elif method == 'random':
        indices = select_prototypes_random(
            X_train, y_train, M, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    time_select = time.time() - t0

    # Get prototypes
    X_proto = X_train[indices]
    y_proto = y_train[indices]

    # Predict
    t0 = time.time()
    y_pred = knn_predict(X_proto, y_proto, X_test)
    time_predict = time.time() - t0

    # Compute accuracy
    accuracy = compute_accuracy(y_test, y_pred)

    return accuracy, time_select, time_predict


def run_experiments(X_train, y_train, X_test, y_test, M, method, n_trials=5):
    """
    Run multiple experiments and compute statistics.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        M: Number of prototypes
        method: 'ours' or 'random'
        n_trials: Number of trials for computing confidence intervals

    Returns:
        results: dict with mean, std, ci, all_accuracies
    """
    accuracies = []
    times_select = []
    times_predict = []

    for trial in range(n_trials):
        acc, t_sel, t_pred = run_single_experiment(
            X_train, y_train, X_test, y_test,
            M, method, random_state=trial
        )
        accuracies.append(acc)
        times_select.append(t_sel)
        times_predict.append(t_pred)

        print(f"  Trial {trial + 1}/{n_trials}: accuracy = {acc:.4f}")

    accuracies = np.array(accuracies)
    times_select = np.array(times_select)
    times_predict = np.array(times_predict)

    # Compute statistics
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)  # Sample std

    # 95% confidence interval: mean +/- 1.96 * std / sqrt(n)
    ci_95 = 1.96 * std_acc / np.sqrt(n_trials)

    return {
        'mean': mean_acc,
        'std': std_acc,
        'ci_95': ci_95,
        'all_accuracies': accuracies,
        'mean_time_select': np.mean(times_select),
        'mean_time_predict': np.mean(times_predict),
    }


def print_results_table(all_results):
    """
    Print results in a formatted table.

    Args:
        all_results: dict mapping (M, method) to results
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTAL RESULTS")
    print("=" * 70)
    print(f"{'M':>8} {'Method':>12} {'Accuracy':>12} {'95% CI':>16} {'Std':>10}")
    print("-" * 70)

    for (M, method), results in sorted(all_results.items()):
        mean = results['mean']
        ci = results['ci_95']
        std = results['std']
        print(f"{M:>8} {method:>12} {mean:>12.4f} {f'[{mean-ci:.4f}, {mean+ci:.4f}]':>16} {std:>10.4f}")

    print("=" * 70)


def print_comparison_table(all_results, M_values):
    """
    Print comparison between our method and random selection.

    Args:
        all_results: dict mapping (M, method) to results
        M_values: list of M values tested
    """
    print("\n" + "=" * 80)
    print("COMPARISON: Our Method vs Random Selection")
    print("=" * 80)
    print(f"{'M':>8} {'Ours':>16} {'Random':>16} {'Improvement':>16}")
    print("-" * 80)

    for M in M_values:
        ours = all_results.get((M, 'ours'), {}).get('mean', 0)
        ours_ci = all_results.get((M, 'ours'), {}).get('ci_95', 0)
        random = all_results.get((M, 'random'), {}).get('mean', 0)
        random_ci = all_results.get((M, 'random'), {}).get('ci_95', 0)
        improvement = ours - random

        ours_str = f"{ours:.4f} +/- {ours_ci:.4f}"
        random_str = f"{random:.4f} +/- {random_ci:.4f}"
        imp_str = f"{improvement:+.4f}"

        print(f"{M:>8} {ours_str:>16} {random_str:>16} {imp_str:>16}")

    print("=" * 80)
