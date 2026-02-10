"""
Main entry point for prototype selection experiments.

Usage:
    python main.py                          # Run full experiments (default algorithms)
    python main.py --quick                  # Quick test
    python main.py --methods variance_centroid knn_boundary random
    python main.py --M 1000 500             # Custom M values
"""

import os
import sys
import argparse
import numpy as np

from data_loader import load_mnist
from experiments import (
    run_experiments,
    print_results_table,
    print_comparison_table,
    save_results,
    ALGORITHMS
)


def main():
    parser = argparse.ArgumentParser(description='Prototype Selection Experiments')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with smaller M values')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing MNIST data')
    parser.add_argument('--n-trials', type=int, default=5,
                        help='Number of trials for each experiment')
    parser.add_argument('--methods', nargs='+', default=None,
                        help=f'Methods to test. Available: {list(ALGORITHMS.keys())}')
    parser.add_argument('--M', nargs='+', type=int, default=None,
                        help='M values to test')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save detailed results (JSON)')
    args = parser.parse_args()

    # Set data directory
    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    # Load MNIST
    print("Loading MNIST dataset...")
    try:
        X_train, y_train, X_test, y_test = load_mnist(args.data_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease download MNIST data from:")
        print("  https://www.kaggle.com/datasets/hojjatk/mnist-dataset")
        print(f"\nAnd place the files in: {os.path.abspath(args.data_dir)}")
        print("\nRequired files:")
        print("  - train-images-idx3-ubyte")
        print("  - train-labels-idx1-ubyte")
        print("  - t10k-images-idx3-ubyte")
        print("  - t10k-labels-idx1-ubyte")
        sys.exit(1)

    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    # Define M values and methods
    if args.quick:
        M_values = [500, 100] if args.M is None else args.M
        n_trials = 2
        methods = ['variance_centroid', 'random'] if args.methods is None else args.methods
        print("\n[Quick mode: using smaller M values and fewer trials]")
    else:
        M_values = [10000, 5000, 1000] if args.M is None else args.M
        n_trials = args.n_trials
        methods = ['variance_centroid', 'random'] if args.methods is None else args.methods

    print(f"\nConfiguration:")
    print(f"  M values: {M_values}")
    print(f"  Methods: {methods}")
    print(f"  Trials: {n_trials}")

    # Run experiments
    all_results = {}
    detailed_results = []

    total_experiments = len(M_values) * len(methods)
    current = 0

    for M in M_values:
        for method in methods:
            current += 1
            print(f"\n{'='*60}")
            print(f"[{current}/{total_experiments}] Running: M={M}, method={method}")
            print('='*60)

            results = run_experiments(
                X_train, y_train, X_test, y_test,
                M=M, method=method, n_trials=n_trials
            )
            all_results[(M, method)] = results
            detailed_results.append(results)

            print(f"\nResult: {results['mean']:.4f} +/- {results['ci_95']:.4f}")
            print(f"  Time: {results['mean_time_select']:.2f}s (select) + {results['mean_time_predict']:.2f}s (predict)")

    # Print summary tables
    print_results_table(all_results)
    print_comparison_table(all_results, M_values, methods)

    # Save results
    results_file = os.path.join(os.path.dirname(__file__), '..', 'results.txt')
    with open(results_file, 'w') as f:
        f.write("Prototype Selection Experiment Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  M values: {M_values}\n")
        f.write(f"  Methods: {methods}\n")
        f.write(f"  Trials: {n_trials}\n\n")

        for (M, method), results in sorted(all_results.items()):
            f.write(f"M={M}, method={method}\n")
            f.write(f"  Mean accuracy: {results['mean']:.4f}\n")
            f.write(f"  Std: {results['std']:.4f}\n")
            f.write(f"  95% CI: [{results['mean']-results['ci_95']:.4f}, {results['mean']+results['ci_95']:.4f}]\n")
            f.write(f"  All accuracies: {results['all_accuracies']}\n")
            f.write(f"  Time (select): {results['mean_time_select']:.2f}s\n")
            f.write(f"  Per-class accuracy: {results['per_class_accuracy']}\n")
            f.write(f"  Prototype distribution: {results['prototype_distribution']}\n")
            f.write("\n")

    print(f"\nResults saved to: {os.path.abspath(results_file)}")

    # Save detailed JSON results
    if args.output_dir:
        json_path = save_results(detailed_results, args.output_dir)
        print(f"Detailed results saved to: {json_path}")


if __name__ == '__main__':
    main()
