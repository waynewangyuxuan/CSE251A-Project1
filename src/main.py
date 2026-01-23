"""
Main entry point for prototype selection experiments.

Usage:
    python main.py              # Run full experiments
    python main.py --quick      # Quick test with smaller M values
"""

import os
import sys
import argparse
import numpy as np

from data_loader import load_mnist
from experiments import (
    run_experiments,
    print_results_table,
    print_comparison_table
)


def main():
    parser = argparse.ArgumentParser(description='Prototype Selection Experiments')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with smaller M values')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing MNIST data')
    parser.add_argument('--n-trials', type=int, default=5,
                        help='Number of trials for each experiment')
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

    # Define M values
    if args.quick:
        M_values = [500, 100]
        n_trials = 2
        print("\n[Quick mode: using smaller M values and fewer trials]")
    else:
        M_values = [10000, 5000, 1000]
        n_trials = args.n_trials

    methods = ['ours', 'random']

    # Run experiments
    all_results = {}

    for M in M_values:
        for method in methods:
            print(f"\n{'='*50}")
            print(f"Running: M={M}, method={method}")
            print('='*50)

            results = run_experiments(
                X_train, y_train, X_test, y_test,
                M=M, method=method, n_trials=n_trials
            )
            all_results[(M, method)] = results

            print(f"\nResult: {results['mean']:.4f} +/- {results['ci_95']:.4f}")

    # Print summary tables
    print_results_table(all_results)
    print_comparison_table(all_results, M_values)

    # Save results to file
    results_file = os.path.join(os.path.dirname(__file__), '..', 'results.txt')
    with open(results_file, 'w') as f:
        f.write("Prototype Selection Experiment Results\n")
        f.write("=" * 50 + "\n\n")

        for (M, method), results in sorted(all_results.items()):
            f.write(f"M={M}, method={method}\n")
            f.write(f"  Mean accuracy: {results['mean']:.4f}\n")
            f.write(f"  Std: {results['std']:.4f}\n")
            f.write(f"  95% CI: [{results['mean']-results['ci_95']:.4f}, {results['mean']+results['ci_95']:.4f}]\n")
            f.write(f"  All accuracies: {results['all_accuracies']}\n")
            f.write("\n")

    print(f"\nResults saved to: {os.path.abspath(results_file)}")


if __name__ == '__main__':
    main()
