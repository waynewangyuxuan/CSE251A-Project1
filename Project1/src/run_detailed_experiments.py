"""
Run detailed experiments with more M values to observe trends.

M values: 10000, 7500, 5000, 2500, 1000, 500, 250, 100, 50, 25, 10
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_mnist
from experiments import run_experiments, save_results

# Configuration
# Already run: 10000, 5000, 1000, 500, 100
# New M values to fill in the gaps:
M_VALUES = [7500, 2500, 250, 50, 25, 10]
METHODS = ['variance_centroid', 'random']  # Focus on main method vs baseline
N_TRIALS = 5

def main():
    # Load data
    print("Loading MNIST...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Store all results
    all_results = {}

    print(f"\n{'='*70}")
    print(f"Running detailed experiments")
    print(f"M values: {M_VALUES}")
    print(f"Methods: {METHODS}")
    print(f"Trials per experiment: {N_TRIALS}")
    print(f"{'='*70}\n")

    for M in M_VALUES:
        print(f"\n{'='*50}")
        print(f"M = {M} (compression ratio: {60000//M}x)")
        print(f"{'='*50}")

        for method in METHODS:
            print(f"\n  Running {method}...")
            result = run_experiments(
                X_train, y_train, X_test, y_test,
                M=M, method=method, n_trials=N_TRIALS, verbose=True
            )
            all_results[(M, method)] = result

            print(f"  {method}: {result['mean']:.4f} +/- {result['std']:.4f} (time: {result['mean_time_select']:.2f}s)")

    # Print summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'M':>8} {'Compression':>12} {'variance_centroid':>20} {'random':>20} {'Improvement':>12}")
    print(f"{'-'*80}")

    for M in M_VALUES:
        vc = all_results.get((M, 'variance_centroid'), {})
        rand = all_results.get((M, 'random'), {})

        vc_acc = vc.get('mean', 0)
        vc_std = vc.get('std', 0)
        rand_acc = rand.get('mean', 0)
        rand_std = rand.get('std', 0)

        improvement = vc_acc - rand_acc
        compression = 60000 // M

        print(f"{M:>8} {compression:>10}x {vc_acc:.4f} +/- {vc_std:.4f}   {rand_acc:.4f} +/- {rand_std:.4f}   {improvement:>+.4f}")

    print(f"{'='*80}")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Convert tuple keys to string for JSON
    results_for_json = {
        'metadata': {
            'M_values': M_VALUES,
            'methods': METHODS,
            'n_trials': N_TRIALS,
            'timestamp': datetime.now().isoformat(),
        },
        'results': {
            f"M={M}_{method}": result
            for (M, method), result in all_results.items()
        }
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f'detailed_results_{timestamp}.json')

    with open(filepath, 'w') as f:
        json.dump(results_for_json, f, indent=2)

    print(f"\nResults saved to: {filepath}")

    # Also print in markdown format for easy copy to PROGRESS.md
    print(f"\n\n### Markdown Table for PROGRESS.md\n")
    print("| M | Compression | variance_centroid | random | Improvement |")
    print("|---|-------------|-------------------|--------|-------------|")
    for M in M_VALUES:
        vc = all_results.get((M, 'variance_centroid'), {})
        rand = all_results.get((M, 'random'), {})
        vc_acc = vc.get('mean', 0)
        vc_std = vc.get('std', 0)
        rand_acc = rand.get('mean', 0)
        improvement = vc_acc - rand_acc
        compression = 60000 // M
        print(f"| {M} | {compression}x | {vc_acc:.2%} +/- {vc_std:.2%} | {rand_acc:.2%} | +{improvement:.2%} |")

if __name__ == '__main__':
    main()
