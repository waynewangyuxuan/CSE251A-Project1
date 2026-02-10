"""Run all coordinate descent experiments and generate plots."""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

from data_loader import load_binary_wine
from logistic_loss import augment, logistic_loss, sigmoid
from baseline import run_baseline, LAM
from coordinate_descent import coordinate_descent, sparse_coordinate_descent

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


def predict(w, X_aug):
    """Predict labels from augmented X and weights."""
    probs = sigmoid(X_aug @ w)
    return (probs >= 0.5).astype(int)


def accuracy(w, X_aug, y):
    return (predict(w, X_aug) == y).mean()


def run_all_experiments():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    X, y = load_binary_wine(standardize=True)
    X_aug = augment(X)

    # Get baseline L*
    w_sklearn, L_star = run_baseline()
    print(f"\n{'='*60}")
    print(f"L* = {L_star:.8f}  (lam = {LAM})")
    print(f"{'='*60}\n")

    max_iter = 2000
    np.random.seed(42)

    # All strategy combinations
    configs = [
        ('gauss-southwell', 'newton', 'GS + Newton'),
        ('gauss-southwell', 'adam',    'GS + Adam'),
        ('momentum',        'newton', 'Momentum + Newton'),
        ('momentum',        'adam',    'Momentum + Adam'),
        ('random',          'newton', 'Random + Newton'),
        ('random',          'adam',    'Random + Adam'),
    ]

    results = {}
    print(f"{'Method':<25} {'Final Loss':>14} {'Accuracy':>10}")
    print('-' * 52)

    for select, update, label in configs:
        w, losses = coordinate_descent(
            X_aug, y,
            select=select, update=update,
            max_iter=max_iter, record_every=1,
            lam=LAM, adam_lr=0.1, momentum_beta=0.9,
        )
        final_loss = losses[-1][1]
        acc = accuracy(w, X_aug, y)
        print(f"{label:<25} {final_loss:>14.8f} {acc:>10.4f}")
        results[label] = {
            'losses': losses,
            'final_loss': float(final_loss),
            'accuracy': float(acc),
            'w': w.tolist(),
        }

    # --- Plot 1: All methods comparison ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for label, res in results.items():
        iters = [l[0] for l in res['losses']]
        vals = [l[1] for l in res['losses']]
        ax.plot(iters, vals, label=label, linewidth=1.5)

    ax.axhline(y=L_star, color='black', linestyle='--', linewidth=1, label=f'L* = {L_star:.6f}')
    ax.set_xlabel('Iteration t')
    ax.set_ylabel('Loss L(w_t)')
    ax.set_title('Coordinate Descent: All Strategy Combinations')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'all_methods_comparison.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'all_methods_comparison.png'), dpi=150)
    print(f"\nSaved: all_methods_comparison.pdf/png")

    # --- Plot 2: Our method vs Random (for report requirement) ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for label in ['Momentum + Newton', 'Random + Newton']:
        iters = [l[0] for l in results[label]['losses']]
        vals = [l[1] for l in results[label]['losses']]
        ax.plot(iters, vals, label=label, linewidth=2)

    ax.axhline(y=L_star, color='black', linestyle='--', linewidth=1, label=f'L* = {L_star:.6f}')
    ax.set_xlabel('Iteration t')
    ax.set_ylabel('Loss L(w_t)')
    ax.set_title('Adaptive vs Random Coordinate Descent')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'adaptive_vs_random.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'adaptive_vs_random.png'), dpi=150)
    print(f"Saved: adaptive_vs_random.pdf/png")

    # Save numeric results
    save_results = {
        'L_star': float(L_star),
        'lam': LAM,
        'max_iter': max_iter,
    }
    for label, res in results.items():
        save_results[label] = {
            'final_loss': res['final_loss'],
            'accuracy': res['accuracy'],
            'w': res['w'],
        }
    with open(os.path.join(RESULTS_DIR, 'experiment_results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"Saved: experiment_results.json")

    return results, L_star


def run_sparse_experiments():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    X, y = load_binary_wine(standardize=True)
    X_aug = augment(X)
    d_features = X.shape[1]  # 13 features (excluding bias)

    w_sklearn, L_star = run_baseline()
    print(f"\nL* (full, regularized) = {L_star:.8f}")

    # Full (non-sparse) baseline with our method
    w_full, losses_full = coordinate_descent(
        X_aug, y, select='momentum', update='newton',
        max_iter=2000, lam=LAM, momentum_beta=0.9,
    )
    L_full = losses_full[-1][1]
    acc_full = accuracy(w_full, X_aug, y)
    nnz_full = np.count_nonzero(w_full[:-1])
    print(f"Full CD: loss={L_full:.8f}, acc={acc_full:.4f}, nnz={nnz_full}")

    # Sparse experiments for various k
    k_values = [1, 2, 3, 4, 5, 7, 9, 11, 13]
    sparse_results = []

    print(f"\n{'k':>3} | {'Loss':>14} | {'Acc':>6} | {'NNZ':>3} | Active features")
    print('-' * 65)

    for k in k_values:
        np.random.seed(42)
        w_sparse, losses_sparse = sparse_coordinate_descent(
            X_aug, y, k=k, select='momentum', update='newton',
            max_iter=3000, lam=LAM, momentum_beta=0.9,
        )
        loss = losses_sparse[-1][1]
        acc = accuracy(w_sparse, X_aug, y)
        active = [j for j in range(d_features) if w_sparse[j] != 0.0]
        print(f"{k:>3} | {loss:>14.8f} | {acc:>6.4f} | {len(active):>3} | {active}")
        sparse_results.append({
            'k': k,
            'loss': float(loss),
            'accuracy': float(acc),
            'n_nonzero': len(active),
            'active_features': active,
            'w': w_sparse.tolist(),
        })

    # Save sparse results
    with open(os.path.join(RESULTS_DIR, 'sparse_results.json'), 'w') as f:
        json.dump({
            'L_star': float(L_star),
            'L_full_cd': float(L_full),
            'lam': LAM,
            'sparse': sparse_results,
        }, f, indent=2)
    print(f"\nSaved: sparse_results.json")

    # --- Plot: Loss vs k ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ks = [r['k'] for r in sparse_results]
    losses = [r['loss'] for r in sparse_results]

    ax.plot(ks, losses, 'o-', linewidth=2, markersize=8, label='Sparse CD')
    ax.axhline(y=L_star, color='black', linestyle='--', linewidth=1, label=f'L* = {L_star:.6f}')
    ax.set_xlabel('Sparsity budget k')
    ax.set_ylabel('Loss')
    ax.set_title('Sparse Coordinate Descent: Loss vs Sparsity')
    ax.set_xticks(ks)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'sparse_loss_vs_k.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'sparse_loss_vs_k.png'), dpi=150)
    print(f"Saved: sparse_loss_vs_k.pdf/png")

    return sparse_results


if __name__ == "__main__":
    run_all_experiments()
    print("\n" + "=" * 60)
    print("SPARSE EXPERIMENTS")
    print("=" * 60)
    run_sparse_experiments()
