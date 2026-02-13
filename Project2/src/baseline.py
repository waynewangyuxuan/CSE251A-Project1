"""Run sklearn logistic regression baseline to get L*."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from data_loader import load_binary_wine
from logistic_loss import logistic_loss, augment

# No regularization per requirement.
# Set C very large to make sklearn's built-in regularization irrelevant.
LAM = 0.0
C_SKLEARN = 1e10


def run_baseline():
    X, y = load_binary_wine(standardize=True)
    X_aug = augment(X)  # (130, 14)

    clf = LogisticRegression(C=C_SKLEARN, solver='lbfgs', max_iter=10000,
                             fit_intercept=True)
    clf.fit(X, y)

    # Reconstruct w vector: [coef, intercept]
    w_sklearn = np.concatenate([clf.coef_.flatten(), clf.intercept_])
    L_star = logistic_loss(w_sklearn, X_aug, y, lam=LAM)

    print(f"sklearn accuracy: {clf.score(X, y):.4f}")
    print(f"L* (final loss): {L_star:.8f}")

    return w_sklearn, L_star


if __name__ == "__main__":
    run_baseline()
