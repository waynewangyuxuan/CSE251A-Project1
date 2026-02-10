"""Run sklearn logistic regression baseline to get L*."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from data_loader import load_binary_wine
from logistic_loss import logistic_loss, augment

# sklearn minimizes:  (1/2)||w||^2 + C * sum_i log_loss_i
# Our loss:           (1/n) sum_i log_loss_i + (lam/2)||w||^2
# Dividing sklearn by C*n: (1/(2Cn))||w||^2 + (1/n) sum log_loss
# So lam = 1/(C*n).  With n=130, lam=0.01 → C = 1/(0.01*130) ≈ 0.769
LAM = 0.01
N_SAMPLES = 130  # class 1 + class 2
C_SKLEARN = 1.0 / (LAM * N_SAMPLES)


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
    print(f"sklearn w norm: {np.linalg.norm(w_sklearn[:-1]):.4f}")
    print(f"L* (regularized loss): {L_star:.8f}")

    return w_sklearn, L_star


if __name__ == "__main__":
    run_baseline()
