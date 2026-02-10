"""Logistic regression loss, gradient, and Hessian diagonal.

Model: P(y=1|x) = sigmoid(w^T x + b)
Loss:  L(w,b) = -(1/n) sum[ y_i log(sigma_i) + (1-y_i) log(1-sigma_i) ]

We append a bias by augmenting X with a column of ones, so w includes bias.
"""

import numpy as np


def sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def augment(X):
    """Append a column of ones (for bias term)."""
    return np.hstack([X, np.ones((X.shape[0], 1))])


def logistic_loss(w, X, y, lam=0.0):
    """Compute average negative log-likelihood + L2 regularization.

    L(w) = (1/n) sum[-y*z + log(1+exp(z))] + (lam/2) ||w_features||^2

    Regularization excludes the bias (last element of w).
    """
    z = X @ w
    loss = -y * z + np.logaddexp(0, z)
    reg = 0.5 * lam * np.dot(w[:-1], w[:-1])  # exclude bias
    return loss.mean() + reg


def logistic_gradient(w, X, y):
    """Compute gradient of the loss w.r.t. w.

    grad_j = (1/n) sum_i (sigma_i - y_i) * x_ij
    """
    z = X @ w
    sigma = sigmoid(z)
    residual = sigma - y  # (n,)
    return (X.T @ residual) / len(y)


def logistic_gradient_j(w, X, y, j, lam=0.0, precomputed_residual=None):
    """Compute partial derivative w.r.t. w[j] only (with regularization)."""
    if precomputed_residual is None:
        z = X @ w
        sigma = sigmoid(z)
        residual = sigma - y
    else:
        residual = precomputed_residual
    g = (X[:, j] @ residual) / len(y)
    if lam > 0 and j < len(w) - 1:  # regularize features, not bias
        g += lam * w[j]
    return g


def logistic_hessian_diag_j(w, X, y, j, lam=0.0, precomputed_sigma=None):
    """Compute H[j,j] = (1/n) sum_i sigma_i(1-sigma_i) * x_ij^2 + lam."""
    if precomputed_sigma is None:
        z = X @ w
        sigma = sigmoid(z)
    else:
        sigma = precomputed_sigma
    sv = sigma * (1 - sigma)  # (n,)
    h = (sv @ (X[:, j] ** 2)) / len(y)
    if lam > 0 and j < len(w) - 1:
        h += lam
    return h
