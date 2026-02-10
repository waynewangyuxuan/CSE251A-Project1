"""Coordinate Descent for Logistic Regression.

Supports multiple strategies for:
  (a) Coordinate selection: gauss-southwell, momentum, random
  (b) Coordinate update: newton, adam
"""

import numpy as np
from logistic_loss import sigmoid, logistic_loss, logistic_gradient_j, logistic_hessian_diag_j


# ---------------------------------------------------------------------------
# Coordinate selection strategies
# ---------------------------------------------------------------------------

def _compute_full_grad(w, X, y, state, lam=0.0):
    """Compute full gradient and cache residual/sigma in state."""
    z = X @ w
    sigma = sigmoid(z)
    residual = sigma - y
    grad = (X.T @ residual) / len(y)
    # Add regularization gradient (exclude bias = last coord)
    if lam > 0:
        grad[:-1] += lam * w[:-1]
    state['residual'] = residual
    state['sigma'] = sigma
    return grad


def select_gauss_southwell(w, X, y, state):
    """Pick coordinate with largest |partial derivative|."""
    lam = state.get('lam', 0.0)
    grad = _compute_full_grad(w, X, y, state, lam)
    return np.argmax(np.abs(grad))


def select_momentum(w, X, y, state):
    """Pick coordinate using momentum-weighted gradient magnitude."""
    beta = state.get('momentum_beta', 0.9)
    lam = state.get('lam', 0.0)
    grad = _compute_full_grad(w, X, y, state, lam)

    if 'grad_ema' not in state:
        state['grad_ema'] = np.abs(grad)
    else:
        state['grad_ema'] = beta * state['grad_ema'] + (1 - beta) * np.abs(grad)

    score = np.abs(grad) + 0.5 * state['grad_ema']
    return np.argmax(score)


def select_random(w, X, y, state):
    """Pick coordinate uniformly at random."""
    lam = state.get('lam', 0.0)
    _compute_full_grad(w, X, y, state, lam)
    return np.random.randint(len(w))


# ---------------------------------------------------------------------------
# Coordinate update rules
# ---------------------------------------------------------------------------

def update_newton(w, X, y, j, state):
    """Newton step: w_j -= g_j / H_jj (clipped for stability)."""
    residual = state.get('residual')
    sigma = state.get('sigma')
    lam = state.get('lam', 0.0)

    g_j = logistic_gradient_j(w, X, y, j, lam=lam, precomputed_residual=residual)
    H_jj = logistic_hessian_diag_j(w, X, y, j, lam=lam, precomputed_sigma=sigma)

    if H_jj > 1e-12:
        step = g_j / H_jj
        step = np.clip(step, -10.0, 10.0)
        w[j] -= step
    else:
        w[j] -= 0.1 * g_j


def update_adam(w, X, y, j, state):
    """Adam-style update for a single coordinate."""
    d = len(w)
    beta1 = state.get('adam_beta1', 0.9)
    beta2 = state.get('adam_beta2', 0.999)
    lr = state.get('adam_lr', 0.1)
    lam = state.get('lam', 0.0)
    eps = 1e-8

    if 'adam_m' not in state:
        state['adam_m'] = np.zeros(d)
        state['adam_v'] = np.zeros(d)
        state['adam_t'] = np.zeros(d, dtype=int)

    residual = state.get('residual')
    g_j = logistic_gradient_j(w, X, y, j, lam=lam, precomputed_residual=residual)

    state['adam_t'][j] += 1
    t = state['adam_t'][j]

    state['adam_m'][j] = beta1 * state['adam_m'][j] + (1 - beta1) * g_j
    state['adam_v'][j] = beta2 * state['adam_v'][j] + (1 - beta2) * g_j ** 2

    m_hat = state['adam_m'][j] / (1 - beta1 ** t)
    v_hat = state['adam_v'][j] / (1 - beta2 ** t)

    w[j] -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ---------------------------------------------------------------------------
# Main coordinate descent loop
# ---------------------------------------------------------------------------

SELECTORS = {
    'gauss-southwell': select_gauss_southwell,
    'momentum': select_momentum,
    'random': select_random,
}

UPDATERS = {
    'newton': update_newton,
    'adam': update_adam,
}


def coordinate_descent(X, y, select='gauss-southwell', update='newton',
                       max_iter=1000, w_init=None, record_every=1, **kwargs):
    """Run coordinate descent.

    Args:
        X: (n, d) augmented feature matrix
        y: (n,) labels
        select: coordinate selection strategy
        update: coordinate update rule
        max_iter: number of iterations
        w_init: initial weights (zeros if None)
        record_every: record loss every N iterations
        **kwargs: extra params (lam, adam_lr, momentum_beta, etc.)
    """
    d = X.shape[1]
    w = w_init.copy() if w_init is not None else np.zeros(d)
    lam = kwargs.get('lam', 0.0)

    selector = SELECTORS[select]
    updater = UPDATERS[update]

    state = dict(kwargs)
    losses = [(0, logistic_loss(w, X, y, lam=lam))]

    for t in range(1, max_iter + 1):
        j = selector(w, X, y, state)
        updater(w, X, y, j, state)

        if t % record_every == 0 or t == max_iter:
            loss = logistic_loss(w, X, y, lam=lam)
            losses.append((t, loss))

    return w, losses


# ---------------------------------------------------------------------------
# Sparse Coordinate Descent
# ---------------------------------------------------------------------------

def sparse_coordinate_descent(X, y, k, select='momentum', update='newton',
                              max_iter=2000, w_init=None, record_every=1,
                              **kwargs):
    """Coordinate descent with k-sparsity constraint.

    Strategy:
      1. Start with all zeros (0-sparse).
      2. At each iteration, consider two moves:
         - Update one of the current active (nonzero) coordinates.
         - Swap: activate the best inactive coordinate, then prune the
           least useful active one to stay within budget k.
      3. Among active coordinates, use the same selector/updater logic.
      4. The bias term (last coordinate) is always active and doesn't
         count toward the sparsity budget.
    """
    d = X.shape[1]
    w = w_init.copy() if w_init is not None else np.zeros(d)
    bias_idx = d - 1
    lam = kwargs.get('lam', 0.0)

    updater = UPDATERS[update]
    state = dict(kwargs)
    losses = [(0, logistic_loss(w, X, y, lam=lam))]

    for t in range(1, max_iter + 1):
        z = X @ w
        sigma = sigmoid(z)
        residual = sigma - y
        grad = (X.T @ residual) / len(y)
        if lam > 0:
            grad[:-1] += lam * w[:-1]
        state['residual'] = residual
        state['sigma'] = sigma

        active = set(j for j in range(d - 1) if w[j] != 0.0)
        inactive = set(range(d - 1)) - active
        candidates = active | {bias_idx}

        if len(active) < k and inactive:
            best_inactive = max(inactive, key=lambda j: abs(grad[j]))
            candidates.add(best_inactive)
        elif len(active) >= k and inactive:
            best_inactive = max(inactive, key=lambda j: abs(grad[j]))
            if active:
                weakest_active = min(active, key=lambda j: abs(w[j]))
                if abs(grad[best_inactive]) > abs(grad[weakest_active]):
                    w[weakest_active] = 0.0
                    candidates.add(best_inactive)

        j = max(candidates, key=lambda j: abs(grad[j]))
        updater(w, X, y, j, state)

        feature_nonzero = [j for j in range(d - 1) if w[j] != 0.0]
        while len(feature_nonzero) > k:
            weakest = min(feature_nonzero, key=lambda j: abs(w[j]))
            w[weakest] = 0.0
            feature_nonzero.remove(weakest)

        if t % record_every == 0 or t == max_iter:
            loss = logistic_loss(w, X, y, lam=lam)
            losses.append((t, loss))

    return w, losses
