"""Ridge-regression readouts trained on the agents' opinions, and the benchmarks they're scored on."""

import numpy as np

ALPHA_GRID = np.logspace(-8, 0, 9)


def add_bias(X):
    return np.hstack([X, np.ones((len(X), 1))])


def ridge_fit(X, y, alpha=1e-4):
    """Closed-form ridge. X: (T, N), y: (T,) or (T, K). Returns weights with bias as the last row.

    The penalty is alpha * T, so alpha is per-sample and comparable across run lengths.
    """
    Xb = add_bias(X)
    G = Xb.T @ Xb
    G[np.diag_indices_from(G)] += alpha * len(Xb)
    return np.linalg.solve(G, Xb.T @ y)


def ridge_predict(X, W):
    return add_bias(X) @ W


def cv_alpha(X, Y, alphas=ALPHA_GRID, val_frac=0.2):
    """Pick alpha on a validation block at the end of the training data (no peeking at test)."""
    cut = int(len(X) * (1 - val_frac))
    errs = [np.mean((ridge_predict(X[cut:], ridge_fit(X[:cut], Y[:cut], a)) - Y[cut:]) ** 2) for a in alphas]
    return float(alphas[int(np.argmin(errs))])


def r2(y, yhat):
    """Squared correlation, the standard memory-capacity score (clipped to [0, 1])."""
    if np.std(yhat) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(y, yhat)[0, 1] ** 2)


def nrmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)) / np.std(y))


def split(X, washout, train_frac=0.7):
    T = len(X) - washout
    cut = washout + int(train_frac * T)
    return slice(washout, cut), slice(cut, len(X))


def memory_capacity(X, u, max_delay=30, washout=200, alpha=1e-4, train_frac=0.7):
    """Jaeger's short-term memory capacity: sum over delays k of r^2(u(t-k), readout_k(x(t))).

    Returns (total MC, per-delay curve). Scored on held-out data. ``alpha="cv"`` picks the
    ridge penalty on a validation block inside the training data.
    """
    assert washout >= max_delay, "washout must cover the longest delay (np.roll wraps around)"
    tr, te = split(X, washout, train_frac)
    delays = np.arange(1, max_delay + 1)
    Y = np.stack([np.roll(u, k) for k in delays], axis=1)  # column k-1 is u(t-k)
    if alpha == "cv":
        alpha = cv_alpha(X[tr], Y[tr])
    W = ridge_fit(X[tr], Y[tr], alpha)
    Yhat = ridge_predict(X[te], W)
    curve = np.array([r2(Y[te, j], Yhat[:, j]) for j in range(len(delays))])
    return float(curve.sum()), curve


def narma10(u01):
    """NARMA-10 target for an input in [0, 0.5]. The classic nonlinear-memory benchmark."""
    T = len(u01)
    y = np.zeros(T)
    for t in range(10, T - 1):
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * y[t - 9:t + 1].sum()
                    + 1.5 * u01[t - 9] * u01[t] + 0.1)
    return y


def narma_score(X, u, washout=200, alpha=1e-4):
    """NRMSE on NARMA-10 predicted one step ahead from the opinion state. Lower is better.

    The reservoir is driven by u in [-1, 1]; NARMA is computed on the same signal mapped to [0, 0.5].
    """
    y = narma10((u + 1) / 4)
    target = np.roll(y, -1)  # predict y(t+1) from x(t), which has seen u(t)
    tr, te = split(X[:-1], washout)
    W = ridge_fit(X[tr], target[tr], alpha)
    return nrmse(target[te], ridge_predict(X[te], W))


def participation_ratio(X):
    """Effective dimensionality of the states: (sum lambda)^2 / sum lambda^2 of their covariance."""
    lam = np.clip(np.linalg.eigvalsh(np.cov(X, rowvar=False)), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())
