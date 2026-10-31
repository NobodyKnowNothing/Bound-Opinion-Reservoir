"""Close the loop: a recommender that uses the trained readout to decide who sees whom.

Every epoch the platform

1. fits a ridge readout on the epoch's opinion states to recover the recent news
   (it wants to know what its users "know");
2. scores every account by its salience to that readout, |W_j| * std(x_j): how much
   the platform's model leans on that account;
3. rewires: each agent, with probability ``rate``, unfollows one account and is
   recommended a new one, drawn with probability proportional to
   exp(beta_salience * salience_j + beta_agree * agreement_ij), where agreement
   is 1 - |x_i - x_j| / 2 (the engagement-friendly "you'll like this one").

With both betas at zero this is random rewiring at the same rate, the control.
"""

import copy
from dataclasses import dataclass

import numpy as np

from .model import OpinionReservoir
from .readout import memory_capacity, participation_ratio, ridge_fit


@dataclass
class Recommender:
    rate: float = 0.1
    beta_salience: float = 0.0
    beta_agree: float = 0.0
    delay: int = 1  # the readout target is u(t - delay)

    def salience(self, X, u):
        W = ridge_fit(X, np.roll(u, self.delay))[:-1]
        s = np.abs(W) * X.std(0)
        return s / (s.max() + 1e-12)

    def rewire(self, res: OpinionReservoir, sal):
        A, x, rng = res.A, res.x, res.rng
        n = len(x)
        for i in np.flatnonzero(rng.random(n) < self.rate):
            followed = np.flatnonzero(A[i])
            candidates = np.flatnonzero(~A[i])
            candidates = candidates[candidates != i]
            if len(followed) == 0 or len(candidates) == 0:
                continue
            agree = 1 - np.abs(x[i] - x[candidates]) / 2
            logits = self.beta_salience * sal[candidates] + self.beta_agree * agree
            prob = np.exp(logits - logits.max())
            A[i, rng.choice(followed)] = False
            A[i, rng.choice(candidates, p=prob / prob.sum())] = True


def gini(v):
    v = np.sort(v.astype(float))
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum() + 1e-12))


def mean_abs_corr(X):
    """Mean |correlation| between distinct columns of X (agents' opinion time series)."""
    C = np.corrcoef(X[:, X.std(0) > 1e-9], rowvar=False)
    off = ~np.eye(len(C), dtype=bool)
    return float(np.abs(C[off]).mean())


def probe(res, T=2000, seed=99, washout=200):
    """Probe a frozen copy of the society with a fresh signal: memory capacity, effective dimension
    of the states, and how alike the agents who don't watch the news have become."""
    frozen = copy.deepcopy(res)
    frozen.rng = np.random.default_rng(seed)
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    X = frozen.run(u)
    informed = res.w_in != 0
    return dict(
        mc=memory_capacity(X, u, washout=washout)[0],
        dimension=participation_ratio(X[washout:]),
        uninformed_corr=mean_abs_corr(X[washout:, ~informed]),
    )


def epoch_metrics(res, X, sal, every=10):
    """Structural metrics averaged over the epoch's states (not a single end-of-epoch snapshot)."""
    Xs = X[::every]
    close = np.abs(Xs[:, None, :] - Xs[:, :, None]) < res.p.eps  # (time, i, j)
    indeg = res.A.sum(0)
    informed = res.w_in != 0
    top = len(indeg) // 10
    return dict(
        # Fraction of follow links that actually transmit influence (within the confidence bound).
        active_links=float((res.A[None] & close).sum() / (len(Xs) * res.A.sum())),
        # Spread of opinions and how extreme they are.
        spread=float(Xs.std(1).mean()),
        extremity=float(np.abs(Xs).mean()),
        # Attention concentration: Gini of follower counts.
        indegree_gini=gini(indeg),
        # Share of the top-10% most-followed accounts that are directly exposed to the news.
        top_informed=float(informed[np.argsort(indeg)[-top:]].mean()),
        # The same share among the top-10% accounts by readout salience: where salience feedback should
        # send attention if it simply follows the readout.
        top_salient_informed=float(informed[np.argsort(sal)[-top:]].mean()),
        salience_on_informed=float(sal[informed].sum() / sal.sum()),
        salience_gini=gini(sal),
    )


def run_loop(params, rec: Recommender, epochs=40, epoch_len=500, probe_every=1, signal_seed=None,
             return_society=False):
    """Alternate driving the society and rewiring it. MC is probed every ``probe_every`` epochs (NaN otherwise).

    Each society seed gets its own news sequence, so runs don't share common shocks.
    With ``return_society`` the rewired society comes back too: (history, society).
    """
    res = OpinionReservoir(params)
    rng = np.random.default_rng(5000 + params.seed if signal_seed is None else signal_seed)
    history = []
    for e in range(epochs):
        u = rng.uniform(-1, 1, epoch_len)
        X = res.run(u)
        sal = rec.salience(X[50:], u[50:])
        m = epoch_metrics(res, X, sal)
        probed = e % probe_every == 0 or e == epochs - 1
        m.update(probe(res) if probed else dict(mc=np.nan, dimension=np.nan, uninformed_corr=np.nan))
        history.append(m)
        rec.rewire(res, sal)
    history = {k: np.array([h[k] for h in history]) for k in history[0]}
    return (history, res) if return_society else history
