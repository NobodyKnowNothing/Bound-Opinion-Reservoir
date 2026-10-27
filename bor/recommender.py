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
from .readout import memory_capacity, ridge_fit


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


def probe_mc(res, T=2000, seed=99):
    """Memory capacity of the current society, measured on a frozen copy with a fresh probe signal."""
    probe = copy.deepcopy(res)
    probe.rng = np.random.default_rng(seed)
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    return memory_capacity(probe.run(u), u)[0]


def epoch_metrics(res, sal):
    x = res.x
    close = np.abs(x[None, :] - x[:, None]) < res.p.eps
    indeg = res.A.sum(0)
    informed = res.w_in != 0
    top = np.argsort(indeg)[-len(x) // 10:]
    return dict(
        mc=probe_mc(res),
        # Fraction of follow links that actually transmit influence (within the confidence bound).
        active_links=float((res.A & close).sum() / res.A.sum()),
        # Spread of opinions and how extreme they are.
        spread=float(x.std()),
        extremity=float(np.abs(x).mean()),
        # Attention concentration: Gini of follower counts.
        indegree_gini=gini(indeg),
        # Share of the top-10% most-followed accounts that are directly exposed to the news.
        top_informed=float(informed[top].mean()),
        salience_gini=gini(sal),
    )


def run_loop(params, rec: Recommender, epochs=40, epoch_len=500, signal_seed=7):
    res = OpinionReservoir(params)
    rng = np.random.default_rng(signal_seed)
    history = []
    for _ in range(epochs):
        u = rng.uniform(-1, 1, epoch_len)
        X = res.run(u)
        sal = rec.salience(X[50:], u[50:])
        history.append(epoch_metrics(res, sal))
        rec.rewire(res, sal)
    return {k: np.array([h[k] for h in history]) for k in history[0]}
