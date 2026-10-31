"""What causes the salience recommender's memory loss? Surgery on frozen societies.

Societies from the same seed share every trait, so follow lists can be spliced between policies.
Each condition is replayed with the same news and noise and scored with validated ridge; every
result is reported paired against the untouched salience society, and against control.

    python scripts/mechanism.py results/societies_0_20.npz
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import copy  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from societies import load  # noqa: E402

from bor import OpinionReservoir  # noqa: E402
from bor.readout import memory_capacity  # noqa: E402
from bor.stats import paired_stats  # noqa: E402

CONTROL, SALIENCE, AGREEMENT, BOTH = range(4)
T = 10_000
WASHOUT = 500
LAGS = 30


def spliced(base, other, rows):
    """base with the follow lists of ``rows`` taken from other."""
    res = copy.deepcopy(base)
    res.A[rows] = other.A[rows]
    return res


def same_mix_no_hubs(res, rng):
    """Every non-watcher keeps how many news-exposed and non-watcher accounts it follows, but the
    accounts are drawn uniformly, so the hubs in non-watchers' feeds go."""
    res = copy.deepcopy(res)
    informed = res.w_in != 0
    for i in np.flatnonzero(~informed):
        for pool_mask in (informed, ~informed):
            pool = np.flatnonzero(pool_mask)
            pool = pool[pool != i]
            n = int(res.A[i, pool].sum())
            res.A[i, pool] = False
            res.A[i, rng.choice(pool, n, replace=False)] = True
    return res


def retrait(res, rng, field, mask=None):
    res = copy.deepcopy(res)
    v = getattr(res, field).copy()
    idx = np.flatnonzero(np.ones(len(v), bool) if mask is None else mask)
    v[idx] = v[rng.permutation(idx)]
    setattr(res, field, v)
    return res


def wide(res):
    res = copy.deepcopy(res)
    res.p = replace(res.p, eps=2.0)
    return res


def conditions(soc, seed):
    rng = np.random.default_rng(seed + 1)
    c, s, a, b = (soc[i] for i in (CONTROL, SALIENCE, AGREEMENT, BOTH))
    informed = c.w_in != 0
    return {
        "control": c,
        "salience": s,
        "agreement": a,
        "both": b,
        # Q1: whose follow lists carry the loss?
        "salience, non-watcher rows from control": spliced(s, c, ~informed),
        "salience, exposed rows from control": spliced(s, c, informed),
        "salience, non-watcher rows from both": spliced(s, b, ~informed),
        "salience, exposed rows from both": spliced(s, b, informed),
        # Q2: is it the hubs?
        "salience, same mix, no hubs": same_mix_no_hubs(s, rng),
        # Q3: is it confidence gating?
        "control, eps 2": wide(c),
        "salience, eps 2": wide(s),
        "both, eps 2": wide(b),
        # Q4: is it the hubs' traits?
        "salience, shuffle w among exposed": retrait(s, rng, "w_in", informed),
        "salience, shuffle stubbornness": retrait(s, rng, "lam"),
        "salience, shuffle priors": retrait(s, rng, "prior"),
    }


class Surgery(OpinionReservoir):
    """Same dynamics, with the confidence gate replaced or overridden link by link."""
    always = None   # links that stay open whatever the opinions
    fixed = None    # a gate that never changes
    flicker = None  # each link opens at random each step with this probability
    gate_rng = None
    script = None   # recorded gate frames, over the links in A
    frame = 0
    tape = None     # when a list, the live gate is recorded here
    mean = None     # mean opinions, standing in for live ones where anchored
    anchor_rows = None
    anchor_cols = None

    def step(self, u):
        p, x = self.p, self.x
        if self.script is not None:  # replay a recorded gate, one frame per step
            close = np.zeros_like(self.A)
            close[self.A] = self.script[self.frame]
            self.frame += 1
        elif self.fixed is not None:
            close = self.fixed
        elif self.flicker is not None:
            close = self.gate_rng.random(self.A.shape) < self.flicker
        else:
            a = x if self.anchor_rows is None else np.where(self.anchor_rows, self.mean, x)  # follower side
            b = x if self.anchor_cols is None else np.where(self.anchor_cols, self.mean, x)  # followed side
            close = np.abs(b[None, :] - a[:, None]) < p.eps
        if self.tape is not None:
            self.tape.append(close[self.A])
        if self.always is not None:
            close = close | self.always
        M = (self.A & close) | np.eye(p.n_agents, dtype=bool)
        x = (1 - p.mu) * x + p.mu * (M @ x) / M.sum(1)
        x = x + self.lam * (self.prior - x)
        x = x + p.input_gain * self.w_in * u + p.noise * self.rng.standard_normal(p.n_agents)
        self.x = np.clip(x, -1.0, 1.0)
        return self.x


def surgery(res, **attrs):
    res = copy.deepcopy(res)
    res.__class__ = Surgery
    for k, v in attrs.items():
        setattr(res, k, v)
    return res


def warm(res, seed, steps=3000):
    """Opinions on a short private replay: per-link open rate and each agent's mean opinion."""
    frozen = copy.deepcopy(res)
    frozen.rng = np.random.default_rng(seed)
    X = frozen.run(np.random.default_rng(seed).uniform(-1, 1, steps))[500:]
    Xs = X[::5]
    return (np.abs(Xs[:, None, :] - Xs[:, :, None]) < res.p.eps).mean(0), X.mean(0)


def swap_links(res, rng, rows, mean, sign=0, tries=30000):
    """Double-edge swaps among ``rows``' follows. All in- and out-degrees are kept exactly, and so
    is every follower's link count. sign -1 pulls followers toward accounts near their own mean
    opinion, +1 pushes them apart, 0 randomises."""
    assert sign in (-1, 0, 1)
    res = copy.deepcopy(res)
    A = res.A
    i_, j_ = np.nonzero(A & rows[:, None])
    for _ in range(tries):
        a, b = rng.integers(len(i_), size=2)
        i, j, k, l = i_[a], j_[a], i_[b], j_[b]
        if i == k or j == l or i == l or k == j or A[i, l] or A[k, j]:
            continue
        change = abs(mean[i] - mean[l]) + abs(mean[k] - mean[j]) - abs(mean[i] - mean[j]) - abs(mean[k] - mean[l])
        if sign * change > 0 or sign == 0:
            A[i, j], A[k, l], A[i, l], A[k, j] = False, False, True, True
            j_[a], j_[b] = l, j
    return res


def conditions2(soc, seed):
    rng = np.random.default_rng(seed + 2)
    c, s, b = soc[CONTROL], soc[SALIENCE], soc[BOTH]
    informed = c.w_in != 0
    nw = ~informed
    out = {"control": c, "salience": s, "both": b}
    links = {
        "non-watcher to exposed": np.outer(nw, informed),
        "non-watcher to non-watcher": np.outer(nw, nw),
        "exposed to anyone": np.outer(informed, np.ones(len(nw), bool)),
    }
    for name, always in links.items():
        out[f"salience, always open: {name}"] = surgery(s, always=always)
        out[f"control, always open: {name}"] = surgery(c, always=always)
    for label, res in (("control", c), ("salience", s), ("both", b)):
        rate, _ = warm(res, seed)
        out[f"{label}, static gate, same open rate"] = surgery(res, fixed=rng.random(rate.shape) < rate)
        out[f"{label}, random flicker, same open rate"] = surgery(res, flicker=rate,
                                                                  gate_rng=np.random.default_rng(seed))
    _, mean_s = warm(s, seed)
    _, mean_b = warm(b, seed)
    out["salience, non-watcher links matched"] = swap_links(s, rng, nw, mean_s, -1)
    out["salience, non-watcher links shuffled"] = swap_links(s, rng, nw, mean_s, 0)
    out["both, non-watcher links unmatched"] = swap_links(b, rng, nw, mean_b, +1)
    out["both, non-watcher links shuffled"] = swap_links(b, rng, nw, mean_b, 0)
    out["both, shuffle priors"] = retrait(b, rng, "prior")
    for eps in (0.2, 0.4, 0.5, 0.7, 1.0):
        for label, res in (("control", c), ("salience", s)):
            frozen = copy.deepcopy(res)
            frozen.p = replace(frozen.p, eps=eps)
            out[f"{label}, eps {eps}"] = frozen
    return out


def record_gate(res, seed):
    """The gate frames of the live replay that score() runs for this seed."""
    u = np.random.default_rng(7000 + seed).uniform(-1, 1, T)
    frozen = surgery(res, tape=[])
    frozen.rng = np.random.default_rng(7000 + seed)
    frozen.run(u)
    return np.array(frozen.tape)


def block_shuffle(G, length, rng):
    n = len(G) // length
    order = rng.permutation(n)
    return np.concatenate([G[b * length:(b + 1) * length] for b in order] + [G[n * length:]])


def conditions3(soc, seed):
    rng = np.random.default_rng(seed + 3)
    c, s, b = soc[CONTROL], soc[SALIENCE], soc[BOTH]
    nw = c.w_in == 0
    out = {"control": c, "salience": s, "both": b}
    for label, res in (("control", c), ("salience", s), ("both", b)):
        G = record_gate(res, seed)
        out[f"{label}, recorded gate"] = surgery(res, script=G)
        out[f"{label}, gate from other news"] = surgery(res, script=record_gate(res, seed + 500))
        for length in (1, 10, 200, 2000):
            out[f"{label}, gate blocks of {length}"] = surgery(res, script=block_shuffle(G, length, rng))
    _, mean_s = warm(s, seed)
    _, mean_b = warm(b, seed)
    out["salience, non-watcher links matched"] = swap_links(s, rng, nw, mean_s, -1)
    out["salience, non-watcher links unmatched"] = swap_links(s, rng, nw, mean_s, +1)
    out["both, non-watcher links matched"] = swap_links(b, rng, nw, mean_b, -1)
    out["both, non-watcher links unmatched"] = swap_links(b, rng, nw, mean_b, +1)
    return out


def shifted(G, groups, rng):
    """Roll each group of links through time by its own random offset: every link keeps its open
    rate and persistence, while synchrony between groups goes."""
    G = G.copy()
    for g in np.unique(groups):
        idx = groups == g
        G[:, idx] = np.roll(G[:, idx], rng.integers(WASHOUT, T - WASHOUT), axis=0)
    return G


def forcing(res, seed):
    """What gate switching does to non-watchers: their social mean under the live gate minus the
    social mean under each link's average open rate."""
    u = np.random.default_rng(7000 + seed).uniform(-1, 1, T)
    frozen = surgery(res, tape=[])
    frozen.rng = np.random.default_rng(7000 + seed)
    X = frozen.run(u)
    G = np.array(frozen.tape)
    A = res.A
    n = len(A)
    P = np.zeros((n, n))
    P[A] = G[WASHOUT:].mean(0)
    P += np.eye(n)
    nw = res.w_in == 0
    f = []
    # x(t) is the state after the gate of step t+1 is taken, so the gate pairs with the previous state.
    for t in range(WASHOUT, T - 1, 5):
        C = np.eye(n)
        M = np.zeros((n, n), bool)
        M[A] = G[t + 1]
        C = C + M
        f.append(((C @ X[t]) / C.sum(1) - (P @ X[t]) / P.sum(1))[nw])
    f = np.array(f)
    m = f[:len(f) // 10 * 10].reshape(-1, 10, f.shape[1]).mean(1)  # 50-step means
    lam = np.clip(np.linalg.eigvalsh(np.cov(m, rowvar=False)), 0, None)[::-1]
    gr = np.nonzero(A)[0]
    open_rate = G[WASHOUT:].mean(0)
    return dict(
        f_var=f.var(0).mean(),
        f_slow=m.var(0).mean(),
        f_slow_top3=lam[:3].sum() / len(lam),
        f_slow_top3_share=lam[:3].sum() / lam.sum(),
        undecided=((open_rate > 0.1) & (open_rate < 0.9))[nw[gr]].mean(),
        flips_nw=np.abs(np.diff(G[WASHOUT:][:, nw[gr]].astype(np.int8), axis=0)).mean(),
    )


def conditions4(soc, seed):
    rng = np.random.default_rng(seed + 4)
    c, s, b = soc[CONTROL], soc[SALIENCE], soc[BOTH]
    nw = c.w_in == 0
    out = {"control": c, "salience": s, "both": b}
    for label, res in (("control", c), ("salience", s), ("both", b)):
        G = record_gate(res, seed)
        rows, cols = np.nonzero(res.A)
        out[f"{label}, all gates shifted together"] = surgery(res, script=np.roll(G, T // 2, axis=0))
        out[f"{label}, shifted by follower"] = surgery(res, script=shifted(G, rows, rng))
        out[f"{label}, shifted by followed account"] = surgery(res, script=shifted(G, cols, rng))
        out[f"{label}, shifted by link"] = surgery(res, script=shifted(G, np.arange(len(rows)), rng))
    _, mean_s = warm(s, seed)
    _, mean_b = warm(b, seed)
    out["salience, non-watcher links matched"] = swap_links(s, rng, nw, mean_s, -1)
    out["salience, non-watcher links unmatched"] = swap_links(s, rng, nw, mean_s, +1)
    out["both, non-watcher links unmatched"] = swap_links(b, rng, nw, mean_b, +1)
    for eps in (0.2, 0.4, 0.5):
        frozen = copy.deepcopy(s)
        frozen.p = replace(frozen.p, eps=eps)
        out[f"salience, eps {eps}"] = frozen
    return out


FORCED = ("control", "salience", "both", "salience, non-watcher links matched", "salience, non-watcher links unmatched",
          "both, non-watcher links unmatched", "salience, eps 0.2", "salience, eps 0.4", "salience, eps 0.5")


def conditions5(soc, seed):
    c, s, b = soc[CONTROL], soc[SALIENCE], soc[BOTH]
    informed = c.w_in != 0
    everyone = np.ones(len(informed), bool)
    out = {"control": c, "salience": s, "both": b}
    for label, res in (("control", c), ("salience", s), ("both", b)):
        _, mean = warm(res, seed)
        for name, rows, cols in (("followers anchored", everyone, None), ("followed anchored", None, everyone),
                                 ("non-watcher followers anchored", ~informed, None),
                                 ("exposed followers anchored", informed, None),
                                 ("followed exposed anchored", None, informed),
                                 ("followed non-watchers anchored", None, ~informed),
                                 ("both sides anchored", everyone, everyone)):
            out[f"{label}, {name}"] = surgery(res, mean=mean, anchor_rows=rows, anchor_cols=cols)
    return out


def consistency(args):
    """Does the same news give the same state? Noise-free twins from different starts, and the gate's pace."""
    seed, soc = args
    out = {}
    u = np.random.default_rng(8000 + seed).uniform(-1, 1, 6000)
    for label, i in (("control", CONTROL), ("salience", SALIENCE), ("both", BOTH)):
        base = soc[i]
        quiet = copy.deepcopy(base)
        quiet.p = replace(quiet.p, noise=0.0)
        kick = np.random.default_rng(seed).standard_normal(len(base.x))
        starts = {"near": np.clip(base.x + 1e-3 * kick, -1, 1), "far": base.prior.copy()}
        a = copy.deepcopy(quiet).run(u)[3000:]
        row = {}
        nw = base.w_in == 0
        for name, x0 in starts.items():
            twin = copy.deepcopy(quiet)
            twin.x = x0
            b = twin.run(u)[3000:]
            row[f"gap_{name}"] = np.mean((a - b)[:, nw] ** 2) / a[:, nw].var(0).mean()
        G = record_gate(base, seed).astype(float)[WASHOUT:]
        row["flip_rate"] = np.abs(np.diff(G, axis=0)).mean()
        m = G[:len(G) // 50 * 50].reshape(-1, 50, G.shape[1]).mean(1)
        row["gate_slow"] = m.var(0).sum() / G.var(0).sum()
        out[label] = row
    return out


def kernels(X, u):
    """Linear impulse responses: h[i, k] = cov(x_i(t), u(t-1-k)) / var(u). u is i.i.d."""
    Xc = X - X.mean(0)
    return np.stack([Xc[k + 1:].T @ u[:-(k + 1)] / (len(u) - k - 1) / u.var() for k in range(LAGS)], axis=1)


def describe(res, X, u):
    """Structure and impulse responses, for non-watchers."""
    informed = res.w_in != 0
    nw = ~informed
    Xs = X[WASHOUT::10]
    close = np.abs(Xs[:, None, :] - Xs[:, :, None]) < res.p.eps
    active = close.mean(0)
    A = res.A
    to_ex, to_nw = A[nw][:, informed], A[nw][:, nw]
    w = np.abs(res.w_in)
    sign_sum = np.abs((A[nw] * res.w_in).sum(1))
    h = kernels(X[WASHOUT:], np.asarray(u[WASHOUT:]))[nw]
    power = (h ** 2).sum(1) + 1e-12
    sv = np.linalg.svd(h, compute_uv=False) ** 2
    gate = (A[None] & close)[:, nw]
    counts = gate.sum(2)  # open links per follower, over time
    p_open = gate.mean(0)
    Z = X[WASHOUT:, :][:, nw]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    return dict(
        row_coherence=(counts.var(0).sum() / ((p_open * (1 - p_open)).sum() + 1e-12)),
        kurtosis=((Z ** 4).mean(0) - 3).mean(),
        low_kurtosis=((Z ** 4).mean(0) - 3 < -0.5).mean(),
        links_to_exposed=to_ex.sum() / A[nw].sum(),
        leaf_share=(A[:, nw].sum(0) == 0).mean(),
        nw_followed_by_nw=to_nw.sum(1).mean(),
        active_to_exposed=active[nw][:, informed][to_ex].mean(),
        active_to_nw=active[nw][:, nw][to_nw].mean() if to_nw.any() else np.nan,
        cancel=(sign_sum / ((A[nw] * w).sum(1) + 1e-12)).mean(),
        kernel_power=power.mean(),
        kernel_time=((h ** 2 * np.arange(1, LAGS + 1)).sum(1) / power).mean(),
        kernel_dim=sv.sum() ** 2 / (sv ** 2).sum(),
    )


def score(res, seed):
    u = np.random.default_rng(7000 + seed).uniform(-1, 1, T)
    frozen = copy.deepcopy(res)
    frozen.rng = np.random.default_rng(7000 + seed)
    X = frozen.run(u)
    informed = res.w_in != 0
    return dict(
        whole=memory_capacity(X, u, washout=WASHOUT, alpha="cv")[0],
        whole_fixed=memory_capacity(X, u, washout=WASHOUT)[0],
        nw=memory_capacity(X[:, ~informed], u, washout=WASHOUT, alpha="cv")[0],
        exposed=memory_capacity(X[:, informed], u, washout=WASHOUT, alpha="cv")[0],
        **describe(res, X, u),
    )


def job(args):
    seed, soc, build = args
    out = {}
    for name, res in build(soc, seed).items():
        out[name] = score(res, seed)
        if build is conditions4:
            out[name].update(forcing(res, seed) if name in FORCED
                             else dict.fromkeys(("f_var", "f_slow", "f_slow_top3", "f_slow_top3_share", "undecided", "flips_nw"), np.nan))
    return seed, out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    socs = load(sys.argv[1])
    build, stem = (conditions2, "mechanism2") if "--round2" in sys.argv else (conditions, "mechanism")
    if "--round3" in sys.argv:
        build, stem = conditions3, "mechanism3"
    if "--round4" in sys.argv:
        build, stem = conditions4, "mechanism4"
    if "--round5" in sys.argv:
        build, stem = conditions5, "mechanism5"
    seeds = sorted({s for _, s in socs})
    with ProcessPoolExecutor() as pool:
        results = dict(pool.map(job, [(s, {i: socs[i, s] for i in range(4)}, build) for s in seeds]))
    names = list(results[seeds[0]])
    keys = [k for k in results[seeds[0]][names[0]] if k != "whole_fixed"]
    data = {n: {k: np.array([results[s][n][k] for s in seeds]) for k in keys} for n in names}
    np.savez(ROOT / "results" / f"{stem}.npz", **{f"{n}|{k}": v for n, d in data.items() for k, v in d.items()})

    if build is conditions3:
        with ProcessPoolExecutor() as pool:
            rows = list(pool.map(consistency, [(s, {i: socs[i, s] for i in range(4)}) for s in seeds]))
        print("gap_*: squared twin difference / state variance, non-watchers, noise-free. 2 = unrelated twins.")
        for label in rows[0]:
            print(f"  {label:10s}", {k: float(np.round(np.mean([r[label][k] for r in rows]), 4)) for k in rows[0][label]},
                  "gap_far by seed", np.round(sorted(r[label]["gap_far"] for r in rows), 2)[[0, 10, 19]])
        print()

    rng = np.random.default_rng(0)
    print(f"{'condition':44s} {'whole':>6s} {'nw':>6s} {'exposed':>7s}   whole vs salience [95% CI]    whole vs control")
    for n in names:
        row = f"{n:44s} {data[n]['whole'].mean():6.2f} {data[n]['nw'].mean():6.2f} {data[n]['exposed'].mean():7.2f}"
        if n != "salience":
            m, lo, hi, _, _ = paired_stats(data[n]["whole"] - data["salience"]["whole"], rng, n_perm=2000)
            row += f"   {m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        if n != "control":
            m, lo, hi, _, _ = paired_stats(data[n]["whole"] - data["control"]["whole"], rng, n_perm=2000)
            row += f"   {m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        print(row)
    if build is conditions4:
        print()
        print("across seeds and conditions, r with whole-society memory:")
        pool_keys = ("f_var", "f_slow", "f_slow_top3", "undecided", "flips_nw")
        mc = np.concatenate([data[n]["whole"] for n in FORCED])
        nwmc = np.concatenate([data[n]["nw"] for n in FORCED])
        centred = lambda k: np.concatenate([data[n][k] - data[n][k].mean() for n in FORCED])
        for k in pool_keys:
            v = np.concatenate([data[n][k] for n in FORCED])
            print(f"  {k:14s} pooled {np.corrcoef(v, mc)[0, 1]:+.2f}  nw {np.corrcoef(v, nwmc)[0, 1]:+.2f}  "
                  f"within conditions {np.corrcoef(centred(k), centred('whole'))[0, 1]:+.2f}")
    print()
    print(f"{'condition':44s} " + " ".join(f"{k[:11]:>11s}" for k in keys[3:]))
    for n in names:
        print(f"{n:44s} " + " ".join(f"{np.nanmean(data[n][k]):11.3f}" for k in keys[3:]))


if __name__ == "__main__":
    main()
