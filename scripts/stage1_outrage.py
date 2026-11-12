"""Stage 1: Outrage x Recommender Topology and Smooth Confidence Gating on cached societies.

Tests how feed amplification of extreme voices (outrage) and confidence-boundary smoothness (tau)
interact with the recommender-rewired topologies (control, salience, agreement, both).

Three mechanistic questions tested on the frozen end-of-loop societies:

1. Topology x Outrage under the live confidence gate:
   In open-loop random graphs (Result 1), moderate outrage (5.6 - 11.3) boosts memory capacity by
   pulling opinions apart. In rewired graphs (Result 2/3), pulling opinions apart across follow
   links triggers confidence-bound switching (f_var) that harms non-watchers. Does moderate outrage
   help or hurt salience- and agreement-rewired societies?

2. Static gate vs. live gate across outrage levels:
   When the confidence gate is frozen at each link's empirical open rate (removing live switching),
   does the salience hub network retain both its structural advantage over control AND the positive
   memory boost from moderate outrage?

3. Smooth logistic confidence gate vs. discontinuous Heaviside step:
   Does replacing the hard cliff 1(|x_i - x_j| < eps) with a smooth logistic gate
   1 / (1 + exp((|x_i - x_j| - eps*) / tau)) (with eps* calibrated on warm-up to preserve mean
   active links) suppress gate-switching forcing and recover the salience network's memory under
   strictly live, opinion-dependent bounded confidence?

Usage:
    python scripts/stage1_outrage.py results/societies_0_20.npz
    python scripts/stage1_outrage.py results/societies_300_324.npz --confirm
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import copy  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from feedback import COLORS  # noqa: E402
from societies import load  # noqa: E402

from bor.readout import memory_capacity, participation_ratio  # noqa: E402
from bor.stats import holm, paired_stats  # noqa: E402

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap


@njit(nogil=True)
def _step_loop_jit(x0, nbrs, lam, prior, w_in, u, noise_mat, mu, input_gain, eps, outrage,
                   mode, fixed_edges, tau, return_gate):
    T_len = len(u)
    n = len(x0)
    k = nbrs.shape[1] - 1
    x = x0.copy()
    X = np.empty((T_len, n), dtype=np.float64)
    G = np.empty((T_len, n, k), dtype=np.float32) if return_gate else np.empty((1, 1, 1), dtype=np.float32)

    for t in range(T_len):
        ut = u[t]
        for i in range(n):
            xi = x[i]
            num = 0.0
            den = 0.0
            for j_idx in range(k):
                nb = nbrs[i, j_idx]
                xnb = x[nb]
                if mode == 1:
                    g = fixed_edges[i, j_idx]
                elif mode == 2:
                    d = abs(xnb - xi)
                    z = (d - eps) / tau
                    if z > 60.0:
                        z = 60.0
                    elif z < -60.0:
                        z = -60.0
                    g = 1.0 / (1.0 + np.exp(z))
                else:
                    g = 1.0 if abs(xnb - xi) < eps else 0.0
                if return_gate:
                    G[t, i, j_idx] = g
                w = g
                if outrage > 0.0 and w > 0.0:
                    w = w * np.exp(outrage * (abs(xnb) - 1.0))
                num += w * xnb
                den += w
            # Self term (always open, g = 1.0)
            w_self = np.exp(outrage * (abs(xi) - 1.0)) if outrage > 0.0 else 1.0
            num += w_self * xi
            den += w_self
            soc = num / den
            x_new = (1.0 - mu) * xi + mu * soc
            x_new = x_new + lam[i] * (prior[i] - x_new)
            x_new = x_new + input_gain * w_in[i] * ut + noise_mat[t, i]
            if x_new > 1.0:
                x_new = 1.0
            elif x_new < -1.0:
                x_new = -1.0
            X[t, i] = x_new
        for i in range(n):
            x[i] = X[t, i]
    return X, G


CONTROL, SALIENCE, AGREEMENT, BOTH = range(4)
POLICY_NAMES = ("control", "salience", "agreement", "both")
OUTRAGE_GRID = (0.0, 2.0, 5.6, 11.3, 32.0, 256.0)
TAU_GRID = (0.01, 0.03, 0.06, 0.12, 0.25)

T = 10_000
WASHOUT = 500
WARM_STEPS = 3000
WARM_WASHOUT = 500


def get_nbrs(A):
    """Return (N, k) followed indices and (N, k+1) indices with self appended last."""
    n = len(A)
    rows, cols = np.nonzero(A)
    followed = cols.reshape(n, -1)
    nbrs = np.column_stack([followed, np.arange(n)])
    return followed, nbrs


def fast_mc_both(X, Y_tr, Y_te, washout=WASHOUT, train_frac=0.7):
    """Return (mc_cv, mc_fixed_1e-4) sharing a single Gram matrix computation."""
    T_rem = len(X) - washout
    cut = washout + int(train_frac * T_rem)
    X_tr, X_te = X[washout:cut], X[cut:]
    Xb_tr = np.hstack([X_tr, np.ones((len(X_tr), 1))])
    Xb_te = np.hstack([X_te, np.ones((len(X_te), 1))])

    vcut = int(len(Xb_tr) * 0.8)
    Xb_sub, Y_sub = Xb_tr[:vcut], Y_tr[:vcut]
    Xb_val, Y_val = Xb_tr[vcut:], Y_tr[vcut:]
    G_sub = Xb_sub.T @ Xb_sub
    rhs_sub = Xb_sub.T @ Y_sub
    n_sub = len(Xb_sub)
    best_a, best_err = 1e-4, np.inf
    diag_idx = np.diag_indices_from(G_sub)
    for a in np.logspace(-8, 0, 9):
        G_a = G_sub.copy()
        G_a[diag_idx] += a * n_sub
        W_a = np.linalg.solve(G_a, rhs_sub)
        err = np.mean((Xb_val @ W_a - Y_val) ** 2)
        if err < best_err:
            best_err = err
            best_a = float(a)

    G_full = Xb_tr.T @ Xb_tr
    rhs_full = Xb_tr.T @ Y_tr
    n_full = len(Xb_tr)
    diag_full = np.diag_indices_from(G_full)

    def score_alpha(a):
        G_a = G_full.copy()
        G_a[diag_full] += a * n_full
        W = np.linalg.solve(G_a, rhs_full)
        Yhat = Xb_te @ W
        # Vectorized squared correlation across all 30 delays
        ym = Y_te - Y_te.mean(0)
        yhm = Yhat - Yhat.mean(0)
        num = (ym * yhm).sum(0) ** 2
        den = (ym ** 2).sum(0) * (yhm ** 2).sum(0)
        valid = den > 1e-24
        return float(np.sum(np.where(valid, num / np.where(valid, den, 1.0), 0.0)))

    mc_cv = score_alpha(best_a)
    mc_fix = mc_cv if best_a == 1e-4 else score_alpha(1e-4)
    return mc_cv, mc_fix


def run_fast(res, u, seed, fixed_edges=None, tau=None, eps_star=None, return_gate=False):
    """Bit-exact fast simulation using neighbor indexing (N, k+1), supporting outrage, fixed gates,
    and smooth logistic gates."""
    p = res.p
    n = p.n_agents
    _, nbrs = get_nbrs(res.A)
    k = nbrs.shape[1] - 1
    x = res.x.copy()
    lam, prior, w_in = res.lam, res.prior, res.w_in
    rng = np.random.default_rng(seed)
    noise_mat = p.noise * rng.standard_normal((len(u), n))
    eps = float(p.eps if eps_star is None else eps_star)

    if HAS_NUMBA:
        mode = 1 if fixed_edges is not None else (2 if tau is not None else 0)
        fe = np.asarray(fixed_edges, dtype=np.float64) if fixed_edges is not None else np.empty((1, 1), dtype=np.float64)
        tau_val = float(tau) if tau is not None else 1.0
        X, G = _step_loop_jit(
            x.astype(np.float64), nbrs.astype(np.int64), lam.astype(np.float64),
            prior.astype(np.float64), w_in.astype(np.float64), np.asarray(u, dtype=np.float64),
            noise_mat.astype(np.float64), float(p.mu), float(p.input_gain), eps, float(p.outrage),
            mode, fe, tau_val, bool(return_gate)
        )
        return (X, G) if return_gate else X

    X = np.empty((len(u), n))
    G = np.empty((len(u), n, k), dtype=np.float32) if return_gate else None

    for t, ut in enumerate(u):
        xn = x[nbrs]  # (n, k+1)
        if fixed_edges is not None:
            g = fixed_edges
        elif tau is not None:
            d = np.abs(xn[:, :-1] - x[:, None])
            g = 1.0 / (1.0 + np.exp(np.clip((d - eps) / tau, -60.0, 60.0)))
        else:
            g = (np.abs(xn[:, :-1] - x[:, None]) < eps).astype(float)

        if return_gate:
            G[t] = g

        m = np.empty((n, k + 1))
        m[:, :-1] = g
        m[:, -1] = 1.0
        if p.outrage:
            m *= np.exp(p.outrage * (np.abs(xn) - 1.0))

        social_mean = (m * xn).sum(1) / m.sum(1)
        x = (1.0 - p.mu) * x + p.mu * social_mean
        x = x + lam * (prior - x)
        x = x + p.input_gain * w_in * ut + noise_mat[t]
        x = np.clip(x, -1.0, 1.0)
        X[t] = x

    return (X, G) if return_gate else X


def warm_stats(res, seed):
    """Run a private 3000-step warm-up to get per-link open rates and opinion distance samples."""
    u_warm = np.random.default_rng(seed).uniform(-1, 1, WARM_STEPS)
    X = run_fast(res, u_warm, seed)[WARM_WASHOUT:]
    Xs = X[::5]
    followed, _ = get_nbrs(res.A)
    d = np.abs(Xs[:, followed] - Xs[:, :, None])  # (time, n, k)
    rate = (d < res.p.eps).mean(0)  # (n, k)
    return rate, d


def calibrate_live_eps_star(res, seed, target_rate, tau):
    """Find eps* by 6-step bisection on live 800-step runs so the realized soft-gate open rate matches target_rate."""
    u_cal = np.random.default_rng(seed).uniform(-1, 1, 800)
    lo, hi = 0.05, 0.45
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        _, G_cal = run_fast(res, u_cal, seed, tau=tau, eps_star=mid, return_gate=True)
        if float(G_cal[200:].mean()) < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def evaluate(res, seed, fixed_edges=None, tau=None, eps_star=None):
    """Single-pass evaluation of MC, dimensionality, extremity, link distances, and gate forcing."""
    u = np.random.default_rng(7000 + seed).uniform(-1, 1, T)
    X, G = run_fast(res, u, 7000 + seed, fixed_edges=fixed_edges, tau=tau, eps_star=eps_star, return_gate=True)
    informed = res.w_in != 0
    nw = ~informed

    T_rem = len(X) - WASHOUT
    cut = WASHOUT + int(0.7 * T_rem)
    Y = np.stack([np.roll(u, k) for k in range(1, 31)], axis=1)
    Y_tr, Y_te = Y[WASHOUT:cut], Y[cut:]

    whole, whole_fixed = fast_mc_both(X, Y_tr, Y_te, washout=WASHOUT)
    nw_mc, _ = fast_mc_both(X[:, nw], Y_tr, Y_te, washout=WASHOUT)
    exposed_mc, _ = fast_mc_both(X[:, informed], Y_tr, Y_te, washout=WASHOUT)

    Xw = X[WASHOUT:]
    Gw = G[WASHOUT:]
    open_rate = Gw.mean(0)  # (n, k)

    # Gate-switching forcing on non-watchers (sampled every 5 steps, matching mechanism.py)
    _, nbrs = get_nbrs(res.A)
    ts = np.arange(WASHOUT, T - 1, 5)
    xt = X[ts]  # (Ts, n)
    xnt = xt[:, nbrs]  # (Ts, n, k+1)

    m_live = np.empty_like(xnt)
    m_live[:, :, :-1] = G[ts + 1]
    m_live[:, :, -1] = 1.0

    m_avg = np.empty_like(xnt)
    m_avg[:, :, :-1] = open_rate[None, :, :]
    m_avg[:, :, -1] = 1.0

    if res.p.outrage:
        w_out = np.exp(res.p.outrage * (np.abs(xnt) - 1.0))
        m_live *= w_out
        m_avg *= w_out

    soc_live = (m_live * xnt).sum(2) / m_live.sum(2)
    soc_avg = (m_avg * xnt).sum(2) / m_avg.sum(2)
    f = (soc_live - soc_avg)[:, nw]  # (Ts, n_nw)
    m_blocks = f[:len(f) // 10 * 10].reshape(-1, 10, f.shape[1]).mean(1)  # 50-step block means

    # Link distance & swing on non-watchers (sampled every 5 steps)
    d_nw = np.abs(xnt[:, nw, :-1] - xt[:, nw, None])  # (Ts, n_nw, k)
    open_nw = open_rate[nw]

    return dict(
        whole=float(whole),
        whole_fixed=float(whole_fixed),
        nw=float(nw_mc),
        exposed=float(exposed_mc),
        dim=float(participation_ratio(Xw)),
        extremity=float(np.abs(Xw).mean()),
        active_links=float(open_rate.mean()),
        f_var=float(f.var(0).mean()),
        f_slow=float(m_blocks.var(0).mean()),
        undecided=float(((open_nw > 0.1) & (open_nw < 0.9)).mean()),
        flips_nw=float(np.abs(np.diff(Gw[:, nw], axis=0)).mean()),
        link_distance=float(d_nw.mean()),
        link_swing=float(d_nw.std(0).mean()),
    )


def with_outrage(res, outrage):
    out = copy.deepcopy(res)
    out.p = replace(out.p, outrage=float(outrage))
    return out


def seed_job(args):
    seed, soc = args
    rng = np.random.default_rng(seed + 100)
    out = {}

    # Cache baseline (outrage=0) warm-up stats for each policy
    base_rates = {}
    base_dists = {}
    base_fixed = {}
    for p_idx, pname in enumerate(POLICY_NAMES):
        rate0, d0 = warm_stats(soc[p_idx], seed)
        base_rates[pname] = rate0
        base_dists[pname] = d0
        base_fixed[pname] = (rng.random(rate0.shape) < rate0).astype(float)

    # 1 & 2. Outrage sweep under live gate and static gate for all 4 policies
    for p_idx, pname in enumerate(POLICY_NAMES):
        for o in OUTRAGE_GRID:
            res_o = with_outrage(soc[p_idx], o)
            out[f"{pname}|live|o={o:g}"] = evaluate(res_o, seed)

            if o == 0.0:
                fixed_o = base_fixed[pname]
            else:
                rate_o, _ = warm_stats(res_o, seed)
                fixed_o = (rng.random(rate_o.shape) < rate_o).astype(float)
            out[f"{pname}|static|o={o:g}"] = evaluate(res_o, seed, fixed_edges=fixed_o)

        # Also test outrage=11.3 with the baseline (o=0) static gate so topology & open-rate are held fixed
        res_11 = with_outrage(soc[p_idx], 11.3)
        out[f"{pname}|static0|o=11.3"] = evaluate(res_11, seed, fixed_edges=base_fixed[pname])

    # 3. Smooth logistic confidence gate sweep (tau) at outrage=0 and outrage=11.3
    for p_idx, pname in ((CONTROL, "control"), (SALIENCE, "salience"), (BOTH, "both")):
        target_rate = float(base_rates[pname].mean())
        for tau in TAU_GRID:
            out[f"{pname}|soft|tau={tau:g}|o=0"] = evaluate(soc[p_idx], seed, tau=tau, eps_star=0.3)
            if tau in (0.06, 0.12):
                eps_star = calibrate_live_eps_star(soc[p_idx], seed, target_rate, tau)
                out[f"{pname}|soft_matched|tau={tau:g}|o=0"] = evaluate(soc[p_idx], seed, tau=tau, eps_star=eps_star)

        # Test soft gate at outrage=11.3
        res_11 = with_outrage(soc[p_idx], 11.3)
        for tau in (0.03, 0.06, 0.12):
            out[f"{pname}|soft|tau={tau:g}|o=11.3"] = evaluate(res_11, seed, tau=tau, eps_star=0.3)

    return seed, out


def make_figure(col, seeds, out_path):
    def ci(v):
        return 1.96 * v.std(ddof=1) / np.sqrt(len(v))

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
    x_pos = np.arange(len(OUTRAGE_GRID))
    x_labels = [f"{o:g}" for o in OUTRAGE_GRID]

    # Panel (a): Live gate MC across outrage for all 4 policies
    ax = axes[0, 0]
    for pname, color in zip(POLICY_NAMES, COLORS):
        means = [col(f"{pname}|live|o={o:g}", "whole").mean() for o in OUTRAGE_GRID]
        cis = [ci(col(f"{pname}|live|o={o:g}", "whole")) for o in OUTRAGE_GRID]
        ax.errorbar(x_pos, means, yerr=cis, fmt="o-", color=color, lw=1.8, ms=4, capsize=2.5, label=pname)
    ax.set_xticks(x_pos, x_labels)
    ax.set_xlabel("outrage (feed amplification of |x_j|)")
    ax.set_ylabel("memory capacity (validated ridge)")
    ax.set_title("(a) Live confidence gate: outrage x policy", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    # Panel (b): Static gate MC across outrage for all 4 policies
    ax = axes[0, 1]
    for pname, color in zip(POLICY_NAMES, COLORS):
        means = [col(f"{pname}|static|o={o:g}", "whole").mean() for o in OUTRAGE_GRID]
        cis = [ci(col(f"{pname}|static|o={o:g}", "whole")) for o in OUTRAGE_GRID]
        ax.errorbar(x_pos, means, yerr=cis, fmt="o--", color=color, lw=1.8, ms=4, capsize=2.5, label=pname)
    ax.set_xticks(x_pos, x_labels)
    ax.set_xlabel("outrage (feed amplification of |x_j|)")
    ax.set_ylabel("memory capacity (frozen gate)")
    ax.set_title("(b) Frozen confidence gate: same open rates", loc="left", fontsize=10)

    # Panel (c): Gate-switching forcing (f_var) across outrage
    ax = axes[0, 2]
    for pname, color in zip(POLICY_NAMES, COLORS):
        means = [col(f"{pname}|live|o={o:g}", "f_var").mean() for o in OUTRAGE_GRID]
        cis = [ci(col(f"{pname}|live|o={o:g}", "f_var")) for o in OUTRAGE_GRID]
        ax.errorbar(x_pos, means, yerr=cis, fmt="s-", color=color, lw=1.8, ms=4, capsize=2.5, label=pname)
    ax.set_xticks(x_pos, x_labels)
    ax.set_xlabel("outrage (feed amplification of |x_j|)")
    ax.set_ylabel("gate switching forcing variance (f_var)")
    ax.set_title("(c) Outrage amplifies gate-switching interference", loc="left", fontsize=10)

    # Panel (d): Non-watcher link swing & undecided fraction across outrage
    ax = axes[1, 0]
    for pname, color in zip(POLICY_NAMES, COLORS):
        means = [col(f"{pname}|live|o={o:g}", "link_swing").mean() for o in OUTRAGE_GRID]
        cis = [ci(col(f"{pname}|live|o={o:g}", "link_swing")) for o in OUTRAGE_GRID]
        ax.errorbar(x_pos, means, yerr=cis, fmt="o-", color=color, lw=1.8, ms=4, capsize=2.5, label=pname)
    ax.set_xticks(x_pos, x_labels)
    ax.set_xlabel("outrage")
    ax.set_ylabel("non-watcher link distance swing (std)")
    ax.set_title("(d) Opinion distance swing across links", loc="left", fontsize=10)

    # Panel (e): Smooth logistic confidence gate (tau sweep) at outrage = 0
    ax = axes[1, 1]
    t_pos = np.arange(len(TAU_GRID) + 1)
    t_labels = ["0\n(hard)"] + [f"{t:g}" for t in TAU_GRID]
    for pname, color in (("control", COLORS[0]), ("salience", COLORS[1]), ("both", COLORS[3])):
        conds = [f"{pname}|live|o=0"] + [f"{pname}|soft|tau={t:g}|o=0" for t in TAU_GRID]
        means = [col(c, "whole").mean() for c in conds]
        cis = [ci(col(c, "whole")) for c in conds]
        ax.errorbar(t_pos, means, yerr=cis, fmt="o-", color=color, lw=1.8, ms=4, capsize=2.5, label=pname)
    ax.set_xticks(t_pos, t_labels, fontsize=8)
    ax.set_xlabel("logistic gate width τ (at ε = 0.3)")
    ax.set_ylabel("memory capacity (validated ridge)")
    ax.set_title("(e) Softening the confidence boundary (outrage = 0)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    # Panel (f): Soft gate forcing vs memory capacity
    ax = axes[1, 2]
    for pname, color in (("control", COLORS[0]), ("salience", COLORS[1]), ("both", COLORS[3])):
        conds = [f"{pname}|live|o=0"] + [f"{pname}|soft|tau={t:g}|o=0" for t in TAU_GRID]
        f_means = [col(c, "f_var").mean() for c in conds]
        m_means = [col(c, "whole").mean() for c in conds]
        ax.plot(f_means, m_means, "o-", color=color, lw=1.6, ms=4.5, label=pname)
        ax.annotate("τ=0", (f_means[0], m_means[0]), xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, color=color)
        ax.annotate(f"τ={TAU_GRID[-1]:g}", (f_means[-1], m_means[-1]), xytext=(5, -8),
                    textcoords="offset points", fontsize=7.5, color=color)
    ax.set_xlabel("gate switching forcing variance (f_var)")
    ax.set_ylabel("memory capacity (validated ridge)")
    ax.set_title("(f) Softening τ collapses forcing & restores MC", loc="left", fontsize=10)

    for a in axes.ravel():
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(alpha=0.25, lw=0.6)

    fig.suptitle(f"Stage 1: Outrage × Recommender Topology & Smooth Confidence Gating "
                 f"({len(seeds)} paired seeds; bars: 95% CI)", x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    soc_path = Path(sys.argv[1])
    is_confirm = "--confirm" in sys.argv
    is_replot = "--replot" in sys.argv
    stem = "stage1_confirm" if is_confirm else "stage1_discovery"
    txt_path = ROOT / "results" / f"{stem}.txt"
    npz_path = ROOT / "results" / f"{stem}.npz"
    fig_path = ROOT / "figures" / f"{stem}.png"

    if is_replot:
        saved = np.load(npz_path)
        seeds = list(saved["seeds"])
        cond_names = []
        metric_keys = []
        for k in saved.files:
            if k == "seeds":
                continue
            c, m = k.rsplit("|", 1)
            if c not in cond_names:
                cond_names.append(c)
            if m not in metric_keys:
                metric_keys.append(m)

        def col(cond, key="whole"):
            return saved[f"{cond}|{key}"]
    else:
        socs = load(soc_path)
        seeds = sorted({s for _, s in socs})
        workers = int(os.environ.get("BOR_WORKERS", os.cpu_count() or 2))
        print(f"Running Stage 1 on {soc_path.name} ({len(seeds)} seeds) with {workers} workers...", flush=True)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        if HAS_NUMBA:
            run_fast(socs[0, seeds[0]], np.zeros(10), seeds[0], return_gate=True)
        PoolCls = ThreadPoolExecutor if HAS_NUMBA else ProcessPoolExecutor
        results = {}
        with PoolCls(max_workers=workers) as pool:
            futs = {pool.submit(seed_job, (s, {i: socs[i, s] for i in range(4)})): s for s in seeds}
            for idx, fut in enumerate(as_completed(futs), 1):
                s, out_s = fut.result()
                results[s] = out_s
                print(f"  [{idx}/{len(seeds)}] seed {s} done", flush=True)

        cond_names = list(results[seeds[0]])
        metric_keys = list(results[seeds[0]][cond_names[0]])

        def col(cond, key="whole"):
            return np.array([results[s][cond][key] for s in seeds])

    rng = np.random.default_rng(0)
    lines = [
        f"Stage 1 ({'CONFIRMATION' if is_confirm else 'DISCOVERY'}): seeds {seeds[0]}-{seeds[-1]} "
        f"({len(seeds)} paired seeds), T = {T}, washout = {WASHOUT}, validated ridge.",
        "",
        "=== 1. Outrage x Policy Table (mean whole MC / non-watcher MC / f_var / link_swing) ===",
    ]
    header = f"{'condition':32s} {'whole':>7s} {'nw':>7s} {'exposed':>8s} {'whole_fix':>9s} {'f_var':>8s} {'undec':>7s} {'swing':>7s} {'extrem':>7s} {'dim':>6s}"
    lines.append(header)
    for c in cond_names:
        lines.append(
            f"{c:32s} "
            f"{col(c, 'whole').mean():7.3f} "
            f"{col(c, 'nw').mean():7.3f} "
            f"{col(c, 'exposed').mean():8.3f} "
            f"{col(c, 'whole_fixed').mean():9.3f} "
            f"{col(c, 'f_var').mean():8.4f} "
            f"{col(c, 'undecided').mean():7.3f} "
            f"{col(c, 'link_swing').mean():7.3f} "
            f"{col(c, 'extremity').mean():7.3f} "
            f"{col(c, 'dim').mean():6.2f}"
        )

    # Pre-registered directional tests (fixed from discovery seeds 0-19 before running on 300-323).
    # Each tuple is (label, paired difference vector, stated sign).
    rule = [
        ("1. Smooth gate (tau=0.12) rescues salience: soft - live",
         col("salience|soft|tau=0.12|o=0") - col("salience|live|o=0"), +1),
        ("2. Smooth gate (tau=0.12) flips salience - control > 0",
         col("salience|soft|tau=0.12|o=0") - col("control|soft|tau=0.12|o=0"), +1),
        ("3. Smooth gate matched open-rate (tau=0.12): salience - control > 0",
         col("salience|soft_matched|tau=0.12|o=0") - col("control|soft_matched|tau=0.12|o=0"), +1),
        ("4. Smooth gate (tau=0.12) cuts salience forcing f_var: live - soft",
         col("salience|live|o=0", "f_var") - col("salience|soft|tau=0.12|o=0", "f_var"), +1),
        ("5. Outrage 11.3 boosts control (live): o=11.3 - o=0",
         col("control|live|o=11.3") - col("control|live|o=0"), +1),
        ("6. Outrage 11.3 boosts salience (live): o=11.3 - o=0",
         col("salience|live|o=11.3") - col("salience|live|o=0"), +1),
        ("7. Outrage 11.3 benefit suppressed in Both vs Control (live DiD)",
         (col("control|live|o=11.3") - col("control|live|o=0")) - (col("both|live|o=11.3") - col("both|live|o=0")), +1),
        ("8. Outrage 11.3 erodes static hub lead: gap(static,0) - gap(static,11.3)",
         (col("salience|static|o=0") - col("control|static|o=0")) - (col("salience|static|o=11.3") - col("control|static|o=11.3")), +1),
        ("9. Outrage 11.3 evacuates salience undecided links: o=0 - o=11.3",
         col("salience|live|o=0", "undecided") - col("salience|live|o=11.3", "undecided"), +1),
    ]

    lines += ["", "=== 2. Pre-Registered Directional Tests (mean [95% CI], d_z, one-sided p, Holm p, seeds matching sign) ==="]
    rows = [paired_stats(v, rng) for _, v, _ in rule]
    one_sided = [p / 2 if m * sign > 0 else 1.0 - p / 2 for (m, _, _, _, p), (_, _, sign) in zip(rows, rule)]
    adj = holm(one_sided)
    ok = []
    for (label, v, sign), (m, lo, hi, dz, _), p1, pa in zip(rule, rows, one_sided, adj):
        passed = bool(m * sign > 0 and pa < 0.05)
        ok.append(passed)
        hits = int(np.sum(v * sign > 0))
        lines.append(
            f"  {label:70s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  d_z {dz:+.2f}  p {p1:.2g}  Holm {pa:.2g}  "
            f"{hits}/{len(v)}  {'pass' if passed else 'FAIL'}"
        )
    lines += ["", f"Rule: {'CONFIRMED' if all(ok) else 'NOT CONFIRMED'} ({sum(ok)} of {len(ok)} pass)"]

    report = "\n".join(lines)
    print(report, flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    txt_path.write_text(report, encoding="utf-8")
    if not is_replot:
        np.savez_compressed(
            npz_path,
            seeds=seeds,
            **{f"{c}|{k}": col(c, k) for c in cond_names for k in metric_keys},
        )
    make_figure(col, seeds, fig_path)
    print(f"Wrote {txt_path}, {npz_path}, {fig_path}", flush=True)


if __name__ == "__main__":
    main()
