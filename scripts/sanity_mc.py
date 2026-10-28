"""Sanity checks for the memory-capacity pipeline.

Runs the same readout on reference reservoirs of the same size (N = 200):

* a linear reservoir with an orthogonal weight matrix, whose MC should approach the
  theoretical ceiling MC <= N (Jaeger 2002) when the readout is well conditioned;
* a standard tanh echo state network (dense, spectral radius 0.9, input to every node);
* a tanh ESN matched to the society: input to 30% of nodes, same noise level;
* the opinion society at its default and its best settings;
* a chance floor: the society's states scored against an unrelated signal.

Delays go out to 400 (washout 1000, T = 20 000) so long memories are not truncated.

    python scripts/sanity_mc.py        # writes figures/mc_sanity.png
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bor import OpinionReservoir, Params  # noqa: E402
from bor.readout import memory_capacity, participation_ratio  # noqa: E402

N, T, WASHOUT, MAX_DELAY = 200, 20_000, 1000, 400


def esn_states(u, rho, in_scale, nonlin, in_frac=1.0, noise=0.0, orthogonal=False, seed=0):
    rng = np.random.default_rng(seed)
    if orthogonal:
        W = rho * np.linalg.qr(rng.standard_normal((N, N)))[0]
    else:
        W = rng.standard_normal((N, N))
        W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
    w_in = in_scale * rng.uniform(-1, 1, N) * (rng.random(N) < in_frac)
    f = np.tanh if nonlin else (lambda z: z)
    x, X = np.zeros(N), np.empty((len(u), N))
    for t, ut in enumerate(u):
        x = f(W @ x + w_in * ut + noise * rng.standard_normal(N))
        X[t] = x
    return X


def run(case):
    label, kind, kw = case
    u = np.random.default_rng(2024).uniform(-1, 1, T)
    if kind == "society":
        X = OpinionReservoir(Params(**kw)).run(u)
    else:
        X = esn_states(u, **kw)
    target = u
    if label.startswith("chance"):
        target = np.random.default_rng(7).uniform(-1, 1, T)  # unrelated to what drove X
    rows = {}
    for alpha in (1e-4, 1e-10, "cv"):
        mc, curve = memory_capacity(X, target, max_delay=MAX_DELAY, washout=WASHOUT, alpha=alpha)
        rows[str(alpha)] = (mc, curve)
    mc30 = memory_capacity(X, target, max_delay=30, washout=WASHOUT)[0]
    return label, rows, mc30, participation_ratio(X[WASHOUT:])


CASES = [
    ("linear, orthogonal W (ρ=0.99)", "esn", dict(rho=0.99, in_scale=0.1, nonlin=False, orthogonal=True)),
    ("tanh ESN (ρ=0.9, input to all)", "esn", dict(rho=0.9, in_scale=0.1, nonlin=True)),
    ("tanh ESN matched (input to 30%, noise 0.01)", "esn",
     dict(rho=0.9, in_scale=0.3, nonlin=True, in_frac=0.3, noise=0.01)),
    ("society, defaults (ε=0.3, gain 0.3)", "society", dict()),
    ("society, best map cell (ε=2, gain 2.5)", "society", dict(eps=2.0, input_gain=2.5)),
    ("chance floor (society vs unrelated signal)", "society", dict()),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console default can't print ε, ρ, ≤
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(run, CASES))

    print(f"{'reservoir':48s} {'MC α=1e-4':>10s} {'MC α=1e-10':>11s} {'MC α=cv':>8s} {'MC ≤30':>7s} {'PR':>6s}")
    for label, rows, mc30, pr in results:
        mcs = [rows[a][0] for a in ("0.0001", "1e-10", "cv")]
        assert max(mcs) <= min(N, MAX_DELAY) + 1e-9, f"{label}: MC above the ceiling"
        print(f"{label:48s} {mcs[0]:10.2f} {mcs[1]:11.2f} {mcs[2]:8.2f} {mc30:7.2f} {pr:6.1f}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#1f2328", "#2563eb", "#7c3aed", "#c2410c", "#ea580c", "#9ca3af"]
    k = np.arange(1, MAX_DELAY + 1)
    for (label, rows, _, _), c in zip(results, colors):
        mc, curve = rows["cv"]
        ax.plot(k, curve, color=c, lw=1.6, label=f"{label}: MC = {mc:.1f}")
    ax.set_xscale("log")
    ax.set_xlabel("delay k (steps)")
    ax.set_ylabel("r² of u(t−k) from x(t)")
    ax.set_title(f"Memory curves, N = {N}, ridge chosen by validation; ceiling is MC ≤ N", loc="left", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "mc_sanity.png", dpi=150)
    print("wrote figures/mc_sanity.png")


if __name__ == "__main__":
    main()
