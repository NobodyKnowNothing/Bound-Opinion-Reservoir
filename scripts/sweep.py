"""Sweep the confidence bound and the outrage gain; score memory capacity and NARMA-10.

    python scripts/sweep.py            # 1-D sweeps + 2-D map, writes figures/ and results/
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bor import OpinionReservoir, Params  # noqa: E402
from bor.readout import memory_capacity, narma_score  # noqa: E402

T = 3000
SEEDS = range(5)
EPS = np.array([0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.4, 2.0])
GAIN = np.geomspace(0.01, 10, 16)


def score(p):
    u = np.random.default_rng(1000 + p.seed).uniform(-1, 1, T)
    X = OpinionReservoir(p).run(u)
    mc, _ = memory_capacity(X, u)
    tail = X[-500:]
    # How many distinct opinion camps: clusters separated by gaps wider than eps/2 at the end.
    xs = np.sort(tail[-1])
    camps = 1 + int((np.diff(xs) > max(p.eps / 2, 0.02)).sum())
    return mc, narma_score(X, u), camps


def sweep(name, values, base=Params()):
    out = np.zeros((len(values), len(SEEDS), 3))
    for i, v in enumerate(values):
        for j, s in enumerate(SEEDS):
            out[i, j] = score(replace(base, **{name: float(v)}, seed=s))
        print(f"{name}={v:.3g}  MC={out[i, :, 0].mean():.2f}  NARMA={out[i, :, 1].mean():.3f}", flush=True)
    return out


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    eps_res = sweep("eps", EPS)
    gain_res = sweep("gain", GAIN)

    eps_grid = EPS[::2]
    gain_grid = GAIN[::2]
    grid = np.zeros((len(gain_grid), len(eps_grid)))
    for a, g in enumerate(gain_grid):
        for b, e in enumerate(eps_grid):
            grid[a, b] = np.mean([score(Params(eps=e, gain=g, seed=s))[0] for s in range(2)])
        print(f"grid row gain={g:.3g} done", flush=True)

    np.savez(ROOT / "results" / "sweep.npz", eps=EPS, gain=GAIN, eps_res=eps_res, gain_res=gain_res,
             eps_grid=eps_grid, gain_grid=gain_grid, grid=grid)


if __name__ == "__main__":
    main()
