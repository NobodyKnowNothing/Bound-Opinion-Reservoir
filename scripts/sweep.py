"""Sweep the confidence bound, the input gain and the outrage; score memory capacity and NARMA-10.

Every 1-D sweep point is one 10 000-step run per seed, scored under several readout
pipelines (run length, washout, train/test split, ridge penalty) so we can check the
curves are not an artefact of how the readout was trained.

    python scripts/sweep.py            # writes results/sweep.npz (~5 min on 12 cores)
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bor import OpinionReservoir, Params  # noqa: E402
from bor.readout import memory_capacity, narma_score, participation_ratio  # noqa: E402

T = 10_000
SEEDS = range(8)
EPS = np.array([0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.4, 2.0])
INPUT_GAIN = np.geomspace(0.01, 10, 16)
OUTRAGE = np.concatenate([[0.0], np.geomspace(0.5, 256, 19)])

# Readout pipelines: (label, slice of the run used, MC kwargs). "long" is the one plotted in the main figure.
PIPELINES = [
    ("long", 10_000, dict(washout=500)),
    ("as first published: T=3000, washout 200", 3000, dict(washout=200)),
    ("washout 2000", 10_000, dict(washout=2000)),
    ("train 50%", 10_000, dict(washout=500, train_frac=0.5)),
    ("train 85%", 10_000, dict(washout=500, train_frac=0.85)),
    ("ridge 1e-7", 10_000, dict(washout=500, alpha=1e-7)),
    ("ridge 1e-2", 10_000, dict(washout=500, alpha=1e-2)),
    ("ridge by validation", 10_000, dict(washout=500, alpha="cv")),
]

MAP_T, MAP_SEEDS = 6000, range(3)
MAP_EPS = EPS[::2]
MAP_GAIN = INPUT_GAIN[::2]
MAP_OUTRAGE = np.concatenate([[0.0], np.geomspace(1, 128, 7)])


def signal(seed, T):
    return np.random.default_rng(1000 + seed).uniform(-1, 1, T)


def score_full(p):
    """All pipelines + NARMA + extremity + effective dimension for one run."""
    u = signal(p.seed, T)
    X = OpinionReservoir(p).run(u)
    mcs = [memory_capacity(X[:n], u[:n], **kw)[0] for _, n, kw in PIPELINES]
    return mcs + [narma_score(X, u, washout=500), float(np.abs(X[500:]).mean()), participation_ratio(X[500:])]


def score_map(p):
    u = signal(p.seed, MAP_T)
    X = OpinionReservoir(p).run(u)
    return memory_capacity(X, u, washout=500)[0]


def main():
    base = Params()
    jobs = {name: [replace(base, **{name: float(v)}, seed=s) for v in values for s in SEEDS]
            for name, values in [("eps", EPS), ("input_gain", INPUT_GAIN), ("outrage", OUTRAGE)]}
    maps = {
        "map_gain": [replace(base, eps=float(e), input_gain=float(g), seed=s)
                     for g in MAP_GAIN for e in MAP_EPS for s in MAP_SEEDS],
        "map_outrage": [replace(base, eps=float(e), outrage=float(o), seed=s)
                        for o in MAP_OUTRAGE for e in MAP_EPS for s in MAP_SEEDS],
    }
    out = {}
    with ProcessPoolExecutor() as pool:
        for name, ps in jobs.items():
            r = np.array(list(pool.map(score_full, ps, chunksize=2)))
            out[f"res_{name}"] = r.reshape(-1, len(SEEDS), r.shape[1])  # (values, seeds, metrics)
            m = out[f"res_{name}"][:, :, 0].mean(1)
            print(f"{name}: long-pipeline MC from {m[0]:.2f}, peak {m.max():.2f}, to {m[-1]:.2f}", flush=True)
        for name, ps in maps.items():
            r = np.array(list(pool.map(score_map, ps, chunksize=2)))
            out[name] = r.reshape(-1, len(MAP_EPS), len(MAP_SEEDS)).mean(2)  # (rows, eps)
            print(f"{name} done", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    np.savez(ROOT / "results" / "sweep.npz", eps=EPS, input_gain=INPUT_GAIN, outrage=OUTRAGE,
             pipelines=np.array([lab for lab, _, _ in PIPELINES]),
             map_eps=MAP_EPS, map_gain_rows=MAP_GAIN, map_outrage_rows=MAP_OUTRAGE, **out)


if __name__ == "__main__":
    main()
