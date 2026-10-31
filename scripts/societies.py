"""Run the feedback loop and cache the end-of-loop societies, so they can be cut up and replayed.

    python scripts/societies.py 0 20      # seeds 0-19, writes results/societies_0_20.npz
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

from feedback import EPOCHS, POLICIES  # noqa: E402

from bor import OpinionReservoir, Params  # noqa: E402
from bor.recommender import run_loop  # noqa: E402

NAMES = list(POLICIES)


def job(args):
    i, seed = args
    _, res = run_loop(Params(seed=seed), POLICIES[NAMES[i]], epochs=EPOCHS, probe_every=10 ** 9, return_society=True)
    return i, seed, res.A, res.x


def load(path):
    """{(policy index, seed): society}. Traits come from the seed, so only A and x are stored."""
    saved = np.load(path)
    out = {}
    for key in saved.files:
        i, seed, field = key.split("_")
        if field == "A":
            res = OpinionReservoir(Params(seed=int(seed)))
            res.A = saved[key]
            res.x = saved[f"{i}_{seed}_x"]
            out[int(i), int(seed)] = res
    return out


def main():
    start, stop = int(sys.argv[1]), int(sys.argv[2])
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(job, [(i, s) for i in range(len(NAMES)) for s in range(start, stop)]))
    arrays = {}
    for i, s, A, x in results:
        arrays[f"{i}_{s}_A"], arrays[f"{i}_{s}_x"] = A, x
    np.savez_compressed(ROOT / "results" / f"societies_{start}_{stop}.npz", **arrays)
    print(f"cached {len(results)} societies")


if __name__ == "__main__":
    main()
