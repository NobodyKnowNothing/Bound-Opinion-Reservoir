"""Confirm on fresh seeds that the salience recommender's memory loss sits in the non-watchers.

The result came from seeds 0-19 (scripts/dimension.py), so it is retested here on 24 seeds no
feedback run has used (200-223), each with its own news and its own replay signal. Societies run
the full loop under the control and salience policies, freeze, and replay for 10 000 steps.

Fixed before the run. Differences are salience minus control, paired by seed, scored under fixed
ridge (the discovery pipeline) and validated ridge:

    1. whole: the whole-society readout loses memory
    2. non-watchers: the non-watcher readout loses memory
    3. gap: the non-watcher readout loses more than the news-exposed one (non-watcher difference
       minus news-exposed difference)

Holm-corrected over the three within each pipeline. The result is CONFIRMED only if 2 and 3 are
both negative with Holm p < 0.05 under both pipelines. Test 1 is reported but isn't part of the
rule. The news-exposed readout's own difference is descriptive: no-change can't be shown by a
non-significant test.

    python scripts/confirm_nonwatchers.py          # writes results/confirm_nonwatchers.txt
    python scripts/confirm_nonwatchers.py --quick  # tiny smoke test, writes nothing
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import copy  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feedback import EPOCHS, POLICIES  # noqa: E402

from bor import Params  # noqa: E402
from bor.readout import memory_capacity  # noqa: E402
from bor.recommender import run_loop  # noqa: E402
from bor.stats import holm, paired_stats  # noqa: E402

T = 10_000
WASHOUT = 500
SEEDS = range(200, 224)
CONTROL, SALIENCE = list(POLICIES)[:2]
PIPELINES = [("fixed ridge (discovery pipeline)", 1e-4), ("validated ridge", "cv")]


def job(args):
    name, seed, epochs, t = args
    _, res = run_loop(Params(seed=seed), POLICIES[name], epochs=epochs, probe_every=10 ** 9, return_society=True)
    u = np.random.default_rng(7000 + seed).uniform(-1, 1, t)
    frozen = copy.deepcopy(res)
    frozen.rng = np.random.default_rng(7000 + seed)
    X = frozen.run(u)
    informed = res.w_in != 0
    washout = min(WASHOUT, t // 4)
    return name, seed, [[memory_capacity(x, u, washout=washout, alpha=alpha)[0]
                         for x in (X, X[:, ~informed], X[:, informed])] for _, alpha in PIPELINES]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    quick = "--quick" in sys.argv
    epochs, seeds, t = (4, range(200, 204), 800) if quick else (EPOCHS, SEEDS, T)
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(job, [(name, s, epochs, t) for name in (CONTROL, SALIENCE) for s in seeds]))
    mc = {name: np.array([r for n, s, r in sorted(results, key=lambda r: r[1]) if n == name])
          for name in (CONTROL, SALIENCE)}  # (seeds, pipelines, [whole, non-watchers, news-exposed])
    diff = mc[SALIENCE] - mc[CONTROL]

    rng = np.random.default_rng(0)
    lines = [f"Salience vs control on {len(seeds)} fresh seeds ({seeds.start}-{seeds.stop - 1}), {epochs} epochs, "
             f"T = {t}. Mean [95% bootstrap CI], d_z, sign-flip p, Holm over tests 1-3, seeds below zero.", ""]
    passed = []
    for j, (label, _) in enumerate(PIPELINES):
        lines.append(label)
        d = diff[:, j]
        tests = [("1 whole", d[:, 0]), ("2 non-watchers", d[:, 1]), ("3 gap (non-watchers - exposed)", d[:, 1] - d[:, 2])]
        rows = [paired_stats(v, rng) for _, v in tests]
        adj = holm([row[4] for row in rows])
        for (name, v), (m, lo, hi, dz, p), pa in zip(tests, rows, adj):
            lines.append(f"  {name:32s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  d_z {dz:+.2f}  p {p:.2g}  Holm {pa:.2g}  "
                         f"{int(np.sum(v < 0))}/{len(v)}")
        m, lo, hi, dz, p = paired_stats(d[:, 2], rng)
        lines.append(f"  {'news-exposed (descriptive)':32s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  d_z {dz:+.2f}  p {p:.2g}")
        levels = mc[CONTROL][:, j].mean(0), mc[SALIENCE][:, j].mean(0)
        lines.append("  control whole/non-watchers/exposed " + " ".join(f"{v:.2f}" for v in levels[0])
                     + ";  salience " + " ".join(f"{v:.2f}" for v in levels[1]))
        lines.append("")
        passed.append(all(rows[i][0] < 0 and adj[i] < 0.05 for i in (1, 2)))
    lines.append(f"Rule: {'CONFIRMED' if all(passed) else 'NOT CONFIRMED'} (fixed ridge {passed[0]}, validated {passed[1]})")
    report = "\n".join(lines)
    print(report)
    if not quick:
        (ROOT / "results" / "confirm_nonwatchers.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
