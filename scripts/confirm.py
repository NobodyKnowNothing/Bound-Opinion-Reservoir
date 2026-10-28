"""Confirm the sweep's contrasts on fresh seeds.

The peak locations were read off the exploratory sweep (seeds 0-7), so testing them on those
same seeds is optimistic. Here the contrasts are fixed in advance from that sweep and scored on
24 new seeds (100-123), paired within seed, under the main readout and the validated-ridge one.

    python scripts/confirm.py          # writes results/confirm.txt
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
from bor.readout import memory_capacity  # noqa: E402
from bor.stats import holm, paired_stats  # noqa: E402

T = 10_000
SEEDS = range(100, 124)
# (parameter, value A, value B, label): the test is MC(A) - MC(B), paired by seed.
CONTRASTS = [
    ("outrage", 11.3, 0.0, "outrage 11.3 vs 0 (plain HK)"),
    ("outrage", 11.3, 256.0, "outrage 11.3 vs 256"),
    ("input_gain", 0.4, 0.04, "input gain 0.4 vs 0.04"),
    ("input_gain", 0.4, 10.0, "input gain 0.4 vs 10"),
    ("eps", 0.25, 2.0, "ε 0.25 vs 2 (full mixing)"),
    ("eps", 2.0, 0.01, "ε 2 vs 0.01 (closed-minded)"),
]
PIPELINES = [("main readout (α 1e-4)", dict(washout=500)), ("validated ridge", dict(washout=500, alpha="cv"))]


def score(p):
    u = np.random.default_rng(1000 + p.seed).uniform(-1, 1, T)
    X = OpinionReservoir(p).run(u)
    return [memory_capacity(X, u, **kw)[0] for _, kw in PIPELINES]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    configs = sorted({(name, v) for name, a, b, _ in CONTRASTS for v in (a, b)})
    jobs = [replace(Params(), **{name: v}, seed=s) for name, v in configs for s in SEEDS]
    workers = int(os.environ.get("CONFIRM_WORKERS", os.cpu_count()))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        r = np.array(list(pool.map(score, jobs, chunksize=2))).reshape(len(configs), len(SEEDS), len(PIPELINES))
    mc = {cfg: r[i] for i, cfg in enumerate(configs)}

    rng = np.random.default_rng(0)
    lines = [f"Pre-specified contrasts on {len(SEEDS)} fresh seeds ({SEEDS.start}-{SEEDS.stop - 1}), "
             f"T = {T}, paired by seed. Difference in memory capacity: mean [95% bootstrap CI], d_z, "
             "sign-flip p, Holm-adjusted p over the six contrasts, seeds where A > B.", ""]
    for j, (plabel, _) in enumerate(PIPELINES):
        lines.append(plabel)
        rows = []
        for name, a, b, label in CONTRASTS:
            A, B = mc[(name, a)][:, j], mc[(name, b)][:, j]
            rows.append((label, A.mean(), B.mean(), *paired_stats(A - B, rng), int(np.sum(A > B))))
        adj = holm([row[7] for row in rows])
        for row, pa in zip(rows, adj):
            label, ma, mb, m, lo, hi, dz, p, up = row
            lines.append(f"  {label:30s} {ma:5.2f} vs {mb:5.2f}   diff {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  "
                         f"d_z {dz:+.2f}  p {p:.1g}  Holm {pa:.1g}  {up}/{len(SEEDS)}")
        lines.append("")
    report = "\n".join(lines)
    print(report)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "confirm.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
