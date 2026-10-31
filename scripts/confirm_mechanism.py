"""Confirm on fresh seeds what causes the salience recommender's memory loss.

The explanation came from seeds 0-19 (scripts/mechanism.py): the salience network is a better
reservoir under any gate that ignores opinions, and the live confidence gate turns it into its
own source of slow switching noise. Everything here is scored on seeds 300-323, which were not
used to find it.

Fixed before the run. "Gap" is salience minus control under one gate, paired by seed, whole-society
readout, validated ridge. Each test is one-sided in the stated direction:

     1. static gate: gap > 0 at the same open rates
     2. random flicker: gap > 0 at the same open rates
     3. gating: gap(static) - gap(live) > 0
     4. followers anchored: gap(anchored) - gap(live) > 0
     5. followed accounts anchored: gap(anchored) - gap(live) > 0
     6. persistence: [salience(frames 1) - salience(blocks 200)] - same for control > 0
     7. links of one follower moving together:
        [salience(by link) - salience(all together)] - same for control > 0
     8. matching salience non-watchers' links, degrees fixed, raises memory
     9. unmatching the combined policy's non-watcher links, degrees fixed, lowers memory
    10. gate forcing on non-watchers: salience > control
    11. gate forcing on non-watchers: salience > combined

CONFIRMED only if all eleven have the stated sign with Holm p < 0.05 over the eleven. A single
failure is reported as that claim failing, not explained away. Fixed-ridge versions, non-watcher
readouts and link distances are descriptive only.

    python scripts/confirm_mechanism.py results/societies_300_324.npz
    # writes results/confirm_mechanism.txt and figures/mechanism.png
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feedback import COLORS  # noqa: E402
from mechanism import (BOTH, CONTROL, SALIENCE, block_shuffle, forcing, record_gate, score, shifted, surgery,  # noqa: E402
                       swap_links, warm)
from societies import load  # noqa: E402

from bor.stats import holm, paired_stats  # noqa: E402


def links(res, seed):
    """How far apart followers and followed accounts sit, and how much that distance swings."""
    frozen = surgery(res)
    frozen.rng = np.random.default_rng(seed)
    X = frozen.run(np.random.default_rng(seed).uniform(-1, 1, 3000))[500::5]
    nw = res.w_in == 0
    i, j = np.nonzero(res.A & nw[:, None])
    d = np.abs(X[:, i] - X[:, j])
    return dict(link_distance=d.mean(), link_swing=d.std(0).mean(),
                link_margin=np.abs(d.mean(0) - res.p.eps).mean())


def figure(col):
    def ci(v):
        return 1.96 * v.std(ddof=1) / np.sqrt(len(v))

    def gap(gate):
        suffix = f" {gate}" if gate else ""
        return col(f"salience{suffix}") - col(f"control{suffix}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    steps = [("static", "frozen"), ("flicker", "random\nflicker"), ("by link", "+ link\npersistence"),
             ("together", "+ links move\ntogether"), (None, "+ news\n(live gate)")]
    for name, color in (("control", COLORS[0]), ("salience", COLORS[1])):
        vals = [col(f"{name} {g}" if g else name) for g, _ in steps]
        ax.errorbar(range(len(steps)), [v.mean() for v in vals], yerr=[ci(v) for v in vals], fmt="o-", color=color,
                    capsize=3, lw=2, label=name)
    ax.set_xticks(range(len(steps)), [label for _, label in steps], fontsize=8)
    ax.set_ylabel("memory capacity")
    ax.set_title("Same open rates, gate rebuilt one property at a time", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    rows = [("live gate", gap(None)), ("followers fixed\nat mean opinion", gap("followers anchored")),
            ("followed accounts fixed\nat mean opinion", gap("followed anchored")), ("frozen gate", gap("static"))]
    for y, (_, v) in enumerate(rows):
        ax.errorbar(v.mean(), y, xerr=ci(v), fmt="o", color=COLORS[1], capsize=3)
    ax.axvline(0, color=COLORS[0], lw=1)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("memory, salience minus control")
    ax.set_title("The loss needs the live gate", loc="left", fontsize=10)

    ax = axes[2]
    points = [("control", "control", 8, 6), ("both", "salience + agreement", 8, 6),
              ("salience matched", "salience, followers matched", 8, -11), ("both unmatched", "combined, followers unmatched", -150, -14),
              ("salience", "salience", -46, 6)]
    for name, label, dx, dy in points:
        f, m = col(name, "f_var"), col(name)
        ax.errorbar(f.mean(), m.mean(), xerr=ci(f), yerr=ci(m), fmt="o", color=COLORS[1 if "salience" in name else 0 if name == "control" else 3], capsize=2)
        ax.annotate(label, (f.mean(), m.mean()), xytext=(dx, dy), textcoords="offset points", fontsize=8)
    ax.set_xlabel("gate forcing on non-watchers (variance)")
    ax.set_ylabel("memory capacity")
    ax.set_title("More switching forcing, less memory", loc="left", fontsize=10)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(alpha=0.25, lw=0.6)
    fig.suptitle("What causes the memory loss (24 fresh seeds; bars: 95% CI of the mean)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "mechanism.png", dpi=150)


def job(args):
    seed, soc = args
    rng = np.random.default_rng(seed + 9)
    c, s, b = soc[CONTROL], soc[SALIENCE], soc[BOTH]
    nw = c.w_in == 0
    everyone = np.ones(len(nw), bool)
    conds = {"control": c, "salience": s, "both": b}
    for label, res in (("control", c), ("salience", s)):
        rate, mean = warm(res, seed)
        G = record_gate(res, seed)
        conds[f"{label} static"] = surgery(res, fixed=rng.random(rate.shape) < rate)
        conds[f"{label} flicker"] = surgery(res, flicker=rate, gate_rng=np.random.default_rng(seed))
        conds[f"{label} followers anchored"] = surgery(res, mean=mean, anchor_rows=everyone)
        conds[f"{label} followed anchored"] = surgery(res, mean=mean, anchor_cols=everyone)
        conds[f"{label} frames 1"] = surgery(res, script=block_shuffle(G, 1, rng))
        conds[f"{label} blocks 200"] = surgery(res, script=block_shuffle(G, 200, rng))
        conds[f"{label} by link"] = surgery(res, script=shifted(G, np.arange(G.shape[1]), rng))
        conds[f"{label} together"] = surgery(res, script=np.roll(G, len(G) // 2, axis=0))
    conds["salience matched"] = swap_links(s, rng, nw, warm(s, seed)[1], -1)
    conds["both unmatched"] = swap_links(b, rng, nw, warm(b, seed)[1], +1)
    out = {name: score(res, seed) for name, res in conds.items()}
    for name in ("control", "salience", "both", "salience matched", "both unmatched"):
        out[name].update(forcing(conds[name], seed))
        out[name].update(links(conds[name], seed))
    return seed, out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if "--replot" in sys.argv:
        saved = np.load(ROOT / "results" / "confirm_mechanism.npz")
        figure(lambda name, key="whole": saved[f"{name}|{key}"])
        return
    socs = load(sys.argv[1])
    seeds = sorted({s for _, s in socs})
    with ProcessPoolExecutor() as pool:
        results = dict(pool.map(job, [(s, {i: socs[i, s] for i in range(4)}) for s in seeds]))

    def col(name, key="whole"):
        return np.array([results[s][name][key] for s in seeds])

    def gap(gate=None, key="whole"):
        suffix = f" {gate}" if gate else ""
        return col(f"salience{suffix}", key) - col(f"control{suffix}", key)

    def contrast(a, b, key="whole"):
        return gap(a, key) - gap(b, key)

    def tests(key):
        return [
            ("1 static gate: gap", gap("static", key), +1),
            ("2 random flicker: gap", gap("flicker", key), +1),
            ("3 gating: static gap - live gap", gap("static", key) - gap(None, key), +1),
            ("4 followers anchored - live gap", gap("followers anchored", key) - gap(None, key), +1),
            ("5 followed anchored - live gap", gap("followed anchored", key) - gap(None, key), +1),
            ("6 persistence interaction", contrast("frames 1", "blocks 200", key), +1),
            ("7 links moving together interaction", contrast("by link", "together", key), +1),
            ("8 salience matched - salience", col("salience matched", key) - col("salience", key), +1),
            ("9 both unmatched - both", col("both unmatched", key) - col("both", key), -1),
        ]

    rng = np.random.default_rng(0)
    rule = tests("whole") + [("10 forcing: salience - control", col("salience", "f_var") - col("control", "f_var"), +1),
                             ("11 forcing: salience - both", col("salience", "f_var") - col("both", "f_var"), +1)]
    rows = [paired_stats(v, rng) for _, v, _ in rule]
    # paired_stats is two-sided; halve only when the sign is the stated one.
    one_sided = [p / 2 if m * sign > 0 else 1 - p / 2 for (m, _, _, _, p), (_, _, sign) in zip(rows, rule)]
    adj = holm(one_sided)
    lines = [f"Seeds {seeds[0]}-{seeds[-1]}, end-of-loop societies, T 10000, whole-society readout, validated ridge.",
             "Mean [95% bootstrap CI], d_z, one-sided p, Holm over the eleven, seeds with the stated sign.", ""]
    ok = []
    for (label, v, sign), (m, lo, hi, dz, _), p, pa in zip(rule, rows, one_sided, adj):
        ok.append(m * sign > 0 and pa < 0.05)
        lines.append(f"  {label:38s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  d_z {dz:+.2f}  p {p:.2g}  Holm {pa:.2g}  "
                     f"{int(np.sum(v * sign > 0))}/{len(v)}  {'pass' if ok[-1] else 'FAIL'}")
    lines += ["", f"Rule: {'CONFIRMED' if all(ok) else 'NOT CONFIRMED'} ({sum(ok)} of {len(ok)} pass)", "",
              "Levels (whole / non-watchers / fixed ridge whole):"]
    for name in results[seeds[0]]:
        lines.append(f"  {name:30s} {col(name).mean():.2f}  {col(name, 'nw').mean():.2f}  {col(name, 'whole_fixed').mean():.2f}")
    lines += ["", "Descriptive, fixed ridge versions of tests 1-9:"]
    for label, v, _ in tests("whole_fixed"):
        lines.append(f"  {label:38s} {v.mean():+.3f}  {int(np.sum(v > 0))}/{len(v)} above zero")
    lines += ["", "Descriptive, non-watchers' links:"]
    for name in ("control", "salience", "both", "salience matched", "both unmatched"):
        lines.append(f"  {name:20s} " + "  ".join(f"{k} {col(name, k).mean():.3f}" for k in (
            "f_var", "f_slow", "undecided", "flips_nw", "link_distance", "link_swing", "link_margin")))
    report = "\n".join(lines)
    print(report)
    (ROOT / "results" / "confirm_mechanism.txt").write_text(report, encoding="utf-8")
    np.savez(ROOT / "results" / "confirm_mechanism.npz", seeds=seeds,
             **{f"{name}|{k}": col(name, k) for name in results[seeds[0]] for k in results[seeds[0]][name]})
    figure(col)


if __name__ == "__main__":
    main()
