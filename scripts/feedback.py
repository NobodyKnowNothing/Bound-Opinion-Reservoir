"""Run the recommender feedback loop under four policies and measure where it pushes the society.

Seeds are paired: seed s gives every policy the same starting society and the same news, so
each policy is compared with the control seed by seed. Effects are summarised over the last
WINDOW epochs as paired differences with bootstrap 95% CIs, Cohen's d_z and a sign-flip
permutation p-value.

    python scripts/feedback.py         # ~20 min on 12 cores; writes figures/feedback.png
    python scripts/feedback.py --quick # tiny smoke test, writes nothing
    python scripts/feedback.py --replot # redo stats and figure from results/feedback.npz
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import sys  # noqa: E402
import tempfile  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bor import Params  # noqa: E402
from bor.recommender import Recommender, run_loop  # noqa: E402
from bor.stats import paired_stats  # noqa: E402

EPOCHS = 300
WINDOW = 50       # summary window: the last WINDOW epochs
PROBE_EVERY = 5
SEEDS = range(20)
POLICIES = {
    "random rewiring (control)": Recommender(),
    "salience: show what the readout uses": Recommender(beta_salience=8),
    "agreement: show what you'll like": Recommender(beta_agree=8),
    "salience + agreement": Recommender(beta_salience=8, beta_agree=8),
}
COLORS = ["#6e7781", "#c2410c", "#2563eb", "#7c3aed"]
PANELS = [
    ("mc", "memory capacity (probe)"),
    ("active_links", "follow links within ε (active)"),
    ("indegree_gini", "attention concentration (in-degree Gini)"),
    ("top_informed", "top-10% accounts that are news-exposed"),
]
EXTRA = ["top_salient_informed", "salience_on_informed", "extremity", "spread"]


def job(args):
    name, seed, epochs, probe_every = args
    return name, seed, run_loop(Params(seed=seed), POLICIES[name], epochs=epochs, probe_every=probe_every)


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console default can't print ε, ρ, ≤
    quick = "--quick" in sys.argv
    if quick:
        global EPOCHS, WINDOW, PROBE_EVERY, SEEDS
        EPOCHS, WINDOW, PROBE_EVERY, SEEDS = 6, 3, 2, range(3)
    keys = [k for k, _ in PANELS] + EXTRA
    if "--replot" in sys.argv:
        saved = np.load(ROOT / "results" / "feedback.npz")
        hist = {name: {k: saved[f"{i}_{k}"] for k in keys} for i, name in enumerate(POLICIES)}
    else:
        jobs = [(name, s, EPOCHS, PROBE_EVERY) for name in POLICIES for s in SEEDS]
        with ProcessPoolExecutor() as pool:
            results = list(pool.map(job, jobs))
        hist = {name: {k: np.stack([h[k] for n, s, h in sorted(results, key=lambda r: r[1]) if n == name])
                       for k in keys} for name in POLICIES}  # each (seeds, epochs), rows ordered by seed

    (ROOT / "results").mkdir(exist_ok=True)
    if not quick and "--replot" not in sys.argv:
        np.savez(ROOT / "results" / "feedback.npz",
                 **{f"{i}_{k}": v for i, name in enumerate(POLICIES) for k, v in hist[name].items()})

    # Paired effect sizes over the last WINDOW epochs, policy minus control, seed by seed.
    rng = np.random.default_rng(0)
    control = list(POLICIES)[0]
    summary = {name: {k: np.nanmean(hist[name][k][:, -WINDOW:], axis=1) for k in keys} for name in POLICIES}
    lines = [f"Last {WINDOW} of {EPOCHS} epochs, {len(SEEDS)} paired seeds. "
             "Difference vs control: mean [95% bootstrap CI], d_z, sign-flip p.", ""]
    stats = {}
    for name in POLICIES:
        lines.append(name)
        for k, label in PANELS:
            level = summary[name][k].mean()
            row = f"  {label:42s} level {level:7.3f}"
            if name != control:
                m, lo, hi, dz, p = paired_stats(summary[name][k] - summary[control][k], rng)
                stats[(name, k)] = (m, lo, hi, dz, p)
                row += f"   diff {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  d_z {dz:+.2f}  p {p:.2g}"
            lines.append(row)
        g = hist[name]["indegree_gini"].mean(0)
        lines.append(f"  Gini change over the last {WINDOW} epochs: {g[-1] - g[-WINDOW]:+.3f} (start {g[0]:.3f})")
        lines.append(f"  top-10% by salience that are news-exposed: {summary[name]['top_salient_informed'].mean():.2f}; "
                     f"salience mass on news-exposed agents: {summary[name]['salience_on_informed'].mean():.2f}")
    e0 = {k: np.mean([hist[n][k][:, 0].mean() for n in POLICIES]) for k in ("top_salient_informed",
                                                                            "salience_on_informed")}
    lines += ["", f"Before any rewiring (epoch 0, all runs): news-exposed agents are 30% of the society but hold "
              f"{e0['salience_on_informed']:.0%} of readout salience and {e0['top_salient_informed']:.0%} "
              f"of the top-10% salient accounts."]
    report = "\n".join(lines)
    print(report)
    if not quick:
        (ROOT / "results" / "feedback_stats.txt").write_text(report, encoding="utf-8")

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 0.8]))
    t = np.arange(EPOCHS)
    for ax, (key, label) in zip(axes, PANELS):
        for color, name in zip(COLORS, POLICIES):
            v = hist[name][key]
            ok = ~np.isnan(v[0])
            m = np.nanmean(v[:, ok], 0)
            ci = 1.96 * np.nanstd(v[:, ok], 0, ddof=1) / np.sqrt(len(v))
            ax.fill_between(t[ok], m - ci, m + ci, color=color, alpha=0.15, lw=0)
            ax.plot(t[ok], m, color=color, lw=1.6, label=name)
        ax.axvspan(EPOCHS - WINDOW, EPOCHS, color="#000", alpha=0.04, lw=0)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("epoch (500 steps, then rewire)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(alpha=0.25, lw=0.6)
    axes[3].axhline(e0["top_salient_informed"], color=COLORS[1], ls="--", lw=1)
    axes[3].text(EPOCHS * 0.3, e0["top_salient_informed"] - 0.06, "news-exposed share of top-salient accounts, epoch 0",
                 color=COLORS[1], fontsize=7.5)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")

    ax = axes[4]
    names = list(POLICIES)[1:]
    for y, (name, color) in enumerate(zip(names, COLORS[1:])):
        m, lo, hi, dz, p = stats[(name, "mc")]
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", color=color, capsize=3)
        ax.text(m, y - 0.22, f"{m:+.2f}, d_z {dz:+.2f}, p {p:.1g}", ha="center", fontsize=8, color=color)
    ax.axvline(0, color="#6e7781", lw=1)
    ax.set_yticks(range(len(names)), [n.split(":")[0] for n in names])
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Δ MC vs control (paired)")
    ax.set_title(f"Δ memory, last {WINDOW} epochs", loc="left", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + 0.6 * (xmax - xmin))

    fig.suptitle(f"Feeding the readout back into who sees whom ({len(SEEDS)} paired seeds; "
                 "shaded: 95% CI of the mean)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    out = Path(tempfile.gettempdir()) / "feedback_quick.png" if quick else ROOT / "figures" / "feedback.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
