"""Run the recommender feedback loop under four policies and plot where it pushes the society.

    python scripts/feedback.py
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bor import Params  # noqa: E402
from bor.recommender import Recommender, run_loop  # noqa: E402

EPOCHS = 40
SEEDS = range(3)
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


def job(args):
    name, seed = args
    return name, seed, run_loop(Params(seed=seed), POLICIES[name], epochs=EPOCHS)


def main():
    jobs = [(name, s) for name in POLICIES for s in SEEDS]
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(job, jobs))
    hist = {name: {k: np.stack([h[k] for n, _, h in results if n == name]) for k, _ in PANELS}
            for name in POLICIES}

    (ROOT / "results").mkdir(exist_ok=True)
    np.savez(ROOT / "results" / "feedback.npz",
             **{f"{i}_{k}": v for i, name in enumerate(POLICIES) for k, v in hist[name].items()})

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    t = np.arange(EPOCHS)
    for ax, (key, label) in zip(axes, PANELS):
        for color, name in zip(COLORS, POLICIES):
            v = hist[name][key]
            ax.fill_between(t, v.mean(0) - v.std(0), v.mean(0) + v.std(0), color=color, alpha=0.12, lw=0)
            ax.plot(t, v.mean(0), color=color, lw=1.8, label=name)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("epoch (500 steps, then rewire)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Feeding the readout back into who sees whom", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "feedback.png", dpi=150)

    for name in POLICIES:
        h = hist[name]
        print(f"{name:40s}" + "  ".join(f"{k}: {h[k][:, :3].mean():.3f}->{h[k][:, -5:].mean():.3f}"
                                         for k, _ in PANELS))


if __name__ == "__main__":
    main()
