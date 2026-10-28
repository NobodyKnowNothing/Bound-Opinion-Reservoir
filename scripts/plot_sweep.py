"""Turn results/sweep.npz into figures: the sweeps, a pipeline-robustness check, and NARMA-10."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED = "#1f2328", "#6e7781"
C_EPS, C_GAIN, C_OUT = "#c2410c", "#2563eb", "#7c3aed"
MC, NARMA, EXTREMITY, PR = 0, 8, 9, 10  # metric columns in res_*: 8 readout pipelines, then these


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.grid(alpha=0.25, lw=0.6)


def mean_ci(res):
    """Mean over seeds and the 95% CI of that mean. res: (values, seeds)."""
    return res.mean(1), 1.96 * res.std(1, ddof=1) / np.sqrt(res.shape[1])


def curve(ax, x, res, color, label=None, peak=True):
    m, ci = mean_ci(res)
    ax.fill_between(x, m - ci, m + ci, color=color, alpha=0.15, lw=0)
    ax.plot(x, m, "o-", color=color, ms=3.5, lw=1.8, label=label)
    if peak:
        ax.axvline(x[np.argmax(m)], color=color, ls=":", lw=1)
    return x[np.argmax(m)]


def outrage_axis(ax, d, res, color):
    """Outrage sweep on a log axis, with outrage = 0 drawn as a reference band."""
    o = d["outrage"]
    m0, ci0 = mean_ci(res[:1])
    ax.axhspan(m0[0] - ci0[0], m0[0] + ci0[0], color=MUTED, alpha=0.15, lw=0)
    ax.axhline(m0[0], color=MUTED, ls="--", lw=1, label="outrage = 0 (plain HK)")
    best = curve(ax, o[1:], res[1:], color)
    ax.set_xscale("log")
    return best


def heatmap(ax, fig, grid, cols, rows, row_label, title):
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks(range(len(cols)), [f"{v:.2g}" for v in cols])
    ax.set_yticks(range(len(rows)), [f"{v:.2g}" for v in rows])
    ax.set_xlabel("confidence bound ε")
    ax.set_ylabel(row_label)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    best = np.unravel_index(np.argmax(grid), grid.shape)
    for r in range(len(rows)):  # mark the best ε in every row
        c = int(np.argmax(grid[r]))
        ax.plot(c, r, "o", ms=4, mfc="none", mec="white" if (r, c) != best else "#22c55e", mew=1.2)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="memory capacity")


def main():
    d = np.load(ROOT / "results" / "sweep.npz")
    pipelines = list(d["pipelines"])
    n_seeds = d["res_eps"].shape[1]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes[0, 0]
    curve(ax, d["eps"], d["res_eps"][:, :, MC], C_EPS, peak=False)
    ax.set_xscale("log")
    ax.set_xlabel("confidence bound ε (log)")
    ax.set_ylabel("memory capacity  Σₖ r²")
    ax.set_title("(a) Confidence bound, input gain 0.3: rises, then plateaus", loc="left", fontsize=10)

    ax = axes[0, 1]
    best = curve(ax, d["input_gain"], d["res_input_gain"][:, :, MC], C_GAIN)
    ax.set_xscale("log")
    ax.set_xlabel("input gain: how hard the news hits (log)")
    ax.set_title(f"(b) Input gain, ε 0.3: peak ≈ {best:.2g}  (input scaling)", loc="left", fontsize=10)

    ax = axes[0, 2]
    best = outrage_axis(ax, d, d["res_outrage"][:, :, MC], C_OUT)
    ax.set_xlabel("outrage: feed weight exp(outrage·|x_j|) on extreme voices (log)")
    ax.set_title(f"(c) Outrage, ε 0.3, input gain 0.3: peak ≈ {best:.2g}", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    for a in axes[0]:
        style(a)

    heatmap(axes[1, 0], fig, d["map_gain"], d["map_eps"], d["map_gain_rows"], "input gain",
            "(d) ε × input gain (circles: best ε per row)")
    heatmap(axes[1, 1], fig, d["map_outrage"], d["map_eps"], d["map_outrage_rows"], "outrage",
            "(e) ε × outrage (circles: best ε per row)")

    ax = axes[1, 2]
    o, res = d["outrage"][1:], d["res_outrage"][1:]
    m, _ = mean_ci(res[:, :, EXTREMITY])
    ax.plot(o, m, "o-", color=C_OUT, ms=3.5, lw=1.8, label="mean |opinion| (extremity)")
    ax.axhline(d["res_outrage"][0, :, EXTREMITY].mean(), color=C_OUT, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("outrage (log)")
    ax.set_ylabel("extremity", color=C_OUT)
    ax2 = ax.twinx()
    m, _ = mean_ci(res[:, :, PR])
    ax2.plot(o, m, "s-", color=INK, ms=3.5, lw=1.4, label="effective dimension")
    ax2.axhline(d["res_outrage"][0, :, PR].mean(), color=INK, ls="--", lw=1)
    ax2.set_ylabel("effective dimension (participation ratio)")
    ax.set_title("(f) What outrage does to the state (dashed: outrage = 0)", loc="left", fontsize=10)
    style(ax)
    ax2.spines["top"].set_visible(False)

    fig.suptitle(f"Memory capacity of the opinion reservoir ({n_seeds} seeds × 10 000 steps per point; "
                 "shaded: 95% CI of the mean; maps: 3 seeds × 6000 steps)", x=0.01, ha="left", fontsize=12.5,
                 color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "sweeps.png", dpi=150)

    # Robustness: the same states scored under every readout pipeline.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    cmap = plt.get_cmap("tab10")
    for ax, key, xkey, xlabel in [(axes[0], "res_eps", "eps", "confidence bound ε"),
                                  (axes[1], "res_input_gain", "input_gain", "input gain"),
                                  (axes[2], "res_outrage", "outrage", "outrage (0 omitted)")]:
        x, res = d[xkey], d[key]
        if xkey == "outrage":
            x, res = x[1:], res[1:]
        for j, label in enumerate(pipelines):
            m = res[:, :, j].mean(1)
            ax.plot(x, m, "-", color="#111" if j == 0 else cmap(j), lw=2.4 if j == 0 else 1.2,
                    label=label + ("  (main figure)" if j == 0 else ""))
            ax.plot(x[np.argmax(m)], m.max(), "o", color="#111" if j == 0 else cmap(j), ms=4)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        style(ax)
    axes[0].set_ylabel("memory capacity (mean over seeds)")
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Same reservoir states, eight readout pipelines (dots: each pipeline's peak)",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "pipeline_robustness.png", dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, key, xkey, color, label in [(axes[0], "res_eps", "eps", C_EPS, "confidence bound ε"),
                                        (axes[1], "res_input_gain", "input_gain", C_GAIN, "input gain"),
                                        (axes[2], "res_outrage", "outrage", C_OUT, "outrage (0 omitted)")]:
        x, res = d[xkey], d[key][:, :, NARMA]
        if xkey == "outrage":
            x, res = x[1:], res[1:]
        m, ci = mean_ci(res)
        ax.fill_between(x, m - ci, m + ci, color=color, alpha=0.15, lw=0)
        ax.plot(x, m, "o-", color=color, ms=3.5)
        ax.set_xscale("log")
        ax.set_xlabel(label)
        style(ax)
    axes[0].set_ylabel("NARMA-10 NRMSE (lower = better)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "narma.png", dpi=150)

    sys.stdout.reconfigure(encoding="utf-8")
    for key, xkey in [("res_eps", "eps"), ("res_input_gain", "input_gain"), ("res_outrage", "outrage")]:
        peaks = [d[xkey][np.argmax(d[key][:, :, j].mean(1))] for j in range(len(pipelines))]
        print(f"{xkey:11s} peak location per pipeline: " + ", ".join(f"{p:.3g}" for p in peaks))
    print("wrote figures/sweeps.png, figures/pipeline_robustness.png, figures/narma.png")


if __name__ == "__main__":
    main()
