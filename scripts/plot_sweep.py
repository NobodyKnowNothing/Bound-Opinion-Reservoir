"""Turn results/sweep.npz into the central figure: memory vs. confidence bound and vs. outrage gain."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, ACCENT, SECOND = "#1f2328", "#6e7781", "#c2410c", "#2563eb"


def band(ax, x, res, color, label):
    m, s = res.mean(1), res.std(1)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)
    ax.plot(x, m, "o-", color=color, ms=4, lw=1.8, label=label)
    k = int(np.argmax(m))
    ax.axvline(x[k], color=color, ls=":", lw=1)
    return x[k]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.grid(alpha=0.25, lw=0.6)


def main():
    d = np.load(ROOT / "results" / "sweep.npz")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))

    ax = axes[0]
    best = band(ax, d["eps"], d["eps_res"][:, :, 0], ACCENT, "memory capacity")
    ax.set_xscale("log")
    ax.set_xlabel("confidence bound ε  (log)")
    ax.set_ylabel("memory capacity (Σ r²)")
    ax.set_title(f"Open-mindedness (gain 0.3): rises, peaks at ε ≈ {best:.2g}, plateaus",
                 loc="left", color=INK, fontsize=10)
    style(ax)

    ax = axes[1]
    best = band(ax, d["gain"], d["gain_res"][:, :, 0], SECOND, "memory capacity")
    ax.set_xscale("log")
    ax.set_xlabel("outrage gain  (log)")
    ax.set_title(f"Outrage (ε 0.3): inverted U, peak at gain ≈ {best:.2g}", loc="left", color=INK, fontsize=10)
    style(ax)

    ax = axes[2]
    im = ax.imshow(d["grid"], origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks(range(len(d["eps_grid"])), [f"{v:.2g}" for v in d["eps_grid"]])
    ax.set_yticks(range(len(d["gain_grid"])), [f"{v:.2g}" for v in d["gain_grid"]])
    ax.set_xlabel("confidence bound ε")
    ax.set_ylabel("outrage gain")
    ax.set_title("Over both: the best ε moves up with the outrage", loc="left", color=INK, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle("Memory of the news peaks at intermediate outrage; the best confidence bound depends on how loud "
                 "the news is", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "inverted_u.png", dpi=150)

    # Secondary: NARMA-10 error (lower is better) on the same sweeps.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, key, xkey, color, lab in [(axes[0], "eps_res", "eps", ACCENT, "confidence bound ε"),
                                      (axes[1], "gain_res", "gain", SECOND, "outrage gain")]:
        res = d[key][:, :, 1]
        m, s = res.mean(1), res.std(1)
        ax.fill_between(d[xkey], m - s, m + s, color=color, alpha=0.15, lw=0)
        ax.plot(d[xkey], m, "o-", color=color, ms=4)
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        style(ax)
    axes[0].set_ylabel("NARMA-10 NRMSE (lower = better)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "narma.png", dpi=150)
    print("wrote figures/inverted_u.png, figures/narma.png")


if __name__ == "__main__":
    main()
