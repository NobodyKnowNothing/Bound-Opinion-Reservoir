"""Why does the salience recommender cut memory while raising effective dimension?

Each society from the feedback loop (same seeds, same policies) is frozen after its last epoch
and replayed. Replicas share the news and differ only in noise, so across replicas the state
splits into a news-locked part (the replica mean) and a noise part (what's left).

Three routes, with the rule for each fixed before the run (salience vs control, paired):

    A. Noise. Hubs broadcast noise and the news is drowned out. Supported if the noise-free
       replay recovers at least half of the memory lost.
    B. Fast hubs. The readout targets u(t-1), so it may promote accounts that forget quickly.
       Supported if the top-10% accounts are significantly more stubborn than the control's,
       and the news-driven memory stays low with the noise removed.
    C. Lost non-watchers. Supported if the readout on non-watchers alone loses more memory than
       the readout on news-exposed agents alone.

The probes above fit 200 weights on few steps, so the memory numbers are also rechecked on a long
replay (T_LONG steps, the sweeps' pipeline), with fixed and validated ridge. The loss counts as
real if the validated-ridge CI excludes zero and at least half of the short-probe loss survives.

    python scripts/dimension.py          # writes figures/dimension.png, results/dimension_stats.txt
    python scripts/dimension.py --quick  # tiny smoke test, writes nothing
    python scripts/dimension.py --replot # redo stats and figure from results/dimension.npz
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import copy  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from feedback import COLORS, EPOCHS, POLICIES, SEEDS  # noqa: E402

from bor import Params  # noqa: E402
from bor.readout import memory_capacity, participation_ratio, r2, ridge_fit, ridge_predict, split  # noqa: E402
from bor.recommender import mean_abs_corr, run_loop  # noqa: E402
from bor.stats import paired_stats  # noqa: E402

T = 4000
T_LONG = 10_000
WASHOUT = 200
WASHOUT_LONG = 500
REPLICAS = 8
BLOCK = 50  # steps; noise variance that survives block-averaging is slow noise
SHORT = ["control", "salience", "agreement", "salience + agreement"]
METRICS = [
    ("mc_short", "memory capacity, loop's probe (T 2000)"),
    ("mc_long", "memory capacity, long replay"),
    ("mc_long_cv", "memory capacity, long replay, validated"),
    ("mc_long_fit", "same fit scored on its training block"),
    ("mc_long_informed", "long replay, news-exposed readout only"),
    ("mc_long_uninformed", "long replay, non-watcher readout only"),
    ("mc_long_noise_free", "long replay, noise removed"),
    ("mc", "memory capacity, T 4000"),
    ("mc_noise_free", "T 4000, noise removed"),
    ("mc_informed", "T 4000, news-exposed readout only"),
    ("mc_uninformed", "T 4000, non-watcher readout only"),
    ("news_share", "state variance locked to the news"),
    ("news_share_uninformed", "same, non-watchers only"),
    ("dim_total", "effective dimension"),
    ("dim_news", "effective dimension, news-locked part"),
    ("dim_noise", "effective dimension, noise part"),
    ("noise_slow", "noise variance surviving 50-step means"),
    ("like_news", "non-watchers' mean |corr|, news-locked"),
    ("like_noise", "non-watchers' mean |corr|, noise"),
    ("hub_stubborn", "top-10% accounts: stubbornness"),
    ("hub_gain", "top-10% news-exposed: |susceptibility|"),
]


def fit_score(X, u, washout, max_delay=30, alpha=1e-4):
    """Memory capacity scored on the readout's own training block: the in-sample ceiling."""
    tr, _ = split(X, washout)
    Y = np.stack([np.roll(u, k) for k in range(1, max_delay + 1)], axis=1)
    Yhat = ridge_predict(X[tr], ridge_fit(X[tr], Y[tr], alpha))
    return sum(r2(Y[tr, j], Yhat[:, j]) for j in range(max_delay))


def decompose(res, T=T, replicas=REPLICAS, t_long=T_LONG):
    u = np.random.default_rng(99).uniform(-1, 1, T)
    u_long = np.random.default_rng(199).uniform(-1, 1, t_long)
    informed = res.w_in != 0

    def replay(seed, signal=u, **change):
        frozen = copy.deepcopy(res)
        frozen.p = replace(frozen.p, **change)
        frozen.rng = np.random.default_rng(seed)
        return frozen.run(signal)

    long = replay(199, u_long)
    quiet_long = replay(199, u_long, noise=0.0)
    u_short = np.random.default_rng(99).uniform(-1, 1, 2000)
    long_scores = dict(
        mc_short=memory_capacity(replay(99, u_short), u_short)[0],
        mc_long=memory_capacity(long, u_long, washout=WASHOUT_LONG)[0],
        mc_long_cv=memory_capacity(long, u_long, washout=WASHOUT_LONG, alpha="cv")[0],
        mc_long_fit=fit_score(long, u_long, WASHOUT_LONG),
        mc_long_informed=memory_capacity(long[:, informed], u_long, washout=WASHOUT_LONG)[0],
        mc_long_uninformed=memory_capacity(long[:, ~informed], u_long, washout=WASHOUT_LONG)[0],
        mc_long_noise_free=memory_capacity(quiet_long, u_long, washout=WASHOUT_LONG)[0],
    )

    X = np.stack([replay(99 + r) for r in range(replicas)])
    quiet = replay(99, noise=0.0)
    Xs = X[:, WASHOUT:]
    S = Xs.mean(0)
    N = Xs - S
    noise_var = N.var(1).mean(0) * replicas / (replicas - 1)  # per agent, corrected for the replica mean
    news_var = np.clip(S.var(0) - noise_var / replicas, 0, None)  # the replica mean still holds noise / K
    blocks = N[:, :N.shape[1] // BLOCK * BLOCK].reshape(replicas, -1, BLOCK, N.shape[2]).mean(2)

    indeg = res.A.sum(0)
    hubs = np.argsort(indeg)[-(len(indeg) // 10):]
    exposed_hubs = hubs[informed[hubs]]
    return dict(
        **long_scores,
        mc=np.mean([memory_capacity(x, u, washout=WASHOUT)[0] for x in X]),
        mc_noise_free=memory_capacity(quiet, u, washout=WASHOUT)[0],
        mc_informed=np.mean([memory_capacity(x[:, informed], u, washout=WASHOUT)[0] for x in X]),
        mc_uninformed=np.mean([memory_capacity(x[:, ~informed], u, washout=WASHOUT)[0] for x in X]),
        curve=np.mean([memory_capacity(x, u, washout=WASHOUT)[1] for x in X], axis=0),
        curve_noise_free=memory_capacity(quiet, u, washout=WASHOUT)[1],
        news_share=news_var.sum() / (news_var.sum() + noise_var.sum()),
        news_share_uninformed=news_var[~informed].sum() / (news_var[~informed].sum() + noise_var[~informed].sum()),
        dim_total=participation_ratio(Xs[0]),
        dim_news=participation_ratio(S),
        dim_noise=participation_ratio(N.reshape(-1, N.shape[2])),
        noise_slow=blocks.var(1).sum() / N.var(1).sum(),
        like_news=mean_abs_corr(S[:, ~informed]),
        like_noise=mean_abs_corr(N[0][:, ~informed]),
        hub_stubborn=res.lam[hubs].mean(),
        hub_gain=np.abs(res.w_in[exposed_hubs]).mean() if len(exposed_hubs) else np.nan,
    )


def job(args):
    name, seed, epochs, T, replicas, t_long = args
    history, res = run_loop(Params(seed=seed), POLICIES[name], epochs=epochs, probe_every=10 ** 9,
                            return_society=True)
    return name, seed, dict(decompose(res, T, replicas, t_long), last_probe=history["mc"][-1])


def dots(ax, data, key, title):
    """One row per policy: seed-level dots behind the mean and its 95% CI."""
    rng = np.random.default_rng(0)
    for y, (color, short) in enumerate(zip(COLORS, SHORT)):
        v = data[y][key]
        ax.scatter(v, y + rng.uniform(-0.16, 0.16, len(v)), s=10, color=color, alpha=0.3, lw=0)
        ax.errorbar(v.mean(), y, xerr=1.96 * v.std(ddof=1) / np.sqrt(len(v)), fmt="o", ms=6, color=color, capsize=3)
    ax.set_yticks(range(len(SHORT)), SHORT if ax.get_subplotspec().is_first_col() else [""] * len(SHORT))
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    quick = "--quick" in sys.argv
    keys = [k for k, _ in METRICS] + ["curve", "curve_noise_free", "last_probe"]
    if "--replot" in sys.argv:
        saved = np.load(ROOT / "results" / "dimension.npz")
        data = [{k: saved[f"{i}_{k}"] for k in keys} for i in range(len(POLICIES))]
    else:
        epochs, seeds, t, replicas, t_long = (4, range(2), 600, 3, 800) if quick else (EPOCHS, SEEDS, T, REPLICAS, T_LONG)
        with ProcessPoolExecutor() as pool:
            results = list(pool.map(job, [(name, s, epochs, t, replicas, t_long) for name in POLICIES for s in seeds]))
        data = [{k: np.stack([r[k] for n, s, r in sorted(results, key=lambda r: r[1]) if n == name]) for k in keys}
                for name in POLICIES]
        if not quick:
            np.savez(ROOT / "results" / "dimension.npz",
                     **{f"{i}_{k}": v for i, d in enumerate(data) for k, v in d.items()})

    rng = np.random.default_rng(0)
    lines = [f"End-of-loop societies replayed: {REPLICAS} noise replicas, {T} steps, paired seeds.",
             "Difference vs control: mean [95% bootstrap CI], d_z, sign-flip p.", ""]
    for i, short in enumerate(SHORT):
        lines.append(short)
        for k, label in METRICS:
            row = f"  {label:42s} level {data[i][k].mean():7.3f}"
            if i:
                m, lo, hi, dz, p = paired_stats(data[i][k] - data[0][k], rng)
                row += f"   diff {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  d_z {dz:+.2f}  p {p:.2g}"
            lines.append(row)
    lost = data[0]["mc"] - data[1]["mc"]
    lost_quiet = data[0]["mc_noise_free"] - data[1]["mc_noise_free"]
    lines += ["", "Decision rules, salience vs control:",
              f"  A. noise: memory lost {lost.mean():.3f}, still lost with noise removed {lost_quiet.mean():.3f}, "
              f"recovered {1 - lost_quiet.mean() / lost.mean():.0%} (supported at 50%)",
              f"  B. fast hubs: stubbornness {data[1]['hub_stubborn'].mean():.3f} vs {data[0]['hub_stubborn'].mean():.3f}",
              f"  C. non-watchers: readout loses {(data[0]['mc_uninformed'] - data[1]['mc_uninformed']).mean():.3f}; "
              f"news-exposed readout loses {(data[0]['mc_informed'] - data[1]['mc_informed']).mean():.3f}"]
    short = (data[0]["mc_short"] - data[1]["mc_short"]).mean()
    long = (data[0]["mc_long_cv"] - data[1]["mc_long_cv"]).mean()
    lines.append(f"  Long-replay check: short-probe loss {short:.3f}, validated long-replay loss {long:.3f}, "
                 f"survives {long / short:.1%} (rule: at least 50%)")
    report = "\n".join(lines)
    print(report)
    if not quick:
        (ROOT / "results" / "dimension_stats.txt").write_text(report, encoding="utf-8")

    fig, grid = plt.subplots(2, 3, figsize=(16, 7.6))
    dots(grid[0, 0], data, "mc_short", "Memory capacity, the loop's short probe (T 2000)")
    dots(grid[0, 1], data, "mc_long_cv", f"Memory capacity, long replay (T {T_LONG}, validated ridge)")
    span = np.concatenate([d[k] for d in data for k in ("mc_short", "mc_long_cv")])
    for ax in grid[0, :2]:  # one scale, so the two probes compare by eye
        ax.set_xlim(span.min() - 0.1, span.max() + 0.1)
    ax = grid[0, 2]
    delays = np.arange(1, data[0]["curve"].shape[1] + 1)
    for i, style in ((0, "-"), (1, "-")):
        ax.plot(delays, data[i]["curve"].mean(0), style, color=COLORS[i], lw=2, label=f"{SHORT[i]}, with noise")
        ax.plot(delays, data[i]["curve_noise_free"].mean(0), "--", color=COLORS[i], lw=2,
                label=f"{SHORT[i]}, noise removed")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("delay k (steps)")
    ax.set_ylabel("r² at delay k")
    ax.set_title(f"Which delays are lost (T {T})", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    dots(grid[1, 0], data, "news_share", "Share of state variance locked to the news")
    dots(grid[1, 1], data, "dim_news", "Effective dimension of the news-locked state")
    dots(grid[1, 2], data, "hub_stubborn", "Stubbornness of the top-10% accounts")
    fig.suptitle("Replaying the end-of-loop societies: same news, different noise "
                 f"({len(data[0]['mc'])} paired seeds; dots: seeds, bars: 95% CI of the mean)",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    out = Path(tempfile.gettempdir()) / "dimension_quick.png" if quick else ROOT / "figures" / "dimension.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
