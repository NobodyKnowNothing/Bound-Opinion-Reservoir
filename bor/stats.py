"""Paired comparisons across seeds: effect size, bootstrap CI and a sign-flip permutation test."""

import numpy as np


def paired_stats(diff, rng, n_boot=20_000, n_perm=200_000):
    """diff: per-seed differences (treatment - reference). Returns mean, CI low, CI high, d_z, two-sided p."""
    diff = np.asarray(diff, dtype=float)
    mean = diff.mean()
    boot = rng.choice(diff, (n_boot, len(diff))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    null = (rng.choice([-1.0, 1.0], (n_perm, len(diff))) * diff).mean(1)
    p = (np.sum(np.abs(null) >= abs(mean)) + 1) / (n_perm + 1)
    return mean, lo, hi, mean / diff.std(ddof=1), p


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the original order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out
