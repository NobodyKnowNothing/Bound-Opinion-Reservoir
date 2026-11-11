# Real-data test: results

These are the results of the Bluesky / GDELT test planned in [REAL_DATA.md](REAL_DATA.md).
The rules were frozen in [REAL_PREREG.md](REAL_PREREG.md) and pushed (commit `933706e`) before
the confirmation half was scored.

Data:
- Failla, A. & Rossetti, G. (2024). *Bluesky social dataset*, Zenodo,
  doi:10.5281/zenodo.14669616.
- GDELT 1.0 daily event files, 17 Feb 2023 – 18 Mar 2024.

Only aggregates appear here. No post text leaves `data/`.

Full output: [results/real_confirm.txt](results/real_confirm.txt) and
[figures/real_confirm.png](figures/real_confirm.png). The discovery half is in
[results/real_discovery.txt](results/real_discovery.txt).

## Short answer

**Prediction 1 fails on both halves.** User states built from daily sentiment hold no news
beyond what the news itself already predicts. This sentiment-based memory capacity sits
inside its shifted-news null. Under the rule set in REAL_DATA.md, the measure is therefore
too blunt to test predictions 2 and 3, which are not interpretable.

**Prediction 4 is a clean negative.** It is measured on links, not on memory capacity, so it
does not depend on prediction 1.
- Middling links do switch more than far links.
- But close links switch more than middling ones. Switching falls steadily with opinion
  distance, with no middling peak.
- Middling links do not switch more on high-news days.

The test does not show the simulation's mechanism in this society. The memory part of the
claim stays untested because the opinion proxy cannot carry it.

## Confirmation half (pre-registered)

| prediction | statistic | null / interval | one-sided p | Holm | verdict |
|---|---|---|---|---|---|
| 1 there is memory | MC = 0.001 (39,233 users, 232 days) | shifted news: mean 0.005, 95th pct 0.019 (207 shifts) | 0.61 | 1.00 | not supported |
| 2 second-hand users hold less | MC watchers − non-watchers = −0.001 (6,050 matched pairs) | pair-swap 95th pct +0.003 | 0.72 | 1.00 | not interpretable |
| 3 middling follows cost memory | slope −0.0004 per group (37,470 non-watchers) | random-group 5th pct −0.0020 | 0.41 | 1.00 | not interpretable |
| 4a middling − close switching | −0.014 [−0.020, −0.009] | cluster bootstrap | 1.00 | 1.00 | not supported |
| 4a middling − far switching | +0.060 [+0.051, +0.069] | cluster bootstrap | 0.001 | 0.006 | passes, but 4a needs both |
| 4b middling news-day excess | −0.003 [−0.007, +0.000] | cluster bootstrap | 0.96 | 1.00 | not supported |

The discovery half gave the same picture on every line. The only non-trivial difference is
the P2 raw p: 0.045 on discovery, 0.72 on confirmation. That is consistent with noise around
zero.

| | discovery | confirmation |
|---|---|---|
| P1 MC (null mean) | 0.001 (0.004) | 0.001 (0.005) |
| P2 difference | +0.002 | −0.001 |
| P3 slope | −0.0002 | −0.0004 |
| P4a middling − close / − far | −0.018 / +0.056 | −0.014 / +0.060 |
| P4b news-day excess | −0.008 | −0.003 |

## What the numbers say

**There is nothing to decode.**
- The news baseline is strong. Each past day of u(t) is predicted by the days since with
  out-of-fold R² from 0.62 at lag 1 down to 0 by lag 14.
- Adding 20 PCs of 39k users' states raises it by less than 0.0005 at every lag.
- Without the baseline, decoding from states alone scores R² ≈ −0.06. The shifted null scores
  the same.

This is not a power problem with the decoder; the signal is absent from the sentiment.
- On 7 Oct 2023, posts about Israel/Gaza rose about 70-fold.
- Mean sentiment on that day moved by 0.02.

On Bluesky, the news shows up in *what* people post about, which these states are built to
exclude, not in *how positive* their other posts sound.

**Switching falls with distance.**
- Raw switches per interaction day rise from close (1.01) to middling (1.06) to far (1.16).
  This is a composition effect: far links are mostly one-off interactions, and a single
  interaction already counts as an opening and a closing.
- The pre-registered comparison holds interaction days fixed. Within each stratum, close
  links open and close most often, then middling, then far.
- So frequent on-and-off contact goes with agreement, not with sitting at the edge of
  tolerance.
- High-news days change all three classes' switching about equally: log high/low ratio
  −0.025, −0.021 and −0.010.

## Caveats

- **Pre-registration text.** The pre-registration put the P1 null at 177 shifts. The frozen
  rule, "every shift from 28 days to T − 28", is applied by the code to the full 262-day window
  rather than to its 232 usable days, so it ran 207 shifts. The rule did not change; the count
  in the text was wrong. Either count would give a p far from significance.
- **Observational design.** All of this is observational. The follow graph is a single
  snapshot taken at the end of the window. Sentiment is a crude stand-in for opinion.
- **What a stronger test needs.** It would need a state that does move with the news, such as
  stance on fixed topics. It would also need a way to keep such a state from simply repeating
  headlines, which the leave-out rule here forbids. That is a new design, and it would need
  its own pre-registration.

## Reproduce

1. Download the data into `data/bluesky/` and `data/gdelt/`.
2. Run the extraction and the analysis:

```
python -I scripts/real_extract_posts.py data/bluesky/user_posts.tar.gz data/derived
python -I scripts/real_extract_edges.py data/bluesky data/derived
python -I scripts/real_gdelt.py data/gdelt data/derived/gdelt_features.npz
python scripts/real_analysis.py run --half confirm --only 12 --tag _p12
python scripts/real_analysis.py run --half confirm --only 34 --tag _p34
python scripts/real_report.py confirm
```

Timing: extraction takes about 3 hours, and each half's analysis about 1.5 hours on 8 cores.
