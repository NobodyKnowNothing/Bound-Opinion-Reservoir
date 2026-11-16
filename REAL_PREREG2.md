# Pre-registration, round 2: topic-attention states (confirmation half)

This file freezes the rules for round 2 of the real-data test planned in
[REAL_DATA.md](REAL_DATA.md). Round 1 (daily sentiment states, [REAL_PREREG.md](REAL_PREREG.md),
[REAL_RESULTS.md](REAL_RESULTS.md)) found that sentiment does not carry the news, so the memory
predictions stayed untested. Round 2 changes only the user state and the opinion distance; every
other rule of round 1 is kept.

- Everything below was developed on the **discovery half** of the users.
- It is committed and pushed **before any round-2 statistic is computed on the confirmation
  half**.
- The confirmation half has been scored once before, in round 1, with sentiment states and
  sentiment distance only. No attention state, attention profile, attention distance or
  round-2 decoder has been built from it. It is a fresh holdout for the attention measure,
  not for the population, window or news rules, which round 1 already used on it.

Data and split: as in REAL_PREREG.md (Failla & Rossetti 2024, Zenodo 14669616; GDELT 1.0 daily
files; split by `numpy.random.default_rng(20261111)`, draw ≥ 0.5 = confirmation half). Data stay
in `data/`; only aggregates are written to `results/`.

## What changes from round 1

**User state x_i(t) (P1–P3).** `real_lib.attention_states`.
- Seven channels per user and day: the number of the user's own English, sentiment-scored posts
  on each of the 6 fixed topics of `real_extract_posts.TOPICS` (Israel/Gaza, Russia/Ukraine,
  Trump/Republicans, Biden/Democrats, AI, Musk/Twitter/Meta), and on any news-lexicon word
  (`n_sent − n_free`).
- Each count becomes log(1 + count). Silent days are 0 and are **not** carried forward, so any
  persistence comes from the user's own posting.
- Then the round-1 filters, unchanged: holiday masking, the causal 28-day high-pass restarting
  at 6 Feb 2024, and removal of weekday means.
- The state matrix has 7 columns per user. P2 and P3 groups take all 7 columns of each member.
- The news-lexicon leave-out of round 1 is dropped **for the state only**. Attention states
  record *what* people post about, which is the point; the price is that a user repeating a
  headline counts as memory. The baseline term still removes what the news since t−k predicts.

**Opinion distance (P3–P4).** `real_lib.attention_profiles` and `real_analysis.dist`.
- An account's position: the square roots of its share of own English, sentiment-scored posts
  on each of the 7 channels over the late window (at least 10 such posts).
- Distance between accounts: Euclidean distance between positions, a Hellinger-type distance
  between attention mixes.
- Close / middling / far are cut at the **discovery** tertiles of follow-edge distance:
  **0.15135 and 0.25188** (`results/real_cuts_r2.json`; the confirmation run reads these and
  does not refit them).

Everything else is as in REAL_PREREG.md: populations (English users, active in ≥ 60% of the
window's weeks, watchers ≥ 3 news-link posts, non-watchers 0), windows (P1–P2 1 Jul 2023 –
18 Mar 2024; P3–P4 1 Oct 2023 – 18 Mar 2024), news u(t) (5 PCs of 48 GDELT features), MC with
the stagewise news baseline, 20 state PCs, 5 blocked folds with 14-day gaps, ridge penalties by
inner blocked CV, the P2 matching and pair-swap null (200), the P3 residualized middling share,
5 groups and random-split null (200), the P4 interaction-day strata and 1000-draw cluster
bootstrap, and all seeds.

**P1 null, corrected count.** Every circular shift from 28 days to T − 28, step 1, where T is
the full window of 262 days: **207 shifts** (REAL_RESULTS.md caveat; round 1's text said 177).

## Pipeline (frozen code)

| step | script |
|---|---|
| extraction (unchanged) | `scripts/real_extract_posts.py`, `real_extract_edges.py`, `real_gdelt.py` |
| states, profiles, news, ridge, MC | `scripts/real_lib.py` |
| tests | `scripts/real_analysis.py --round 2` (`CFG` + `ROUND2` at the top) |
| report | `scripts/real_report.py <half> 2` |

The confirmation run is exactly:

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/real_analysis.py run --half confirm --round 2 --only 12 --tag _p12 --nproc 4
python scripts/real_analysis.py run --half confirm --round 2 --only 34 --tag _p34 --nproc 4
python scripts/real_analysis.py run --half confirm --round 2 --only 5  --tag _p5
python scripts/real_report.py confirm 2
```

The thread settings only stop four workers from each starting four BLAS threads (about 15×
slower); they do not change the statistics.

## Tests and pass rules

The six tests, their statistics, nulls and one-sided directions are those of REAL_PREREG.md
(P1, P2, P3, P4a-close, P4a-far, P4b), computed with the round-2 state and distance. The six
p values are Holm-adjusted together; a test passes at adjusted p < 0.05.

- Prediction 1 is supported if P1 passes.
- Predictions 2 and 3 count only if P1 passes; otherwise they are reported as not
  interpretable, whatever their p values.
- Prediction 4a is supported only if **both** middling-vs-close and middling-vs-far pass.
  Prediction 4b is supported if P4b passes.
- P4 does not depend on P1: it is measured on links.

**Secondary, outside the Holm family (option 5, key `p4s`).** P4 repeated with positions only
for accounts with at least **50** late-window posts (same users, same discovery cut points).
Noisy attention profiles of quiet accounts sit at larger distances, and the interaction-day
strata do not control for that. The secondary check is read as follows: if 4a or 4b passes but
its sign flips or its 95% interval covers 0 in the secondary run, the primary result is
reported as possibly driven by profile noise.

**Positive control, descriptive (`scripts/real_posctrl.py`, discovery only).** It calibrates how
large memory must be before P1 can see it. It is not rerun on confirmation.

## What discovery showed

Full text: `results/real_discovery_r2.txt`, `results/real_posctrl_r2.txt`.

| test | discovery result |
|---|---|
| P1 | 39,129 users, 231 days. MC = 0.012, positive at every lag but ≤ 0.003 at each; null mean −0.004, 95th pct 0.025 (207 shifts); p = 0.25. |
| P2 | 5,995 matched pairs. MC watchers 0.007, non-watchers 0.011; p = 0.82. |
| P3 | 37,494 non-watchers, 141 days. Group MC +0.020, +0.023, +0.024, −0.020, −0.067; slope −0.022; p = 0.025 raw, 0.075 Holm. Not interpretable: P1 failed. |
| P4a | 2.57 M live edges from 53,878 users. Within strata, middling − close = +0.031 [+0.023, +0.039], middling − far = +0.034 [+0.024, +0.044]; both p = 0.001. Middling links switch most. |
| P4b | Middling news-day excess +0.012 [+0.008, +0.017], p = 0.001. All three classes switch slightly *less* on high-news days (log ratios −0.021, −0.015, −0.033); middling drops least, far drops most. |
| P4 secondary | ≥ 50 posts: 2.22 M edges. +0.042, +0.036, +0.012; all p = 0.001. |
| positive control | Planted per-user exponential news traces (λ 0.8) in 30% of users. Oracle (traces alone) MC 0.31. MC exceeds the null 95th pct only at amplitude 0.5 (trace sd half the column sd; MC 0.040) and 1.0 (0.078); at ≤ 0.3, MC stays at the unplanted 0.012. |

Raw switches per interaction day rise from close (0.96) to middling (1.15) to far (1.19); the
middling peak appears only within interaction strata, as the pre-registered comparison requires.
In round 1 (sentiment distance) the same comparison gave close > middling > far.

## Every variant tried on discovery

Recorded here so that nothing tried is hidden; none of them is a test.

- **States.** 20 user-state PCs (MC ≈ 0.012, kept) and 5 PCs (MC 0.013, p ≈ 0.16 against 21
  shifts). Crowd-level readouts (the user mean per channel; crowd log shares): MC ≤ 0, dropped.
- **Control.** A time-reversed "future news" control was dominated by overfitting (−0.20 to
  −0.29) and is dropped.
- **Distance.** Per-topic stance (topic-conditional sentiment) was too sparse: only 1.7–9 k users
  have five or more posts on any one topic. Replaced by attention profiles.
- **Positive control.** A first version planted a single news direction shared by all users.
  Its oracle MC was −0.004, so it could not calibrate anything, and it was replaced by per-user
  directions before any result of it was read beyond the oracle.
- **Secondary P4.** Added after the discovery P4 result, as a guard; its threshold (50 posts)
  was set before it was run, and its discovery result is reported above.
- **Engineering.** The round-2 smoke failure ("exit code 4") did not recur; peak memory is
  about 3.7 GB. Two container restarts cut the discovery run; each was rerun from scratch and
  P1 reproduced exactly.

## Limits

- Observational, as in round 1. The follow graph is a single end-of-window snapshot.
- Attention is not stance. A user who keeps posting about a topic after it leaves the news
  shows memory here whether or not their view changed.
- Without the lexicon leave-out, repeated headlines can count as memory; the news baseline
  removes only what recent news itself predicts.
- The positive control shows that P1 can only detect memory far stronger than the observed
  0.012: a failed P1 means "no memory this pipeline can see", not "no memory".
