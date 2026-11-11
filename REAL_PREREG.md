# Pre-registration: the Bluesky / GDELT test (confirmation half)

This file freezes the rules for the real-data test planned in [REAL_DATA.md](REAL_DATA.md).

- Everything below was developed on the **discovery half** of the users.
- It is committed and pushed **before any statistic is computed on the confirmation half**.
- The confirmation half has only been through the shared extraction step, which counts posts
  for every user. No state, decoder, group or test has been built from it.

Data: Failla & Rossetti 2024, *Bluesky social dataset* (Zenodo 14669616), and GDELT 1.0 daily
event files, 17 Feb 2023 – 18 Mar 2024. Both stay in `data/` and are not committed. Only
aggregates are written to `results/`.

## The split

Every user id gets a uniform draw from `numpy.random.default_rng(20261111)`:

- draw below 0.5: discovery half (`real_lib.half_of`);
- otherwise: confirmation half.

## Pipeline (frozen code)

| step | script | output |
|---|---|---|
| posts → per-user-day aggregates | `scripts/real_extract_posts.py` | `data/derived/users.bin`, `userdays.bin` |
| follows and interactions | `scripts/real_extract_edges.py` | `follows.npy`, `interactions.npy` |
| GDELT daily features | `scripts/real_gdelt.py` | `gdelt_features.npz` |
| states, news, ridge, MC | `scripts/real_lib.py` | — |
| predictions 1–4 | `scripts/real_analysis.py` (`CFG` at the top) | `results/real_<half>_p12.npy`, `_p34.npy` |
| text and figure | `scripts/real_report.py` | `results/real_<half>.txt`, `figures/real_<half>.png` |

The confirmation run is exactly:

```
python scripts/real_analysis.py run --half confirm --only 12 --tag _p12
python scripts/real_analysis.py run --half confirm --only 34 --tag _p34
python scripts/real_report.py confirm
```

## Definitions

**Opinion of a post.**
- (`sent_label` − 1) × `sent_score`, which ranges from −1 to +1.
- Only the user's own English posts that carry a sentiment score count. Reposts are excluded.
- REAL_DATA.md asks to leave out posts that mention the event being decoded. Because u(t)
  mixes all events, **every** post that matches a broad news lexicon is dropped. The lexicon
  covers countries, demonyms, heads of state and parties, and event words such as war,
  election, strike and indictment. The remaining posts are "lexicon-free" (`field='free'`).

**User state x_i(t).**
- Raw state: the mean opinion of the user's lexicon-free posts that day, with the last value
  carried forward on days without posts.
- Causal high-pass filter: subtract the trailing 28-day mean (at least 7 prior days needed).
- The filter restarts at **6 Feb 2024**, when Bluesky opened to the public; this is the break.
- Then subtract each user's day-of-week means.
- US holidays and 23 Dec – 2 Jan are masked before filtering and left out of decoding. In
  development, Thanksgiving and the post-Christmas dip created a spurious same-day
  news–state correlation (r ≈ 0.46) through the trailing mean.

**News u(t).**
- 48 daily GDELT features: log event count, 20 CAMEO root-code shares, mean tone, mean
  Goldstein score, and the shares of the 25 leading action countries.
- Processing:
  - 22–23 Mar 2023 are dropped and holidays are masked;
  - the same causal 28-day high-pass and weekday removal as the states;
  - z-scoring;
  - the first **5 principal components**, scaled to unit variance.

**Memory capacity (MC).**
- MC = Σ_{k=1..14} [R²(u_{t−k} ← state_t, u_{t−k+1..t}) − R²(u_{t−k} ← u_{t−k+1..t})], with R²
  averaged over the 5 news components.
- State features: the leading 20 PCs of user-standardized states, with the PCA fitted on the
  training days of each fold.
- Cross-validation: 5 contiguous time blocks with a 14-day gap on each side of the test block.
- Ridge penalty: chosen from logspace(−1, 5, 13) by an inner 3-block blocked CV.
- The first term is fitted **stagewise**:
  1. fit the news-only ridge (the baseline term);
  2. fit a second ridge from the state PCs to its training residuals;
  3. the increment is the out-of-fold R² of the sum minus that of the baseline.
- In development, a single joint ridge on [state PCs, news lags] overfitted the 100
  extra columns (MC ≈ −2 at every shift), so it was replaced.

**Null for MC.**
- The news series is shifted circularly, by every shift from 28 days to T − 28 days in steps
  of 1 (177 shifts for P1).
- One-sided p = (1 + #{null ≥ observed}) / (1 + #null).
- Development used shifts in steps of 7 (30 shifts). That rule could never reach the Holm
  threshold for six tests (smallest possible p = 0.032 > 0.05/6), so it was changed before
  this freeze.

**Populations.**
- English users: at least 20 sentiment-scored posts, and those posts are at least 50% of
  their own posts.
- Active in a window: a lexicon-free post in at least 60% of the window's weeks.
- Watchers: at least 3 own posts linking to a news site (a fixed list of 87 outlet
  domains).
- Non-watchers: no such posts.
- Users with 1–2 news-link posts belong to neither group.

**Windows.**
- P1–P2: **1 Jul 2023 – 18 Mar 2024**. REAL_DATA.md implies the whole GDELT span. Bluesky was
  invite-only and nearly silent before the July 2023 influx, and decoding over the empty
  months collapsed every fold to an intercept-only fit.
- P3–P4: 1 Oct 2023 – 18 Mar 2024, as specified.

## Tests

1. **There is memory.** MC of all active users in the half, against the shifted-news null.
2. **Second-hand users hold less.**
   - Each watcher, taken in random order (seed 7), is greedily matched to the nearest unused
     non-watcher on standardized (log lexicon-free posts, active days).
   - Statistic: MC(watchers) − MC(matched non-watchers).
   - Null: 200 permutations that swap labels within pairs; one-sided (watchers higher).
3. **Middling follows cost memory.**
   - Opinion of an account: its mean lexicon-free opinion over the late window (at least 10
     posts).
   - Distance classes (close / middling / far) are cut at the discovery tertiles of follow
     distance: **0.1189 and 0.2754** (`results/real_cuts.json`; the confirmation run reads
     these, it does not refit them).
   - Sample: non-watchers active in the late window with at least 10 followees that have an
     opinion.
   - Middling share is residualized on log out-degree, log opinionated followees, log posts
     and log active days.
   - Users are split into 5 equal groups by the residual. Statistic: the slope of group MC
     against group rank.
   - Null: 200 random equal splits; one-sided (slope below null).
4. **Middling links switch with the news.**
   - A follow edge i→j is *open* on day t if i replied to, reposted or quoted j in the
     previous 7 days. Openings and closings are counted as switches.
   - Edges: follow edges from late-window active users with at least 1 interaction in the
     late window.
   - Edges are stratified by interaction days (1, 2, 3–4, 5–8, 9–16, 17+).
   - **4a:** the switch rate of middling minus close, and of middling minus far. Each is a
     within-stratum difference weighted by the middling edge count.
   - **4b:** the log ratio of the switch rate on high-news days to low-news days, for
     middling minus the mean of close and far. High and low news days are the top and bottom
     thirds of late-window days by ‖u_t‖.
   - Inference: a 1000-draw cluster bootstrap over source users; one-sided p = share of
     draws ≤ 0.

## Pass rules

- All six one-sided p values (P1, P2, P3, P4a-close, P4a-far, P4b) are Holm-adjusted
  together. Each test passes at adjusted p < 0.05.
- Prediction 1 is supported if P1 passes.
- Predictions 2 and 3 count only if P1 passes. If P1 fails, REAL_DATA.md's rule applies: the
  measure is too blunt, and 2–3 are reported as not interpretable whatever their p values.
- Prediction 4a is supported only if **both** middling-vs-close and middling-vs-far pass.
  Prediction 4b is supported if P4b passes.
- If 1 holds and 2–4 fail, the mechanism does not show up in this society.

## What discovery showed

Full text: `results/real_discovery.txt` (P1 used the old 30-shift null).

| test | discovery result |
|---|---|
| P1 | 39,129 users, 232 days. MC = 0.001; shifted-news null mean 0.004, 95th pct 0.016; p = 0.68. Every per-lag increment is within ±0.0005. |
| P2 | 5,995 matched pairs with near-identical activity. MC 0.003 vs 0.001; p = 0.045 raw, 0.22 after Holm. Not interpretable, since P1 failed. |
| P3 | 37,473 non-watchers, 141 days. Group MC +0.002, +0.009, +0.020, +0.003, +0.005; slope −0.0002; p = 0.46. |
| P4a | 2.57 M live edges from 53,871 users. Within interaction strata, middling − close = −0.018 [−0.024, −0.013] and middling − far = +0.056 [+0.047, +0.065]. Switching falls steadily with opinion distance; there is no middling peak. |
| P4b | The middling news-day excess is −0.008 [−0.012, −0.004]. Middling links do not switch more on high-news days. |

Development checks that did not change the result (all on discovery):
- All own posts instead of lexicon-free posts.
- The daily mean state instead of PCs.
- Decoding from states alone without the news baseline: R² ≈ −0.06 at every lag, the same as
  the null.

An event study around 7 Oct 2023 found:
- Israel/Gaza posts rose from about 30 a day to about 2,000.
- Mean sentiment barely moved (0.156 → 0.135).

The news moves what people post about far more than how positive they sound. The sentiment
proxy is the likely reason prediction 1 fails. This was not used to retune anything.
