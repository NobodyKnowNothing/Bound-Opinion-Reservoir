# Real-data test: Bluesky and GDELT

## Why

The simulation makes a specific claim: a feed that links people to accounts near the edge of
what they will accept makes the society's opinions hold *less* of the recent news, because
those links keep switching on and off as the news moves the accounts. People who get the news
second-hand lose the most, and linking people to accounts they already nearly agree with
removes the loss (memory capacity 2.24 → 1.85 under the salience feed on long replays; see
"What causes the loss" in the README). All of this rests on an assumed tolerance rule and
made-up parameters. The test below asks whether the same signatures show up in a real
society.

## Data (in `data/`, not committed)

| model | real stand-in | source |
|---|---|---|
| agents | English-language Bluesky users active throughout the window | Failla & Rossetti 2024, Zenodo 14669616 |
| follow graph | `followers.csv.gz` (snapshot 25 Feb – 2 Mar 2024) | same |
| opinion x_i(t) | each user's daily sentiment score (`sent_score`), later stance on fixed topics | `user_posts.tar.gz` |
| link open at time t | user i replied to, reposted or quoted account j in the past week | `interactions.csv.gz` |
| watchers | users who post news links first-hand; fallback: users whose posts lead GDELT topics | posts |
| news u(t) | the first few principal components of GDELT's daily event counts, tone and Goldstein scores | GDELT 1.0, 17 Feb 2023 – 18 Mar 2024 |

## Measure

The real equivalent of memory capacity is how much of day t−k's news can be decoded from
users' states on day t **beyond what the news since then already predicts**:

  MC = Σ_k [ R²(u_{t−k} ← state_t, u_{t−k+1..t}) − R²(u_{t−k} ← u_{t−k+1..t}) ],  k = 1…14

Use ridge on the leading principal components of the user states, blocked time
cross-validation, and a null from circular shifts of the news series. The baseline term is
there because real news is autocorrelated, unlike the model's random input.

## Predictions

1. **There is memory:** MC is above the shifted-news null.
2. **Second-hand users hold less:** non-watchers' MC is below watchers' MC, with groups
   matched on size and activity.
3. **Edge-of-tolerance links cost memory:** among non-watchers, MC falls as more of a user's
   follows sit at middling opinion distance (neither close nor far), controlling for follow
   count and activity.
4. **Those links switch with the news:** links at middling distance open and close more
   often, and more on high-news days, than close or far links.

If prediction 1 fails, the measure is too blunt for the rest. If 1 holds and 2–4 fail, the
mechanism does not show up in this society.

## Design rules

- Develop on a random half of the users and pre-register the rules (push them before
  scoring), then confirm on the other half, as in the simulation work.
- Run predictions 3–4 only on Oct 2023 – Mar 2024, close to the follow snapshot. Treat
  Bluesky's public opening on 6 Feb 2024 as a break point.
- Leave out posts that mention a given event when decoding it, so the score reflects opinion,
  not people repeating headlines.
- Drop the two flawed GDELT days (22–23 Mar 2023; see `data/gdelt/NOTES.txt`).

## Limits

- This is observational. Predictions 2–4 are correlations, and nothing here randomizes a feed.
  `feed_bookmarks.csv` (who subscribed to which algorithmic feed, and when) allows only a weak
  before/after comparison.
- Sentiment is a crude stand-in for opinion. Bluesky users are unrepresentative. The follow
  graph is a single snapshot taken at the end of the window.
- Post text is real people's writing: keep it local, report only aggregates, and cite the
  dataset paper.
