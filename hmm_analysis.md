# Modelling Bowling "Strategy" with Hidden Markov Models

## The idea

The k-means analysis (`analysis.md`) found each bowler's repertoire of **delivery types** (clusters), based purely on the physical properties of the ball (speed, length, line, swing, seam, angles). That analysis treats every delivery independently — it has no notion of time or sequence. This document uses **exactly the same clusters** (the over/round-the-wicket split, e.g. Morkel's O0 "Short", R2 "Out-swinger", etc., with the same expert-given names) — so a cluster name means the same thing in both documents.

In reality, a bowler doesn't pick each ball at random. Across an over or a spell they tend to settle into a **plan** — e.g. "attack the stumps with the new ball" or "bowl tight and build pressure" — and that plan shapes which delivery types and outcomes show up in a run of consecutive balls. We can't observe the plan directly, but we can observe its consequences: which cluster each ball belongs to, and what happened (dot ball / runs / wicket).

This is exactly the setup a **Hidden Markov Model (HMM)** is designed for ([background reading](https://luisdamiano.github.io/BayesHMM/articles/introduction.html)):

- A small number of **hidden states** represent the bowler's underlying "strategy" at each point in time.
- A **transition matrix** describes how likely the bowler is to stay in the same strategy from ball to ball, or switch to another.
- An **emission distribution** describes what we expect to observe (cluster + outcome) given the current hidden strategy.

Given a sequence of observations, we can fit these three pieces and then run the **Viterbi algorithm** to decode the most likely hidden-strategy sequence underlying each bowler's deliveries.

### Bayesian vs. simple fitting

The BayesHMM article fits this model fully Bayesian (Stan, MCMC, priors on every parameter). That's the rigorous way to do it, but it's a lot of machinery for ~350-560 observations per bowler. In keeping with "nothing fancy", this analysis fits the **same model structure** (categorical HMM: discrete states, discrete observations) using the standard **Baum-Welch / EM algorithm** (`hmmlearn`'s `CategoricalHMM`), which gives maximum-likelihood estimates of the same π (initial probabilities), A (transition matrix) and θ (emission probabilities). The interpretation — filtering, smoothing, Viterbi decoding — is identical; we're just using point estimates instead of posterior distributions.

---

## Data preparation

For each of the four bowlers, every delivery is reduced to a single observed **symbol**:

**symbol = (ball cluster) × (outcome)**

- **Ball cluster**: the per-bowler over/round-the-wicket-split cluster from `analysis.md` (4–6 clusters depending on bowler, e.g. O0/O1/.../R0/R1/...), capturing *what kind of ball it was*. The HMM's "alphabet size" (`n_clusters x 3` outcomes) is therefore the same total cluster count as in `analysis.md`: Morkel 6, Broad 5, Starc 6, Cummins 4.
- **Outcome**: one of three categories capturing *what happened*:
  - **Dot** — no runs conceded, no wicket
  - **Runs** — runs conceded off the bat, no wicket
  - **Wicket** — the bowler took the wicket

Deliveries are ordered **chronologically** (match start date → innings → over → ball), so the HMM sees each bowler's deliveries in the order they were actually bowled, across all 51 matches in the dataset. The 7 tracking-error deliveries (<75 mph) excluded from the clustering are excluded here too.

### Strategies reset each match

A bowler doesn't carry a "strategy" across a gap of weeks or months between Test matches — whatever plan he had going into the last ball of one match has no bearing on the first ball of the next. Originally this analysis concatenated all of a bowler's deliveries into one long sequence and let the HMM model transitions across match boundaries too, which implicitly assumes the strategy *can* carry over from the last ball of match N to the first ball of match N+1.

To fix this, each bowler's deliveries are still placed in one long chronological array (348–385 symbols, alphabet size `n_clusters × 3` = 12–18), but `hmmlearn` is given a `lengths` array marking where each of the bowler's matches starts and ends. This means:

- The **transition matrix** `A` is only ever applied *within* a match — there's no learned transition from the last ball of one match to the first ball of the next.
- The **start probabilities** π are applied at the start of *every* match, not just once at the start of the whole sequence — so π now has a real interpretation: "what state does this bowler tend to begin a match in?"
- The **emission distributions** are still shared and fit across all matches (pooling data for statistical power), but decoding (Viterbi) is run match-by-match.

This is a standard way to handle multiple independent sequences with a single HMM, and it directly answers the question of whether a bowler's "starting strategy" is consistent match-to-match (persistent) or essentially reset/random.

---

## Choosing the number of hidden states

We fit the HMM separately for each bowler with **2, 3, 4, and 5 hidden states** (15 random restarts each, keeping the best-likelihood fit), and compare them with the **Bayesian Information Criterion (BIC)**, which penalises extra parameters — useful here because more states quickly add a lot of emission parameters relative to the amount of data.

![Choosing the number of hidden states](hmm_state_selection.png)

For **all four bowlers, BIC is minimised at 2 states** and rises steadily and substantially through 3, 4 and 5 (e.g. Morkel goes from 1556.8 at 2 states to 1896.2 at 5). Statistically, a two-state model is about as much structure as 350–560 balls per bowler can support — at 5 states, several states are decoded from only 10–20 balls, so they should be read as suggestive rather than well-estimated.

**Despite that, the results below deliberately use 5 hidden states per bowler.** The 2-state model (used in earlier drafts of this analysis) turned out to map almost exactly onto each bowler's **over/round-the-wicket split** — interesting, but it doesn't tell us much about *strategy* beyond "which side of the stumps". The aim of pushing to 5 states is to see whether, underneath that dominant split, there's any finer-grained structure — e.g. different "modes" *within* a bowler's round-the-wicket spells, or a short-lived high-wicket "strike" state that a 2-state model would average away. As the BIC plot shows, this is a **deliberate trade of statistical parsimony for exploratory detail** — treat the small, rarely-occupied states (anything under ~10% occupancy) as a hint to investigate further with more data, not a confirmed finding.

States are relabelled (after fitting) in order of "aggression" (wicket rate ×10 + runs rate) — `State 0` is the most defensive, `State 4` the most aggressive — purely so states can be compared consistently across the figures. This relabelling means `State 0`...`State 4` here are not the "same" states as in the 2-state version of this analysis — the model has been refit from scratch with 5 components.

---

## Results

![Decoded hidden state through time](hmm_timelines.png)

![Emission profiles per hidden state](hmm_emissions.png)

![Delivery types bowled within each decoded state](hmm_cluster_histograms.png)

The timeline plot shows the Viterbi-decoded state for every delivery in chronological order (★ = wicket); the thin grey dotted lines mark **match boundaries** — the HMM resets to its start distribution at each one, and no transition is modelled across them. The emission plot shows, for each state, the model's *fitted* probability of bowling each ball cluster (top row) and each outcome (bottom row). The histogram below it shows the same idea from the *decoded* side: for each bowler, it takes every delivery the Viterbi path actually assigned to State 0 / State 1, and plots what share of those deliveries were each cluster. Cluster codes (O0, O1, R0, ...) are the same ones used in `analysis.md` — see each bowler's table below for what they mean.

### M Morkel

| | State 0 (16.1%, ~11 balls) | State 1 (14.4%, ~3 balls) | State 2 (25.0%, ~7 balls) | State 3 (41.4%, ~6 balls) | State 4 (3.2%, ~1 ball) |
|---|---|---|---|---|---|
| Dominant clusters | R1 Full outside off (59%), R0 In-swinger (22%), R2 Out-swinger (13%) | R0 In-swinger (58%), R2 Out-swinger (38%) | O0 Short (27%), O1 Good length (73%) | R2 Out-swinger (48%), R1 Full outside off (22%), R0 In-swinger (19%), R3 Bouncer (10%) | O1 Good length (57%), O0 Short (21%), R2 Out-swinger (22%) |
| Outcome | Dot 95%, Runs 5%, Wicket 0% | Dot 93%, Runs 7%, Wicket 0% | Dot 79%, Runs 21%, Wicket 0% | Dot 60%, Runs 37%, Wicket 2% | Dot 22%, Runs 60%, **Wicket 18%** |
| Starts a match in this state | 1 / 12 | 6 / 12 | 5 / 12 | 0 / 12 | 0 / 12 |

The over/round-the-wicket split is still the dominant axis — States 0, 1 and 3 are essentially 100% **round-the-wicket** clusters, State 2 is essentially 100% **over-the-wicket** (O0/O1) — but at 5 states the round-the-wicket repertoire now splits into **three sub-modes of increasing "leakiness"**: State 0 (full outside off-heavy, almost nothing scored, Dot 95%), State 1 (in-swinger/out-swinger mix, Dot 93%), and State 3 — by far the most-occupied state overall (41.4%) — where the same out-swinger/in-swinger mix concedes far more freely (Runs 37%) and produces his only "normal" wickets (2%). State 2 (over-the-wicket short/good length) sits in between (Dot 79%). Finally, **State 4 is a tiny (3.2% ≈ 11 balls) mixed Over/Round state with a striking Wicket rate of 18%** — the model's transition matrix shows State 2 occasionally drops into it (13% of the time) before bouncing straight back to State 1. Given it represents only ~11 deliveries, this should be read as "something interesting may be happening here" rather than a confirmed pattern — but it's exactly the kind of short-lived, high-impact state that a 2-state model would average into the background.

### SCJ Broad

| | State 0 (24.9%, ~1 ball) | State 1 (16.6%, ~14 balls) | State 2 (8.6%, ~9 balls) | State 3 (21.3%, ~8 balls) | State 4 (28.6%, ~1 ball) |
|---|---|---|---|---|---|
| Dominant clusters | R0 Back of length, nip-away (61%), R1 Full and straight (28%), R2 Good length, big swing away (10%) | R2 Good length, big swing away (71%), R1 Full and straight (28%) | O1 Good length, angle across (76%), O0 Short ball/bouncer (15%) | R1 Full and straight (73%), R0 Back of length, nip-away (26%) | R0 Back of length, nip-away (52%), R1 Full and straight (35%), R2 Good length, big swing away (12%) |
| Outcome | Dot 79%, Runs 21%, Wicket 0% | Dot 77%, Runs 23%, Wicket 0% | Dot 58%, Runs 42%, Wicket 0% | Dot 86%, Runs 10%, Wicket 3% | Dot 69%, Runs 22%, **Wicket 8%** |
| Starts a match in this state | 0 / 21 | 2 / 21 | 5 / 21 | 5 / 21 | 9 / 21 |

Broad's deliveries are overwhelmingly **round-the-wicket** (States 0, 1, 3, 4 are all 99–100% round-the-wicket clusters); only State 2 (8.6%) is a genuinely over-the-wicket excursion, and it's also his leakiest non-wicket-taking state (Dot 58%, Runs 42%). Within his round-the-wicket bowling, the model now distinguishes a **persistent containment mode** (State 1, ~14-ball dwell, swing-away heavy, no wickets), a **tight full-and-straight mode** (State 3, Dot 86% — his most defensive state — but still carrying a 3% wicket rate), and a pair of **rapidly-alternating modes** (States 0 and 4, both ~1-ball dwell, transitioning into each other almost every ball — see the transition matrix: State 0 → State 4 always, State 4 → State 0 90% of the time). Of that oscillating pair, **State 4 carries by far his highest wicket rate (8%)** despite a very similar cluster mix to State 0 (which has 0% wickets) — both are dominated by the same back-of-length/full-and-straight/swing-away trio, but the model is picking up some other signal (perhaps sequencing or build-up) that distinguishes the two. As with Morkel's State 4, this 8%-wicket state is built from a meaningful chunk of data (28.6% of deliveries) so is more trustworthy than a 1-ball curiosity, but the *mechanism* distinguishing it from its near-identical neighbour (State 0) isn't obvious from the cluster mix alone.

### MA Starc

| | State 0 (18.0%, ~11 balls) | State 1 (39.4%, near-absorbing) | State 2 (18.8%, ~1 ball) | State 3 (18.5%, ~1 ball) | State 4 (5.2%, ~9 balls) |
|---|---|---|---|---|---|
| Dominant clusters | O2 Length, in-swinger (87%) | O1 Length, seam in (39%), O3 Full in-swinger (28%), O4 Bouncer (28%) | O2 Length, in-swinger (34%), O3 Full in-swinger (26%), O1 Length, seam in (21%), O4 Bouncer (19%) | O1 Length, seam in (38%), O2 Length, in-swinger (27%), O3 Full in-swinger (25%), O4 Bouncer (10%) | O0 Length, out-swinger (55%), R0 Round-the-wicket length (40%) |
| Outcome | Dot 79%, Runs 20%, Wicket 2% | Dot 70%, Runs 28%, Wicket 2% | Dot 65%, Runs 31%, Wicket 4% | Dot 69%, Runs 25%, Wicket 5% | Dot 65%, Runs 25%, **Wicket 10%** |
| Starts a match in this state | 6 / 17 | 0 / 17 | 7 / 17 | 1 / 17 | 3 / 17 |

Starc bowls almost entirely **over the wicket** — only State 4 (5.2%) mixes in any round-the-wicket deliveries (40%), so the over/round split barely registers for him at all. State 0 reprises the **tight, repetitive in-swinger-length opening** (O2, 87%, Dot 79%) seen at 2 states. State 1 is again **near-absorbing** (dwell ~142 balls — once Starc settles into this broad seam-in/full-in-swinger/bouncer mix, he stays there for the rest of the spell) and is by far his most-bowled state (39.4%). The new structure at 5 states is States 2 and 3: a pair of **ball-to-ball alternating states** (State 2 → State 3 92% of the time, State 3 → State 2 always) that split what was previously lumped into "the broad mix" into two slightly different blends — both noticeably more wicket-prone (4–5%) than States 0/1 (2%). **State 4 — the same out-swinger/round-the-wicket-length mix flagged in the 2-state version — again stands out with a 10% wicket rate**, the highest of any state for Starc. *Caveat*: as before, `analysis.md` notes O0 (Length, out-swinger) contains a handful of deliveries with an anomalous (physically implausible) positive Drop Angle reading — a Hawk-Eye tracking artefact, not a genuine delivery type — and these sit inside State 4.

### PJ Cummins

| | State 0 (58.7%, ~10 balls) | State 1 (9.2%, ~1 ball) | State 2 (13.4%, ~1 ball) | State 3 (13.4%, ~1 ball) | State 4 (5.3%, ~1 ball) |
|---|---|---|---|---|---|
| Dominant clusters | O1 Angle-in (60%), O0 Good swinger (31%), O2 Bouncer (9%) | O0 Good swinger (62%), O2 Bouncer (38%) | O0 Good swinger (70%), O1 Angle-in (24%) | O0 Good swinger (85%), O2 Bouncer (10%) | O1 Angle-in (90%), O0 Good swinger (10%) |
| Outcome | Dot 78%, Runs 22%, Wicket 0% | Dot 59%, Runs 41%, Wicket 0% | Dot 54%, Runs 44%, Wicket 2% | Dot 51%, Runs 47%, Wicket 2% | Dot 62%, Runs 16%, **Wicket 23%** |
| Starts a match in this state | 13 / 15 | 0 / 15 | 0 / 15 | 0 / 15 | 0 / 15 |

Cummins' over/round split barely matters here too (round-the-wicket is only 3 balls all dataset, so all 5 states are essentially "over the wicket"). **State 0 is by far his default mode** (58.7%, ~10-ball dwell, angle-in heavy, Dot 78%, Wicket 0%) and is how 13/15 matches begin — this matches the "containment" state from the 2-state model. The new structure is a **four-state escalation cycle** the model decodes once it leaves State 0: State 0 → State 4 (10.1% of the time) → State 2 (almost always) → State 3 (always) → back to State 0 (27%) or State 1 (73%) → State 2 (always) → ... Through States 1→2→3, the cluster mix shifts steadily from swinger/bouncer towards almost pure good-swinger, and the Dot rate steadily falls (59% → 54% → 51%) while Runs rises — a gradual "loosening" similar to the old State 1, but now resolved into discrete steps. **State 4 is the standout: just 5.3% of deliveries (≈20 balls), almost entirely the angle-in ball (90%), with a 23% wicket rate** — by far the highest of any state for any bowler in this analysis. It sits right after State 0 in the cycle (State 0 → State 4 10% of the time, State 4 → State 2 99.9% of the time), so it reads as a short, sharp "strike" sub-mode breaking out of the containment phase. With ~20 balls behind it, this is the clearest "5-state-only" finding — but also the one most in need of more data before drawing conclusions.

---

## How does the strategy mix change match-by-match?

With each match treated as its own sequence, this plot is now a direct readout of the model's per-match behaviour rather than an artefact of one long concatenated decode.

![Hidden-state mix per match](hmm_match_states.png)

Each bar is one match (in chronological order, left to right), showing what fraction of that bowler's deliveries in that match were decoded as State 0 (blue), State 1 (orange), State 2 (red), State 3 (purple) or State 4 (green). The grey number above each bar is how many deliveries that bowler bowled at this batter in that match — many matches are single-figure samples, so individual bars should be read cautiously, but the overall pattern across many matches is informative.

- **M Morkel**: most matches are dominated by a mix of the round-the-wicket states (0/blue, 1/orange, 3/purple), with State 2 (over-the-wicket, red) appearing as a smaller recurring component in many matches rather than a separate match-level mode — consistent with **wicket-position choice being a within-match tactical variable** for Morkel, not just a between-match one. The green State 4 (his 18%-wicket state) appears as thin slivers in roughly a third of matches.
- **SCJ Broad**: most matches show a substantial mix of his round-the-wicket states (0/blue, 1/orange, 3/purple, 4/green), with State 2 (over-the-wicket, red) appearing as a sizeable or even dominant component in about a third of matches. The green State 4 — his highest-wicket state — appears in noticeable amounts across many matches rather than being concentrated in just one or two, supporting the read that it's a real recurring sub-mode rather than a one-off.
- **MA Starc**: State 1 (orange, the near-absorbing broad mix) and State 0 (blue, tight in-swinger) trade off as the dominant component match-to-match — visually similar to the 2-state picture — but now States 2/3 (red/purple, the alternating finer-grained mixes) also appear as sizeable chunks in many matches. The green State 4 shows up as a small slice in a few matches, and **one whole short match (6 balls) is decoded entirely as State 4** — exactly the kind of match a 2-state model would have folded into "the broad mix" without comment.
- **PJ Cummins**: State 0 (blue, containment) is the largest component in almost every match, consistent with it being the default starting state. States 1–3 (orange/red/purple, the escalation cycle) appear in varying proportions match-to-match. The green State 4 (his 23%-wicket "strike" state) is visible in roughly half of matches, usually as a small slice — consistent with it being a short, occasional sub-mode rather than something Cummins settles into for long.

---

## Home vs away

The dataset records whether the bowling team was playing at home, away, or (rarely) on neutral ground. This field is blank for all the England-vs-Australia (Ashes) matches, so for those we derive it from `Ground Country`: a ground in England or Wales counts as "home" for England and "away" for Australia, and vice versa for grounds in Australia.

![State mix and outcomes: home vs away](hmm_home_away.png)

The top row shows the State 0–4 split for home vs away deliveries; the bottom row shows the overall wicket rate for home vs away (not split by state, since the per-state-per-venue samples get very small).

- **Wicket rate is higher at home for three of the four bowlers** (Morkel 2.3% vs 1.3%, Broad 3.3% vs 2.5%, Starc 4.0% vs 2.7%; Cummins is the exception at 1.4% home vs 2.1% away). This matches the general expectation that home conditions and crowd support tend to favour the bowling side.
- **Morkel's home sample (n=43, essentially one match) is dominated by State 3** (70% — his "leakiest" round-the-wicket mode) **vs only 37% away**, where the remaining states (0, 1, 2) are each better represented. The 18%-wicket State 4 is a similarly tiny sliver both home (2%) and away (3%).
- **Broad's highest-wicket state (State 4) is actually slightly *more* common away (33%) than at home (27%)**, yet his wicket rate is higher at home (3.3% vs 2.5%) — so, as with the 2-state version, the home/away wicket-rate gap doesn't come from spending more time in the high-wicket state; something else about home conditions is driving it.
- **Starc's tight in-swinger opening (State 0) is much more common at home** (33% vs 1% away), while away matches are dominated by the near-absorbing broad mix (State 1: 59% vs 21% home) — yet the *home* wicket rate is higher (4.0% vs 2.7%). Spending less time in the "broad mix" at home doesn't reduce his home wicket rate, reinforcing that the home/away gap is more about conditions than which state he's in.
- **Cummins' state mix is fairly similar home and away** (State 0 dominant in both: 56% home vs 61% away; State 4, his 23%-wicket state, is 5% in both) — yet his *away* wicket rate is higher (2.1% vs 1.4%). With an almost identical state mix, the away wicket-rate edge must come from within-state differences rather than which "strategy" he's in.

---

## Caveats

- **5 states is a deliberate over-fit, by choice.** BIC prefers 2 states for every bowler, by a wide margin (e.g. Morkel: 1556.8 at 2 states vs 1896.2 at 5). The 2-state model is the statistically "safer" choice, but it mostly just recovers the over/round-the-wicket split. 5 states was chosen here to dig for finer structure underneath that split — some of it (e.g. the small high-wicket states for Morkel, Broad, Starc and Cummins) is genuinely interesting, but it should be treated as **hypothesis-generating, not confirmed** — especially the states occupying under ~10% of deliveries (often only 10–20 balls), which a model with this many parameters can fit to noise.
- **Sample size**: 350–560 balls per bowler, split across 12–21 matches, is small for HMM fitting — many matches contribute only a handful of observations to the per-match decoding, and this gets more acute the more states are added.
- **The "strategies reset each match" assumption is itself a modelling choice**, not a certainty — international bowlers do carry plans and form across a series (e.g. "this batter struggled against X last time, do it again"). Treating matches as fully independent is a reasonable simplification, but match-to-match continuity is not literally zero.
- **The states are descriptive, not causal.** The HMM finds statistical regimes in the sequence; it doesn't know *why* the regime changed (new ball, declaration tactics, a different batter on strike, fatigue, etc.).
- **Label switching**: hidden states from EM have no inherent ordering. States were sorted by an "aggression score" (wicket rate ×10 + runs rate) purely so State 0...State 4 mean roughly the same thing (low to high aggression) across bowlers, but the absolute values are not directly comparable across bowlers (each HMM was fit independently).
- **Shared clustering with `analysis.md`**: the ball clusters feeding the HMM are identical to the over/round-the-wicket-split clusters described and named in `analysis.md`, so cluster names (O0, R1, ...) mean the same thing in both documents.
- **Starc's O0 (Length, out-swinger)** contains a small number of deliveries with an anomalous (physically implausible) positive Drop Angle — a Hawk-Eye tracking artefact flagged in `analysis.md`, not a genuine delivery type. It sits inside State 4 (5.2% of deliveries) and shouldn't be over-interpreted.
