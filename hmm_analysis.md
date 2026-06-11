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

- **Ball cluster**: the per-bowler over/round-the-wicket-split cluster from `analysis.md` (4–6 clusters depending on bowler, e.g. O0/O1/.../R0/R1/...), capturing *what kind of ball it was*. The HMM's "alphabet size" (`n_clusters × 4` outcomes) is therefore 4× the total cluster count from `analysis.md`: Morkel 24, Broad 20, Starc 24, Cummins 16.
- **Outcome**: one of four categories capturing *what happened*:
  - **Dot** — no runs conceded, no wicket
  - **Runs** — runs conceded off the bat, no wicket
  - **Chance** — the bowler beat the bat, found an edge, or got the batter playing-and-missing/under appeal but didn't take the wicket this time: `Ball Events` contains "Edge", "Catch Chance", "Play and Miss" or "Appeal", with `Is Wicket` False
  - **Wicket** — the bowler took the wicket

With this broader definition, chances are a genuinely sizeable share of deliveries — Morkel 12 (3.4%), Broad 43 (11.2%), Starc 21 (5.5%), Cummins 23 (6.1%) — large enough to noticeably reshape several states' fitted emission probabilities (not just one-off annotations). A state's raw "Wicket" rate can understate how close it actually came to producing a wicket if a chance went begging. Adding this category does mean the alphabet is noticeably bigger (16–24 symbols vs 12–18 previously), which is part of why BIC values in this version are higher across the board — there are simply more emission probabilities to estimate.

### A descriptive extra: the dot-ball "pressure" streak

Alongside the per-ball outcome, we also compute — for context only, *not* as part of the HMM's alphabet — how many **consecutive Dot-outcome deliveries** have just been bowled, up to and including the current ball, resetting to 0 on any Runs/Chance/Wicket ball (and at the start of each innings). E.g. three dots followed by another dot gives 1, 2, 3, 4; a run scored anywhere resets the count to 0. This is a simple proxy for "building pressure" that a bowler/captain might feel even though the HMM never sees it directly. Raw streak values run as high as 19–21 balls, far too many distinct values to add to the emission alphabet (it would multiply the alphabet size several-fold and overfit badly on ~350–560 observations), so instead we report the **mean streak length of the deliveries decoded into each state** — see the figure and discussion below.

Deliveries are ordered **chronologically** (match start date → innings → over → ball), so the HMM sees each bowler's deliveries in the order they were actually bowled, across all 51 matches in the dataset. The 7 tracking-error deliveries (<75 mph) excluded from the clustering are excluded here too.

### Strategies reset each innings

A bowler doesn't carry a "strategy" across a gap of weeks or months between Test matches — or even across the gap between bowling in the first innings of a match and coming back for the second. Whatever plan he had going into the last ball of one innings has no bearing on the first ball of the next. Originally this analysis concatenated all of a bowler's deliveries into one long sequence (or, in an earlier version, split only at match boundaries) and let the HMM model transitions across those gaps too, which implicitly assumes the strategy *can* carry over.

To fix this, each bowler's deliveries are still placed in one long chronological array (348–385 symbols, alphabet size `n_clusters × 3` = 12–18), but `hmmlearn` is given a `lengths` array marking where each of the bowler's **innings** (not just matches) starts and ends — splitting on (Match ID, Innings ID) rather than just Match ID. A bowler can appear in both innings of the same Test, so this gives noticeably more (shorter) sequences per bowler than splitting by match alone: 15 innings for Morkel (vs 12 matches), 35 for Broad (vs 21), 28 for Starc (vs 17), and 24 for Cummins (vs 15). This means:

- The **transition matrix** `A` is only ever applied *within* an innings — there's no learned transition from the last ball of one innings to the first ball of the next.
- The **start probabilities** π are applied at the start of *every* innings, not just once at the start of the whole sequence — so π now has a real interpretation: "what state does this bowler tend to begin an innings in?"
- The **emission distributions** are still shared and fit across all innings (pooling data for statistical power), but decoding (Viterbi) is run innings-by-innings.

This is a standard way to handle multiple independent sequences with a single HMM, and it directly answers the question of whether a bowler's "starting strategy" is consistent innings-to-innings (persistent) or essentially reset/random.

---

## Choosing the number of hidden states

We fit the HMM separately for each bowler with **2, 3, 4, and 5 hidden states** (15 random restarts each, keeping the best-likelihood fit), and compare them with the **Bayesian Information Criterion (BIC)**, which penalises extra parameters — useful here because more states quickly add a lot of emission parameters relative to the amount of data.

![Choosing the number of hidden states](hmm_state_selection.png)

For **all four bowlers, BIC is minimised at 2 states** and rises steadily and substantially through 3, 4 and 5 (e.g. Morkel goes from 1694.6 at 2 states to 2122.3 at 5). Statistically, a two-state model is about as much structure as 350–560 balls per bowler can support — at 5 states, several states are decoded from only a handful of innings, so they should be read as suggestive rather than well-estimated.

**Despite that, the results below deliberately use 5 hidden states per bowler.** The 2-state model (used in earlier drafts of this analysis) turned out to map almost exactly onto each bowler's **over/round-the-wicket split** — interesting, but it doesn't tell us much about *strategy* beyond "which side of the stumps". The aim of pushing to 5 states is to see whether, underneath that dominant split, there's any finer-grained structure — e.g. different "modes" *within* a bowler's round-the-wicket spells, or a short-lived high-wicket "strike" state that a 2-state model would average away. As the BIC plot shows, this is a **deliberate trade of statistical parsimony for exploratory detail** — treat the small, rarely-occupied states (anything under ~10% occupancy) as a hint to investigate further with more data, not a confirmed finding.

States are relabelled (after fitting) in order of "aggression" (wicket rate ×10 + chance rate ×5 + runs rate) — `State 0` is the most defensive, `State 4` the most aggressive — purely so states can be compared consistently across the figures. Chances are weighted at half a wicket: a near-miss is more "aggressive" than a plain run conceded, but less than an actual wicket. This relabelling means `State 0`...`State 4` here are not the "same" states as in earlier versions of this analysis — the model has been refit from scratch each time.

---

## Results

![Decoded hidden state through time](hmm_timelines.png)

![Emission profiles per hidden state](hmm_emissions.png)

![Delivery types bowled within each decoded state](hmm_cluster_histograms.png)

![Mean dot-ball pressure streak by hidden state](hmm_dot_streaks.png)

The timeline plot shows the Viterbi-decoded state for every delivery in chronological order (★ = wicket); the thin grey dotted lines mark **innings boundaries** — the HMM resets to its start distribution at each one, and no transition is modelled across them (whether between the two innings of the same match or across different matches). The emission plot shows, for each state, the model's *fitted* probability of bowling each ball cluster (top row) and each outcome (bottom row). The histogram below it shows the same idea from the *decoded* side: for each bowler, it takes every delivery the Viterbi path actually assigned to State 0 / State 1, and plots what share of those deliveries were each cluster. Cluster codes (O0, O1, R0, ...) are the same ones used in `analysis.md` — see each bowler's table below for what they mean. The last figure shows, for each state, the **mean dot-ball "pressure" streak** (see "A descriptive extra" above) of the deliveries decoded into it — i.e. on average, how many consecutive dot balls had just been bowled by the time a ball in that state was bowled.

### M Morkel

| | State 0 (12.6%, ~1.9 balls) | State 1 (17.8%, ~1 ball) | State 2 (27.6%, ~9.1 balls) | State 3 (16.7%, ~1 ball) | State 4 (25.3%, ~27.1 balls) |
|---|---|---|---|---|---|
| Dominant clusters | R0 In-swinger (63%), R2 Out-swinger (37%) | R1 Full outside off (16%), R2 Out-swinger (77%), R3 Bouncer (7%) | O0 Short (27%), O1 Good length (73%) | R0 In-swinger (30%), R1 Full outside off (16%), R2 Out-swinger (36%), R3 Bouncer (16%) | R0 In-swinger (27%), R1 Full outside off (48%), R2 Out-swinger (19%), R3 Bouncer (6%) |
| Outcome | Dot 94%, Runs 6%, Chance 0%, Wicket 0% | Dot 83%, Runs 17%, Chance 0%, Wicket 0% | Dot 71%, Runs 25%, Chance 2% (+2), **Wicket 2%** | Dot 43%, Runs 55%, Chance 2% (+1), Wicket 0% | Dot 70%, Runs 17%, **Chance 10%** (+9), **Wicket 3%** |
| Mean dot-streak | 3.1 | 2.3 | 2.0 | 1.8 | 3.6 |
| Starts an innings in this state | 6 / 15 | 0 / 15 | 6 / 15 | 0 / 15 | 3 / 15 |

The over/round-the-wicket split is still the dominant axis — State 2 is essentially 100% **over-the-wicket** (O0/O1, 27.6% of deliveries, persistent ~9-ball dwell), while States 0, 1, 3 and 4 are all **round-the-wicket**. With the broader "Edge / Catch Chance / Play and Miss / Appeal" definition of Chance, almost all of Morkel's 12 recorded chances (9 of 12) land in **State 4 (25.3%, his largest single state, persistent ~27-ball dwell, full-outside-off/in-swinger heavy)** — which now combines a 3% wicket rate with a striking **10% chance rate**, a 13% combined "threat rate", clearly his most dangerous mode. It's also his **highest-pressure state by mean dot-streak (3.6 balls)** — i.e. it's both where he's most threatening *and* where he's built up the most consecutive dots beforehand, consistent with a settled, building-pressure spell. It's reached mainly from State 3 (86%) and self-sustains 96% of the time. State 2 (over-the-wicket short/good-length mix, sticky, dwell 9.1) is his other large settled state and a common starting state (6/15), Dot 71% with modest 2% wicket and 2% chance rates. State 0 (12.6%, in-swinger/out-swinger round, dwell 1.9) is extremely tight (Dot 94%, no chances/wickets) and his other common starting state (6/15), with the second-highest mean dot-streak (3.1) — a clean pressure-building mode. States 1 and 3 (17.8%/16.7%, both ~1-ball dwell) are leakier transitional states with the lowest mean streaks (2.3/1.8) — the moments the model decodes pressure being released.

### SCJ Broad

| | State 0 (27.0%, ~1.9 balls) | State 1 (24.9%, ~1 ball) | State 2 (10.1%, absorbing) | State 3 (14.0%, ~25.0 balls) | State 4 (23.9%, ~1 ball) |
|---|---|---|---|---|---|
| Dominant clusters | R0 Back of length, nip-away (27%), R1 Full and straight (52%), R2 Good length, big swing away (21%) | R0 Back of length, nip-away (66%), R1 Full and straight (31%), R2 Good length, big swing away (2%) | O0 Short ball/bouncer (15%), O1 Good length, angle across (62%), R0 Back of length, nip-away (18%), R1 Full and straight (5%) | R1 Full and straight (26%), R2 Good length, big swing away (72%), R0 (2%) | R0 Back of length, nip-away (44%), R1 Full and straight (50%), R2 Good length, big swing away (5%) |
| Outcome | Dot 72%, Runs 24%, Chance 2% (+2), **Wicket 2%** | Dot 73%, Runs 18%, **Chance 9%** (+8), Wicket 0% | Dot 64%, Runs 28%, Chance 7% (+3), Wicket 0% | Dot 53%, Runs 19%, **Chance 27%** (+15), Wicket 0% | Dot 64%, Runs 9%, **Chance 16%** (+15), **Wicket 11%** |
| Mean dot-streak | 2.7 | 2.4 | 1.6 | 0.9 | 2.4 |
| Starts an innings in this state | 0 / 35 | 31 / 35 | 4 / 35 | 0 / 35 | 0 / 35 |

The broader Chance definition reshapes Broad's states substantially — Chance is now his **single most common non-Dot outcome** in three of five states. **State 1 (24.9%, transient ~1-ball dwell, back-of-length nip-away dominant) is overwhelmingly his starting state (31/35 innings)**: Dot 73% with a 9% chance rate (8 recorded), but it *always* transitions into **State 4** — by far his most dangerous state (23.9%, full-and-straight/back-of-length mix), which combines **Wicket 11% + Chance 16% = 27% combined threat, accounting for 10 of his 12 wickets and 15 of his 43 chances**. State 4 then mostly returns to State 0 (83%), forming a State 1 → State 4 → State 0 opening cycle. State 0 (27.0%, dwell 1.9, full-and-straight/big-swing-away mix) is reasonably contained (Dot 72%, 2 wickets, 2 chances) and has the **highest mean dot-streak (2.7)** — his closest thing to a settled containment mode. State 3 (14.0%, persistent ~25-ball dwell, big-swing-away dominant, self-transition 0.96) is an unusual state: **his highest chance rate by far (27%, 15 recorded) but zero wickets** — and, fittingly, the **lowest mean dot-streak (0.9)**, since so many of its deliveries are "Chance" rather than "Dot". State 2 (10.1%, absorbing — self-transition 1.0, over-the-wicket O1-heavy) is the small minority over-the-wicket mode: the 4 innings (4/35) that start here stay here for their entirety, Dot 64%, Chance 7%, no wickets.

### MA Starc

| | State 0 (47.3%, ~36.6 balls) | State 1 (31.6%, ~20.8 balls) | State 2 (2.9%, ~10.0 balls) | State 3 (9.1%, ~1.1 balls) | State 4 (9.1%, ~1.0 ball) |
|---|---|---|---|---|---|
| Dominant clusters | O1 Length, seam in (35%), O2 Length, in-swinger (12%), O3 Full in-swinger (25%), O4 Bouncer (28%) | O1 Length, seam in (10%), O2 Length, in-swinger (67%), O3 Full in-swinger (6%), O4 Bouncer (9%), R0 (7%) | O0 Length, out-swinger (100%) | O1 Length, seam in (53%), O2 Length, in-swinger (9%), O3 Full in-swinger (26%), O4 Bouncer (12%) | O1 Length, seam in (32%), O2 Length, in-swinger (10%), O3 Full in-swinger (58%) |
| Outcome | Dot 68%, Runs 30%, Chance 0%, **Wicket 1%** | Dot 70%, Runs 24%, Chance 3% (+4), **Wicket 3%** | Dot 73%, Runs 18%, **Wicket 9%** | Dot 61%, Runs 9%, **Chance 21%** (+7), **Wicket 9%** | Dot 39%, Runs 27%, **Chance 27%** (+10), **Wicket 7%** |
| Mean dot-streak | 1.9 | 1.7 | 1.9 | 1.2 | 0.7 |
| Starts an innings in this state | 9 / 28 | 13 / 28 | 2 / 28 | 3 / 28 | 1 / 28 |

Starc bowls almost entirely **over the wicket** — only State 1 mixes in any round-the-wicket deliveries (7%), so the over/round split barely registers for him at all. **States 0 and 1 between them now make up 78.9% of his deliveries and account for 22 of his 28 innings starts**, and both are extremely sticky (self-transitions 0.973 and 0.952, dwells of ~37 and ~21 balls) — once Starc settles into either, it's effectively his mode for the rest of the innings. State 0 (47.3%, his single largest state, broad seam-in/full-in-swinger/bouncer mix) is moderately tight (Dot 68%, 1% wicket rate, no chances). State 1 (31.6%, near-pure length-in-swinger at 67%, his most common starting state at 13/28) carries a 3% wicket rate plus 3% chance rate (4 recorded). States 3 and 4 (9.1% each, ~1-ball dwell) form a tight, high-risk cycle (State 3 → State 4 95%, State 4 → State 3 87%): **State 3 carries Wicket 9% + Chance 21% = 30% combined threat (7 chances + 3 wickets)**, and **State 4 — his leakiest state by far (Dot 39%, the lowest of any state, and the lowest mean dot-streak at 0.7) — carries Wicket 7% + Chance 27% = 34%, the highest combined threat for Starc**, with 10 of his 21 recorded chances. State 2 (2.9%, ~10.0-ball dwell, 100% O0 Length out-swinger) carries a 9% wicket rate but is by far his smallest, rarest-occupied state. *Caveat*: as in `analysis.md`, O0 (Length, out-swinger) contains a handful of deliveries with an anomalous (physically implausible) positive Drop Angle reading — a Hawk-Eye tracking artefact, not a genuine delivery type — and these sit inside State 2.

### PJ Cummins

| | State 0 (30.5%, ~5.0 balls) | State 1 (36.3%, ~9.5 balls) | State 2 (2.4%, ~1.0 ball) | State 3 (15.3%, ~1.0 ball) | State 4 (15.5%, ~1.0 ball) |
|---|---|---|---|---|---|
| Dominant clusters | O0 Good swinger (39%), O1 Angle-in (59%), O2 Bouncer (2%) | O0 Good swinger (73%), O1 Angle-in (12%), O2 Bouncer (15%) | O1 Angle-in (59%), O2 Bouncer (41%) | O0 Good swinger (35%), O1 Angle-in (58%), O2 Bouncer (5%), R0 (2%) | O0 Good swinger (11%), O1 Angle-in (70%), O2 Bouncer (15%), R0 (4%) |
| Outcome | Dot 86%, Runs 12%, Chance 0%, **Wicket 2%** | Dot 50%, Runs 46%, Chance 4% (+5), **Wicket 1%** | Dot 92%, Runs 0%, Chance 0%, **Wicket 8%** | Dot 56%, Runs 34%, **Chance 9%** (+5), Wicket 0% | Dot 50%, Runs 23%, **Chance 23%** (+13), **Wicket 4%** |
| Mean dot-streak | 3.0 | 1.3 | 1.6 | 1.2 | 1.1 |
| Starts an innings in this state | 19 / 24 | 5 / 24 | 0 / 24 | 0 / 24 | 0 / 24 |

The broader Chance definition changes Cummins' picture substantially. **State 0 (30.5%, dwell 5.0, swinger/angle-in mix) is overwhelmingly his starting state (19/24 innings)**: tight (Dot 86%), his highest "converted" wicket rate (2%, 3 wickets), no chances at all, and by far his **highest mean dot-streak (3.0)** — a settled, pressure-building default opening. State 1 (36.3%, his largest state, persistent ~9.5-ball dwell, near-pure good-swinger at 73%) is his leakiest by Dot rate (50%) and carries a 4% chance rate (5 recorded) plus a 1% wicket rate — a long but only moderately threatening containment spell. States 3 and 4 (15.3%/15.5%, both ~1-ball dwell) form a tight cycle (State 3 → State 4 96%, State 4 → State 3 always): State 3 carries a 9% chance rate (5 recorded) with no wickets, while **State 4 — Dot 50%, Wicket 4% + Chance 23% = 27% combined threat, the highest for Cummins, with 13 of his 23 recorded chances** — is reached almost exclusively from State 3 and always returns there. State 2 (2.4%, tiny, angle-in/bouncer mix, Dot 92%, Wicket 8%) is too small a sample (n=9) to read with confidence.

---

## Dot-ball pressure streaks by state

For each bowler, the dot-streak plot above shows the **mean "consecutive dots so far" streak** of the deliveries decoded into each state — a rough proxy for how much pressure had been built up by the time that ball was bowled, regardless of what happens on the ball itself. A few patterns stand out:

- **High mean streak does not simply mean "safe"**: Morkel's State 4 has *both* the highest mean streak (3.6) *and* his highest combined threat rate (13%) — i.e. his most dangerous state is reached after he's already strangled the batter for several dot balls, not despite it.
- **Low mean streak often coincides with high Chance rates**, almost mechanically: Broad's State 3 (27% chance rate) has the lowest streak of any state in the dataset (0.9), and Starc's State 4 (27% chance rate) is similarly low (0.7) — a "Chance" outcome breaks the streak by definition, so states where chances cluster necessarily show shorter streaks on average.
- **Cummins' State 0 is the standout "pressure" state across all four bowlers (mean streak 3.0)** — his most common starting state, tight (Dot 86%) and his most converting (2% wickets, 0% chances) — a clean, sustained squeeze rather than a string of near-misses.
- For all four bowlers, the state(s) with the highest combined wicket+chance "threat" rate have noticeably *lower* mean streaks than their tightest state — consistent with these high-threat states being short, eventful bursts rather than long accumulations of dots.

---

## How does the strategy mix change innings-by-innings?

With each innings treated as its own sequence, this plot is a direct readout of the model's per-innings behaviour rather than an artefact of one long concatenated decode.

![Hidden-state mix per innings](hmm_innings_states.png)

Each bar is one innings (in chronological order, left to right), showing what fraction of that bowler's deliveries in that innings were decoded as State 0 (blue), State 1 (orange), State 2 (red), State 3 (purple) or State 4 (green). The number above each bar is how many deliveries that bowler bowled at this batter in that innings — many innings are single-figure samples, so individual bars should be read cautiously, but the overall pattern across many innings is informative.

- **M Morkel**: most innings show a mix of several states, with State 2 (over-the-wicket settled mode, red) and State 4 (his largest, highest-threat round-the-wicket mode, green) both frequently the largest components — several innings are decoded as almost entirely State 2 (his over-the-wicket innings) or almost entirely State 4 (his highest-threat round-the-wicket innings), while others mix all five states within the innings, consistent with **wicket-position and mode choice being a within-innings tactical variable** for Morkel.
- **SCJ Broad**: a handful of innings (including his very first and last) are decoded as **almost entirely State 2 (red, his absorbing, wicket-free over-the-wicket mode)** — exactly as expected from a self-transition probability of 1.0: once he goes there for an innings, he never leaves. One innings is decoded almost entirely as **State 3 (purple, his persistent, highest-chance-rate mode)**. Most other innings instead show the **State 1 → State 4 → State 0 cycle (orange/green/blue) repeating within the innings** — his core round-the-wicket repertoire, with State 4 (his highest-threat state) a recurring, often-large component.
- **MA Starc**: most innings are decoded as **predominantly State 0 (blue, his largest, broad seam-in/in-swinger/bouncer mix)** or **predominantly State 1 (orange, his near-pure length-in-swinger mode)** — echoing how these two sticky states make up 78.9% of his deliveries and 22/28 innings starts. A handful of innings show the **State 3/State 4 high-threat cycle (purple/green) as a sizeable block**, and one innings carries a slice of **State 2 (red, the rare 100%-O0 state)**.
- **PJ Cummins**: many innings are decoded as **predominantly State 1 (orange, his largest, leakiest containment mode)** or **predominantly State 0 (blue, his tight, pressure-building default opening)** — his two largest states. Several other innings show the **State 3 → State 4 cycle (purple/green) as a recurring block**, consistent with these being his two highest-chance states. State 2 (red, just 2.4% overall) appears only as a thin sliver in one or two innings.

---

## Home vs away

The dataset records whether the bowling team was playing at home, away, or (rarely) on neutral ground. This field is blank for all the England-vs-Australia (Ashes) matches, so for those we derive it from `Ground Country`: a ground in England or Wales counts as "home" for England and "away" for Australia, and vice versa for grounds in Australia.

![State mix and outcomes: home vs away](hmm_home_away.png)

The top row shows the State 0–4 split for home vs away deliveries; the bottom row shows the overall wicket rate for home vs away (not split by state, since the per-state-per-venue samples get very small).

- **Wicket rate is higher at home for three of the four bowlers** (Morkel 2.3% vs 1.3%, Broad 3.3% vs 2.5%, Starc 4.0% vs 2.7%; Cummins is the exception at 1.4% home vs 2.1% away). This matches the general expectation that home conditions and crowd support tend to favour the bowling side, and these wicket rates are unaffected by the Chance redefinition (Chance and Wicket are separate categories).
- **Morkel's home sample (n=43, essentially one innings) leans more on States 1 and 4** (28% + 28% = 56% home vs 16% + 25% = 41% away), while away spends much more time in the over-the-wicket State 2 (30% vs 9% home). State 4 — his largest, highest-threat state (3% wicket + 10% chance) — is actually slightly *more* common at home (28% vs 25% away), consistent with the higher home wicket rate, but with only n=43 at home this should be read cautiously.
- **Broad's State 3 (his persistent, highest-chance, zero-wicket mode) essentially disappears away (0% vs 18% home)**, while State 2 (his absorbing, wicket-free over-the-wicket mode) is almost twice as common away (19% vs 8% home) — both of these are "safe" states for the batter, yet the away wicket rate is *lower* (2.5% vs 3.3%), so removing time from State 3 hasn't obviously hurt Broad. State 4 (his highest-threat state, 11% wicket + 16% chance) is fractionally *more* common away (28% vs 23% home) but doesn't translate into more away wickets — again pointing to within-state, match-to-match variation rather than a clean state-mix story.
- **Starc's home deliveries lean heavily towards State 1 (53% vs 8% away)** — his moderately threatening length-in-swinger mode (3% wicket + 3% chance) — while away is dominated by State 0 (58% vs 37% home), his largest but more contained state (1% wicket, 0% chance). Home also carries more of his high-threat pocket (States 3/4 combined: 2%+2%=4% home vs 16%+17%=33% away) — so although *away* spends much more time in the highest-threat states 3/4, the *home* wicket rate is still higher (4.0% vs 2.7%), suggesting the home/away gap here is more about overall conditions than which of these five states he's in.
- **Cummins' state mix differs sharply home vs away**: State 0 (his tight, pressure-building default opening, 0% chance, 2% wicket) is more than twice as common at home (46% vs 21% away), while States 3 and 4 (his two highest-chance states, especially State 4 at 4% wicket + 23% chance) are far more common away (21%+22%=43% away vs 5%+4%=9% home). Despite home leaning on his "safest" state, the *away* wicket rate is higher (2.1% vs 1.4%) — consistent with away innings spending much more time in his highest-chance state (State 4), even if not all those chances convert.

---

## Caveats

- **5 states is a deliberate over-fit, by choice.** BIC prefers 2 states for every bowler, by a wide margin (e.g. Morkel: 1694.6 at 2 states vs 2122.3 at 5). The 2-state model is the statistically "safer" choice, but it mostly just recovers the over/round-the-wicket split. 5 states was chosen here to dig for finer structure underneath that split — some of it (e.g. the high-threat states for Morkel, Broad, Starc and Cummins) is genuinely interesting, but it should be treated as **hypothesis-generating, not confirmed** — especially the transient states with ~1-ball expected dwell times, which a model with this many parameters can fit to noise.
- **Sample size**: 350–560 balls per bowler, split across 15–35 innings, is small for HMM fitting — many innings contribute only a handful of observations to the per-innings decoding, and this gets more acute the more states are added.
- **The "strategies reset each innings" assumption is itself a modelling choice**, not a certainty — international bowlers do carry plans and form across a match or series (e.g. "this batter struggled against X last time, do it again"). Treating innings as fully independent is a reasonable simplification, but innings-to-innings continuity is not literally zero.
- **The states are descriptive, not causal.** The HMM finds statistical regimes in the sequence; it doesn't know *why* the regime changed (new ball, declaration tactics, a different batter on strike, fatigue, etc.).
- **Label switching**: hidden states from EM have no inherent ordering. States were sorted by an "aggression score" (wicket rate ×10 + chance rate ×5 + runs rate) purely so State 0...State 4 mean roughly the same thing (low to high aggression) across bowlers, but the absolute values are not directly comparable across bowlers (each HMM was fit independently).
- **Shared clustering with `analysis.md`**: the ball clusters feeding the HMM are identical to the over/round-the-wicket-split clusters described and named in `analysis.md`, so cluster names (O0, R1, ...) mean the same thing in both documents.
- **Chances**: deliveries where `Ball Events` records "Edge", "Catch Chance", "Play and Miss" or "Appeal", but no wicket, are their own outcome category alongside Dot/Runs/Wicket (12–43 per bowler, 3.4%–11.2% of deliveries). With this broader definition, Chance is genuinely common enough to be a primary driver of several states' fitted emission probabilities — e.g. Broad's State 3 (27% chance rate) and State 4 (16%), Starc's States 3/4 (21%/27%), Cummins' State 4 (23%) — rather than a one-off annotation. The per-state "+N chance" counts in each table are the underlying raw figures the percentages are derived from.
- **The dot-ball pressure streak is a derived, history-dependent feature, not an HMM emission.** It's computed *after* fitting, purely as a way of describing what each decoded state "feels like" in terms of recent run-up — it played no role in the model fitting itself, and (unlike the four outcome categories) is not mutually exclusive with them: a ball can simultaneously be, say, the 4th dot in a row *and* be classified as a Chance (which then resets the streak for the next ball).
- **Starc's O0 (Length, out-swinger)** contains a small number of deliveries with an anomalous (physically implausible) positive Drop Angle — a Hawk-Eye tracking artefact flagged in `analysis.md`, not a genuine delivery type. It now sits inside State 2 (2.9% of deliveries) and shouldn't be over-interpreted.
