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

Separately, the dataset also flags **chances** — deliveries where a fielder dropped a catch, missed a run-out, or the keeper fumbled a stumping/catch (`Fielder Action` = Dropped Catch / Run Out Chance / Keeper Error), i.e. a wicket *should* have happened but didn't. These are very rare (0–4 per bowler across the whole dataset) — far too few to add as a fourth outcome category in the HMM's alphabet without destabilising the fit — so they aren't part of the symbol/emission model. Instead, where a chance occurred, it's flagged against the hidden state it was decoded into in the results below, as extra context: a state's "Wicket" rate alone can understate how close it actually came to producing a wicket.

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

For **all four bowlers, BIC is minimised at 2 states** and rises steadily and substantially through 3, 4 and 5 (e.g. Morkel goes from 1549.9 at 2 states to 1887.3 at 5). Statistically, a two-state model is about as much structure as 350–560 balls per bowler can support — at 5 states, several states are decoded from only a handful of innings, so they should be read as suggestive rather than well-estimated.

**Despite that, the results below deliberately use 5 hidden states per bowler.** The 2-state model (used in earlier drafts of this analysis) turned out to map almost exactly onto each bowler's **over/round-the-wicket split** — interesting, but it doesn't tell us much about *strategy* beyond "which side of the stumps". The aim of pushing to 5 states is to see whether, underneath that dominant split, there's any finer-grained structure — e.g. different "modes" *within* a bowler's round-the-wicket spells, or a short-lived high-wicket "strike" state that a 2-state model would average away. As the BIC plot shows, this is a **deliberate trade of statistical parsimony for exploratory detail** — treat the small, rarely-occupied states (anything under ~10% occupancy) as a hint to investigate further with more data, not a confirmed finding.

States are relabelled (after fitting) in order of "aggression" (wicket rate ×10 + runs rate) — `State 0` is the most defensive, `State 4` the most aggressive — purely so states can be compared consistently across the figures. This relabelling means `State 0`...`State 4` here are not the "same" states as in the 2-state version of this analysis — the model has been refit from scratch with 5 components.

---

## Results

![Decoded hidden state through time](hmm_timelines.png)

![Emission profiles per hidden state](hmm_emissions.png)

![Delivery types bowled within each decoded state](hmm_cluster_histograms.png)

The timeline plot shows the Viterbi-decoded state for every delivery in chronological order (★ = wicket); the thin grey dotted lines mark **innings boundaries** — the HMM resets to its start distribution at each one, and no transition is modelled across them (whether between the two innings of the same match or across different matches). The emission plot shows, for each state, the model's *fitted* probability of bowling each ball cluster (top row) and each outcome (bottom row). The histogram below it shows the same idea from the *decoded* side: for each bowler, it takes every delivery the Viterbi path actually assigned to State 0 / State 1, and plots what share of those deliveries were each cluster. Cluster codes (O0, O1, R0, ...) are the same ones used in `analysis.md` — see each bowler's table below for what they mean.

### M Morkel

| | State 0 (7.2%, ~7.7 balls) | State 1 (17.8%, ~1 ball) | State 2 (17.5%, ~1 ball) | State 3 (27.9%, ~8.4 balls) | State 4 (29.6%, ~18.5 balls) |
|---|---|---|---|---|---|
| Dominant clusters | R0 In-swinger (33%), R1 Full outside off (41%), R3 Bouncer (27%) | R0 In-swinger (22%), R2 Out-swinger (65%), R3 Bouncer (5%) | R0 In-swinger (52%), R2 Out-swinger (38%), R3 Bouncer (10%) | O0 Short (27%), O1 Good length (73%) | R0 In-swinger (22%), R1 Full outside off (42%), R2 Out-swinger (34%) |
| Outcome | Dot 93%, Runs 7%, Wicket 0% | Dot 89%, Runs 11%, Wicket 0% | Dot 56%, Runs 44%, Wicket 0% | Dot 71%, Runs 27%, Wicket 2% | Dot 76%, Runs 22%, **Wicket 3%** |
| Starts an innings in this state | 0 / 15 | 3 / 15 | 5 / 15 | 6 / 15 | 1 / 15 |

The over/round-the-wicket split is still the dominant axis — State 3 is essentially 100% **over-the-wicket** (O0/O1, 27.9% of deliveries), while States 0, 1, 2 and 4 are all **round-the-wicket**. Within that round-the-wicket majority the model finds **four sub-modes**: State 0 (in-swinger/full-outside-off/bouncer mix, very tight, Dot 93%), State 2 (a similar in-swinger/out-swinger/bouncer mix that's by far his leakiest, Dot only 56%), State 1 (out-swinger heavy, also tight, Dot 89%), and **State 4 — his single most-occupied state (29.6%, ~18.5-ball dwell)** — a settled full-outside-off/out-swinger/in-swinger mix that's moderately leaky (Dot 76%) and produces his only "normal" wickets (3%). The transition matrix traces a loose cycle: the persistent over-the-wicket State 3 occasionally branches into State 0 (3.5%) or State 1 (8.5%); State 0 and State 2 swap back and forth as a transient detour; State 2 mostly feeds State 1 (92%), which in turn settles into State 4 (16%); and State 4 eventually drifts back to State 3 (5%). No chances (dropped catches/missed run-outs) were recorded for Morkel in this dataset.

### SCJ Broad

| | State 0 (19.0%, ~8.9 balls) | State 1 (8.1%, ~24.9 balls) | State 2 (23.9%, ~1 ball) | State 3 (27.8%, ~9.3 balls) | State 4 (21.3%, ~1 ball) |
|---|---|---|---|---|---|
| Dominant clusters | R2 Good length, big swing away (68%), R1 Full and straight (30%) | O0 Short ball/bouncer (16%), O1 Good length, angle across (81%) | R0 Back of length, nip-away (74%), R1 Full and straight (16%), R2 Good length, big swing away (8%) | R0 Back of length, nip-away (22%), R1 Full and straight (77%) | R0 Back of length, nip-away (57%), R1 Full and straight (34%), R2 Good length, big swing away (8%) |
| Outcome | Dot 79%, Runs 21%, Wicket 0% | Dot 61%, Runs 39%, Wicket 0% | Dot 68%, Runs 31%, **Wicket 1%** (+1 chance) | Dot 82%, Runs 13%, **Wicket 6%** | Dot 79%, Runs 15%, **Wicket 6%** (+1 chance) |
| Starts an innings in this state | 1 / 35 | 4 / 35 | 19 / 35 | 11 / 35 | 0 / 35 |

State 1 (8.1%) is now Broad's only near-pure **over-the-wicket** state (O0/O1, 97%, persistent dwell ~24.9 balls — once he goes over the wicket he tends to stay there for a long spell) and it's also his leakiest non-wicket-taking state (Dot 61%). The other four states are all **round-the-wicket**. State 2 (23.9%, transient ~1-ball dwell, back-of-length nip-away dominant) is his most common starting state by far (19/35 innings) but converts that mix into a wicket only 1% of the time (plus one dropped catch). **State 4 (21.3%, also transient) shares an almost identical cluster mix to State 2** — back-of-length nip-away plus full-and-straight — yet carries a **6% wicket rate** (plus another dropped catch); the two alternate with each other almost every ball (State 2 → State 4 89%, State 4 → State 2 92%), so this looks like the same physical "type" of ball producing very different outcomes depending on some other factor the model is picking up (sequencing, build-up, etc.). State 0 (19.0%, persistent ~8.9-ball dwell, swing-away dominant, Dot 79%, no wickets) is a containment mode. State 3 (27.8%, persistent ~9.3-ball dwell, full-and-straight dominant) is **his most defensive state by raw Dot rate (82%) yet also ties State 4 for his highest wicket rate (6%)** — together States 3 and 4 (49% of his deliveries) account for almost all of his wicket-taking.

### MA Starc

| | State 0 (20.4%, ~8.4 balls) | State 1 (52.7%, near-absorbing) | State 2 (11.2%, ~1 ball) | State 3 (10.4%, ~1 ball) | State 4 (5.2%, ~17 balls) |
|---|---|---|---|---|---|
| Dominant clusters | O1 Length, seam in (6%), O2 Length, in-swinger (87%), O3 Full in-swinger (7%) | O1 Length, seam in (38%), O2 Length, in-swinger (7%), O3 Full in-swinger (31%), O4 Bouncer (24%) | O1 Length, seam in (12%), O2 Length, in-swinger (47%), O3 Full in-swinger (23%), O4 Bouncer (18%) | O1 Length, seam in (43%), O2 Length, in-swinger (23%), O3 Full in-swinger (14%), O4 Bouncer (20%) | O0 Length, out-swinger (55%), O2 Length, in-swinger (5%), R0 Round-the-wicket length (40%) |
| Outcome | Dot 79%, Runs 21%, Wicket 0% | Dot 70%, Runs 27%, **Wicket 2%** (+1 chance) | Dot 60%, Runs 36%, **Wicket 4%** (+1 chance) | Dot 68%, Runs 23%, **Wicket 9%** | Dot 65%, Runs 25%, **Wicket 10%** |
| Starts an innings in this state | 11 / 28 | 12 / 28 | 0 / 28 | 2 / 28 | 3 / 28 |

Starc bowls almost entirely **over the wicket** — only State 4 (5.2%) mixes in any round-the-wicket deliveries (40%), so the over/round split barely registers for him at all. State 0 (20.4%, ~8.4-ball dwell) is the **tight, repetitive in-swinger-length opening** (O2, 87%, Dot 79%, no wickets) and, together with State 1, is how almost every innings begins (11+12 of 28). State 1 is **near-absorbing** (dwell ~180 balls — effectively "for the rest of the innings" once he settles into this broad seam-in/full-in-swinger/bouncer mix) and is by far his most-bowled state (52.7%); a dropped catch and a missed run-out chance both occurred here. States 2 and 3 are a pair of **ball-to-ball alternating states** (State 2 → State 3 always, State 3 → State 2 always) splitting "the broad mix" into two slightly different blends, both noticeably more wicket-prone (4% and **9%**) than States 0/1. **State 4 — the same out-swinger/round-the-wicket-length mix flagged in earlier versions of this analysis — again stands out with a 10% wicket rate**, the highest of any state for Starc, and is now a more persistent mode (~17-ball dwell) than before. *Caveat*: as in `analysis.md`, O0 (Length, out-swinger) contains a handful of deliveries with an anomalous (physically implausible) positive Drop Angle reading — a Hawk-Eye tracking artefact, not a genuine delivery type — and these sit inside State 4.

### PJ Cummins

| | State 0 (20.5%, ~1 ball) | State 1 (17.4%, ~1 ball) | State 2 (37.4%, ~5.3 balls) | State 3 (21.3%, ~1 ball) | State 4 (3.4%, ~2 balls) |
|---|---|---|---|---|---|
| Dominant clusters | O0 Good swinger (37%), O1 Angle-in (61%) | O0 Good swinger (4%), O1 Angle-in (96%) | O0 Good swinger (77%), O1 Angle-in (9%), O2 Bouncer (14%) | O0 Good swinger (36%), O1 Angle-in (52%), O2 Bouncer (12%) | O0 Good swinger (29%), O2 Bouncer (58%), R0 Round-the-wicket bouncer (13%) |
| Outcome | Dot 75%, Runs 25%, Wicket 0% | Dot 71%, Runs 29%, **Wicket 0%** (+1 chance) | Dot 55%, Runs 44%, **Wicket 1%** (+1 chance) | Dot 84%, Runs 10%, **Wicket 6%** (+1 chance) | Dot 52%, Runs 42%, **Wicket 6%** |
| Starts an innings in this state | 0 / 24 | 3 / 24 | 5 / 24 | 16 / 24 | 0 / 24 |

State 3 (21.3%, transient ~1-ball dwell, angle-in heavy) is by far Cummins' most common starting state (16/24 innings) — a "new spell" mode that's his **most defensive by Dot rate (84%) yet ties for his highest wicket rate (6%)**, plus one dropped catch. State 2 (37.4%, persistent ~5.3-ball dwell, good-swinger dominant) is his **single largest state and his leakiest (Dot 55%)** — also with a missed run-out chance recorded against it. States 0 and 1 (20.5% and 17.4%, both transient, swinger/angle-in mixes) sit in between, with State 1 carrying a keeper-error chance despite 0% wickets. **State 4 (3.4%, now a short-lived ~2-ball mode rather than a single-ball one) is the standout**: a bouncer/round-the-wicket-bouncer mix with a **6% wicket rate** — tied for his highest. The transition matrix traces a rough cycle: State 3 → State 0 (99%) → State 1 (60%) or State 2 (39%) → State 2 settles in (81% self-loop) → occasionally State 4 (7%) → back to State 0 (44%) or State 1 (7%) → State 1 → State 3 (always) — i.e. Cummins typically opens defensively (State 3), drifts into his settled good-swinger mode (State 2), and only occasionally breaks out into the high-wicket bouncer mode (State 4) before cycling back.

---

## How does the strategy mix change innings-by-innings?

With each innings treated as its own sequence, this plot is a direct readout of the model's per-innings behaviour rather than an artefact of one long concatenated decode.

![Hidden-state mix per innings](hmm_innings_states.png)

Each bar is one innings (in chronological order, left to right), showing what fraction of that bowler's deliveries in that innings were decoded as State 0 (blue), State 1 (orange), State 2 (red), State 3 (purple) or State 4 (green). The number above each bar is how many deliveries that bowler bowled at this batter in that innings — many innings are single-figure samples, so individual bars should be read cautiously, but the overall pattern across many innings is informative.

- **M Morkel**: most innings show a mix of the round-the-wicket states (0/blue, 1/orange, 2/red, 4/green), with State 3 (over-the-wicket, purple) appearing as a sizeable secondary component in many innings rather than a separate innings-level mode — consistent with **wicket-position choice being a within-innings tactical variable** for Morkel, not just a between-innings one. State 4 (his settled, most-occupied state) shows up as large blocks in several innings, particularly later in the sequence.
- **SCJ Broad**: most innings show a substantial mix of his round-the-wicket states (0/blue, 2/red, 3/purple, 4/green), but a handful of innings (roughly 6 of 35) are decoded as **almost entirely State 1 (orange, his over-the-wicket mode)** — i.e. on those occasions he spent essentially the whole innings on the other side of the stumps, rather than switching within the innings. State 4 (one of his two highest-wicket states) appears in noticeable amounts across many innings, supporting the read that it's a real recurring sub-mode rather than a one-off.
- **MA Starc**: many innings are decoded as **almost entirely State 1 (orange, the near-absorbing broad mix)** or **almost entirely State 0 (blue, the tight in-swinger opening)**, echoing how dominant these two states are overall (73% combined). States 2/3 (red/purple, the alternating finer-grained mixes) appear as sizeable chunks in several other innings. State 4 (green, his 10%-wicket state) shows up as a small slice in a few innings, and **at least one short innings is decoded almost entirely as State 4** — exactly the kind of innings a 2-state model would have folded into "the broad mix" without comment.
- **PJ Cummins**: State 2 (red, his settled good-swinger mode) is present in almost every innings, often as the largest single component, consistent with it being his largest-occupancy state overall. State 3 (purple, his defensive "new spell" mode) is prominent at the start of many innings, matching it being the dominant starting state (16/24). State 4 (green, tied for his highest-wicket state) appears as a small slice in only a handful of innings — consistent with it being a short, occasional sub-mode rather than something Cummins settles into for long.

---

## Home vs away

The dataset records whether the bowling team was playing at home, away, or (rarely) on neutral ground. This field is blank for all the England-vs-Australia (Ashes) matches, so for those we derive it from `Ground Country`: a ground in England or Wales counts as "home" for England and "away" for Australia, and vice versa for grounds in Australia.

![State mix and outcomes: home vs away](hmm_home_away.png)

The top row shows the State 0–4 split for home vs away deliveries; the bottom row shows the overall wicket rate for home vs away (not split by state, since the per-state-per-venue samples get very small).

- **Wicket rate is higher at home for three of the four bowlers** (Morkel 2.3% vs 1.3%, Broad 3.3% vs 2.5%, Starc 4.0% vs 2.7%; Cummins is the exception at 1.4% home vs 2.1% away). This matches the general expectation that home conditions and crowd support tend to favour the bowling side.
- **Morkel's home sample (n=43, essentially one innings) is dominated by State 4** (72% — his settled, moderately-leaky round-the-wicket mode, 3% wicket rate) **vs only 24% away**, where the remaining states (0, 1, 2, 3) are all better represented, including 30% in the over-the-wicket State 3. Despite home leaning heavily on a state with a non-trivial wicket rate, the small sample (n=43) makes this hard to read with confidence.
- **Broad's two highest-wicket states (State 3 and State 4, both 6%) together make up 47% of his home deliveries vs 55% away** — slightly *more* common away — yet his wicket rate is higher at home (3.3% vs 2.5%). As with earlier versions of this analysis, the home/away wicket-rate gap doesn't come from spending more time in the high-wicket states; something else about home conditions is driving it.
- **Starc's innings-based split now tells a more internally-consistent story than the per-match version**: at home, States 2 and 3 (his two highest-wicket states, 4% and 9%) make up 28% of deliveries, compared to just 14% away — and home's wicket rate is correspondingly higher (4.0% vs 2.7%). Away is dominated by State 1 (79%, his near-absorbing 2%-wicket mix). So for Starc, the home/away wicket-rate gap now lines up with which states he spends time in, not just "conditions" in the abstract.
- **Cummins shows the same improved alignment**: away, State 4 (his 6%-wicket bouncer/round-the-wicket-bouncer state) makes up 5% of deliveries vs only 1% at home — and his *away* wicket rate is higher (2.1% vs 1.4%). The rest of the state mix is broadly similar home and away (State 2 dominant in both: 43% home vs 34% away), so this one state looks like the main driver of the away-wicket-rate edge.

---

## Caveats

- **5 states is a deliberate over-fit, by choice.** BIC prefers 2 states for every bowler, by a wide margin (e.g. Morkel: 1549.9 at 2 states vs 1887.3 at 5). The 2-state model is the statistically "safer" choice, but it mostly just recovers the over/round-the-wicket split. 5 states was chosen here to dig for finer structure underneath that split — some of it (e.g. the high-wicket states for Morkel, Broad, Starc and Cummins) is genuinely interesting, but it should be treated as **hypothesis-generating, not confirmed** — especially the transient states with ~1-ball expected dwell times, which a model with this many parameters can fit to noise.
- **Sample size**: 350–560 balls per bowler, split across 15–35 innings, is small for HMM fitting — many innings contribute only a handful of observations to the per-innings decoding, and this gets more acute the more states are added.
- **The "strategies reset each innings" assumption is itself a modelling choice**, not a certainty — international bowlers do carry plans and form across a match or series (e.g. "this batter struggled against X last time, do it again"). Treating innings as fully independent is a reasonable simplification, but innings-to-innings continuity is not literally zero.
- **The states are descriptive, not causal.** The HMM finds statistical regimes in the sequence; it doesn't know *why* the regime changed (new ball, declaration tactics, a different batter on strike, fatigue, etc.).
- **Label switching**: hidden states from EM have no inherent ordering. States were sorted by an "aggression score" (wicket rate ×10 + runs rate) purely so State 0...State 4 mean roughly the same thing (low to high aggression) across bowlers, but the absolute values are not directly comparable across bowlers (each HMM was fit independently).
- **Shared clustering with `analysis.md`**: the ball clusters feeding the HMM are identical to the over/round-the-wicket-split clusters described and named in `analysis.md`, so cluster names (O0, R1, ...) mean the same thing in both documents.
- **Chances**: dropped catches, missed run-outs and keeper errors are extremely rare (0–4 per bowler) and aren't part of the HMM's symbol alphabet — they're noted against the relevant state in the results tables purely as extra context, not as something the model "knows about".
- **Starc's O0 (Length, out-swinger)** contains a small number of deliveries with an anomalous (physically implausible) positive Drop Angle — a Hawk-Eye tracking artefact flagged in `analysis.md`, not a genuine delivery type. It sits inside State 4 (5.2% of deliveries) and shouldn't be over-interpreted.
