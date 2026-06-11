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
  - **Chance** — the bowler beat the bat or found an edge but didn't take the wicket this time: `Ball Events` contains "Edge" or "Catch Chance", with `Is Wicket` False
  - **Wicket** — the bowler took the wicket

Chances defined this way are uncommon but not negligible — Morkel 5 (1.4% of his deliveries), Broad 15 (3.9%), Starc 7 (1.8%), Cummins 4 (1.1%) — a meaningful enough share that they show up directly in several states' fitted emission probabilities, not just as one-off annotations. A state's raw "Wicket" rate can understate how close it actually came to producing a wicket if a chance went begging. Adding this category does mean the alphabet is noticeably bigger (16–24 symbols vs 12–18 previously), which is part of why BIC values in this version are higher across the board — there are simply more emission probabilities to estimate.

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

For **all four bowlers, BIC is minimised at 2 states** and rises steadily and substantially through 3, 4 and 5 (e.g. Morkel goes from 1657.0 at 2 states to 2089.0 at 5). Statistically, a two-state model is about as much structure as 350–560 balls per bowler can support — at 5 states, several states are decoded from only a handful of innings, so they should be read as suggestive rather than well-estimated.

**Despite that, the results below deliberately use 5 hidden states per bowler.** The 2-state model (used in earlier drafts of this analysis) turned out to map almost exactly onto each bowler's **over/round-the-wicket split** — interesting, but it doesn't tell us much about *strategy* beyond "which side of the stumps". The aim of pushing to 5 states is to see whether, underneath that dominant split, there's any finer-grained structure — e.g. different "modes" *within* a bowler's round-the-wicket spells, or a short-lived high-wicket "strike" state that a 2-state model would average away. As the BIC plot shows, this is a **deliberate trade of statistical parsimony for exploratory detail** — treat the small, rarely-occupied states (anything under ~10% occupancy) as a hint to investigate further with more data, not a confirmed finding.

States are relabelled (after fitting) in order of "aggression" (wicket rate ×10 + chance rate ×5 + runs rate) — `State 0` is the most defensive, `State 4` the most aggressive — purely so states can be compared consistently across the figures. Chances are weighted at half a wicket: a near-miss is more "aggressive" than a plain run conceded, but less than an actual wicket. This relabelling means `State 0`...`State 4` here are not the "same" states as in earlier versions of this analysis — the model has been refit from scratch each time.

---

## Results

![Decoded hidden state through time](hmm_timelines.png)

![Emission profiles per hidden state](hmm_emissions.png)

![Delivery types bowled within each decoded state](hmm_cluster_histograms.png)

The timeline plot shows the Viterbi-decoded state for every delivery in chronological order (★ = wicket); the thin grey dotted lines mark **innings boundaries** — the HMM resets to its start distribution at each one, and no transition is modelled across them (whether between the two innings of the same match or across different matches). The emission plot shows, for each state, the model's *fitted* probability of bowling each ball cluster (top row) and each outcome (bottom row). The histogram below it shows the same idea from the *decoded* side: for each bowler, it takes every delivery the Viterbi path actually assigned to State 0 / State 1, and plots what share of those deliveries were each cluster. Cluster codes (O0, O1, R0, ...) are the same ones used in `analysis.md` — see each bowler's table below for what they mean.

### M Morkel

| | State 0 (13.5%, ~2.0 balls) | State 1 (19.8%, ~1 ball) | State 2 (27.6%, ~9.0 balls) | State 3 (21.6%, ~1 ball) | State 4 (17.5%, ~15.4 balls) |
|---|---|---|---|---|---|
| Dominant clusters | R0 In-swinger (69%), R2 Out-swinger (31%) | R1 Full outside off (29%), R2 Out-swinger (66%), R3 Bouncer (4%) | O0 Short (27%), O1 Good length (73%) | R0 In-swinger (35%), R1 Full outside off (6%), R2 Out-swinger (44%), R3 Bouncer (14%) | R0 In-swinger (24%), R1 Full outside off (61%), R2 Out-swinger (9%), R3 Bouncer (6%) |
| Outcome | Dot 96%, Runs 0%, **Chance 4%** (+2) | Dot 80%, Runs 20%, Wicket 0% | Dot 72%, Runs 25%, Chance 1% (+1), **Wicket 2%** | Dot 53%, Runs 46%, Chance 1% (+1), Wicket 0% | Dot 80%, Runs 13%, Chance 2% (+1), **Wicket 5%** |
| Starts an innings in this state | 8 / 15 | 0 / 15 | 6 / 15 | 0 / 15 | 1 / 15 |

The over/round-the-wicket split is still the dominant axis — State 2 is essentially 100% **over-the-wicket** (O0/O1, 27.6% of deliveries, persistent ~9-ball dwell), while States 0, 1, 3 and 4 are all **round-the-wicket**. With the new "Chance" category now tied to *Ball Events* (edges/near-edges), the round-the-wicket majority resolves into **four sub-modes**: State 0 (in-swinger/out-swinger mix, by far his tightest state, Dot 96%) is also — perhaps surprisingly — **his most common starting state (8/15 innings)**, and despite leaking almost nothing in runs it accounts for 2 of his 5 recorded chances — i.e. the bat is beaten here without conceding. From there the model decodes a transient pair, State 1 (full-outside-off/out-swinger, Dot 80%, no chances) and State 3 (a similar in-swinger/out-swinger/bouncer mix that's by far his leakiest, Dot only 53%, 1 chance), which mostly feed each other and State 2. State 2 (over-the-wicket short/good-length mix) is his other common starting state (6/15), Dot 72% with a small 2% wicket rate and 1 chance. **State 4 — his settled round-the-wicket mode (17.5%, ~15.4-ball dwell, full-outside-off/in-swinger heavy)** — combines a 5% wicket rate (his highest) with a 2% chance rate (1 recorded), an 7% combined "threat rate", the highest of his five states, and once entered the model keeps him there 94% of the time ball-to-ball.

### SCJ Broad

| | State 0 (33.0%, ~15.1 balls) | State 1 (17.1%, ~10.9 balls) | State 2 (21.0%, ~1.3 balls) | State 3 (10.6%, absorbing) | State 4 (18.2%, ~1.0 ball) |
|---|---|---|---|---|---|
| Dominant clusters | R0 Back of length, nip-away (66%), R1 Full and straight (26%), R2 Good length, big swing away (7%) | R1 Full and straight (25%), R2 Good length, big swing away (73%) | R0 Back of length, nip-away (42%), R1 Full and straight (58%) | O0 Short ball/bouncer (15%), O1 Good length, angle across (61%), R0 Back of length, nip-away (17%), R1 Full and straight (7%) | R0 Back of length, nip-away (20%), R1 Full and straight (64%), R2 Good length, big swing away (16%) |
| Outcome | Dot 73%, Runs 25%, Chance 1% (+1), Wicket 1% | Dot 77%, Runs 17%, **Chance 6%** (+4), Wicket 0% | Dot 78%, Runs 18%, Chance 1% (+1), **Wicket 3%** | Dot 61%, Runs 29%, **Chance 10%** (+4), Wicket 0% | Dot 77%, Runs 3%, **Chance 7%** (+5), **Wicket 13%** |
| Starts an innings in this state | 14 / 35 | 1 / 35 | 15 / 35 | 5 / 35 | 0 / 35 |

State 0 (33.0%, his single largest state, persistent ~15.1-ball dwell, back-of-length nip-away dominant, sticky self-transition 0.93) is reasonably tight (Dot 73%) and his most common starting state (14/35). State 2 (21.0%, transient ~1.3-ball dwell, back-of-length/full-and-straight mix) is the other common starting state (15/35), and sits in a tight cycle with State 4 (State 2 → State 4 75%, State 4 → State 2 always). **State 4 (18.2%, transient ~1-ball dwell, full-and-straight dominant) is by far his most threatening state: a 13% wicket rate plus a 7% chance rate — a 20% combined "threat rate", the highest of any state for any bowler in this analysis** — and carries 5 of his 15 recorded chances. State 3 (10.6%, mostly over-the-wicket O0/O1 with some back-of-length nip-away) is the most extreme state in the model: **its self-transition probability is 1.0 — once Broad enters this state, the model never decodes him leaving it again** — and it's an unusual one, with his **highest chance rate (10%, 4 recorded) but zero wickets**: plenty of near-misses that never convert, in the 5 innings (5/35) that are decoded as this mode for their entirety. State 1 (17.1%, persistent ~10.9-ball dwell, swing-away dominant, Dot 77%) is a similar story — 4 chances at a 6% rate but no wickets — a long containment spell where he beats the bat repeatedly without reward.

### MA Starc

| | State 0 (25.6%, ~8.9 balls) | State 1 (58.0%, ~73.9 balls) | State 2 (2.9%, ~10.0 balls) | State 3 (6.0%, ~1.0 ball) | State 4 (7.6%, ~1.0 ball) |
|---|---|---|---|---|---|
| Dominant clusters | O1 Length, seam in (10%), O2 Length, in-swinger (79%), O3 Full in-swinger (8%), O4 Bouncer (3%) | O1 Length, seam in (35%), O2 Length, in-swinger (8%), O3 Full in-swinger (30%), O4 Bouncer (22%), R0 Round-the-wicket length (4%) | O0 Length, out-swinger (100%) | O1 Length, seam in (59%), O2 Length, in-swinger (5%), O3 Full in-swinger (6%), O4 Bouncer (30%) | O1 Length, seam in (14%), O2 Length, in-swinger (45%), O3 Full in-swinger (20%), O4 Bouncer (22%) |
| Outcome | Dot 81%, Runs 19%, Wicket 0% | Dot 67%, Runs 30%, Chance 1% (+2), **Wicket 2%** | Dot 73%, Runs 18%, **Wicket 9%** | Dot 73%, Runs 4%, **Chance 9%** (+2), **Wicket 14%** | Dot 38%, Runs 37%, **Chance 10%** (+3), **Wicket 14%** |
| Starts an innings in this state | 12 / 28 | 14 / 28 | 0 / 28 | 0 / 28 | 0 / 28 |

Starc bowls almost entirely **over the wicket** — only State 1 mixes in any round-the-wicket deliveries (4%), so the over/round split barely registers for him at all. **States 0 and 1 between them account for 26 of his 28 innings starts (12 and 14 respectively)** — Starc almost always opens with one of two modes. **State 1 (58.0%, ~73.9-ball dwell, his single largest state by far, a broad seam-in/full-in-swinger/bouncer mix) is now massively dominant — once entered it's effectively his default for the rest of an innings** (self-transition 0.986), Dot 67%, with a 2% wicket rate plus 2 recorded chances. State 0 (25.6%, ~8.9-ball dwell, near-pure in-swinger-length at 79%, Dot 81%, no chances or wickets) is his other major opening state, occasionally diverting into State 4 (11%). States 3 and 4 (6.0% and 7.6%, both ~1-ball dwell) form a high-risk pocket: **State 3 carries Wicket 14% + Chance 9% = 23% combined threat**, and always transitions straight into **State 4 — his leakiest state by far (Dot 38%, the lowest of any state) — which carries Wicket 14% + Chance 10% = 24%, the highest combined threat for Starc**, with 3 of his 7 recorded chances. State 2 (2.9%, ~10.0-ball dwell, 100% O0 Length out-swinger) carries a 9% wicket rate but is by far his smallest, rarest-occupied state. *Caveat*: as in `analysis.md`, O0 (Length, out-swinger) contains a handful of deliveries with an anomalous (physically implausible) positive Drop Angle reading — a Hawk-Eye tracking artefact, not a genuine delivery type — and these sit inside State 2.

### PJ Cummins

| | State 0 (28.7%, ~1.2 balls) | State 1 (15.8%, ~1.0 ball) | State 2 (13.4%, ~1.0 ball) | State 3 (31.3%, ~1.0 ball) | State 4 (10.8%, ~1.1 balls) |
|---|---|---|---|---|---|
| Dominant clusters | O0 Good swinger (29%), O1 Angle-in (64%), O2 Bouncer (8%) | O0 Good swinger (60%), O1 Angle-in (3%), O2 Bouncer (37%) | O0 Good swinger (94%), O2 Bouncer (4%), R0 Round-the-wicket bouncer (2%) | O0 Good swinger (33%), O1 Angle-in (61%), O2 Bouncer (6%) | O0 Good swinger (51%), O1 Angle-in (45%), R0 Round-the-wicket bouncer (5%) |
| Outcome | Dot 77%, Runs 20%, **Chance 3%** (+3), Wicket 0% | Dot 61%, Runs 39%, Wicket 0% | Dot 76%, Runs 20%, Chance 2% (+1), **Wicket 2%** | Dot 84%, Runs 12%, **Wicket 4%** | **Dot 0%, Runs 98%**, Wicket 2% |
| Starts an innings in this state | 0 / 24 | 4 / 24 | 0 / 24 | 20 / 24 | 0 / 24 |

Cummins' picture has changed substantially under the new Chance definition: every state now has an expected dwell of roughly **one ball**, so the model decodes him as switching mode almost every delivery, with his overall mix governed by which states he cycles between rather than long settled spells. **State 3 (31.3%, his tightest state at Dot 84% and his highest "converted" wicket rate at 4%) is overwhelmingly his starting state (20/24 innings)** — his default opening gambit. State 0 (28.7%, swinger/angle-in mix, almost as large) is the state where he most often beats the bat without converting: **3 of his 4 recorded chances, a 3% chance rate**. The standout new finding is **State 4 (10.8%, swinger/angle-in mix): Dot 0%, Runs 98%, Wicket 2% — essentially "the ball that goes for runs"**, occurring in short ~1.1-ball bursts rather than as a sustained mode. State 1 (15.8%, swinger/bouncer mix) is his leakiest by Dot rate (61%) but carries no wickets or chances. State 2 (13.4%, near-pure good-swinger at 94%) sits in the middle with a modest 2% wicket rate plus 1 chance. The transition matrix shows two loosely-linked loops: State 3 and State 0 mostly feed each other (State 3 → State 0 77%, State 0 → State 3 87%) — his "default" pairing — while State 1 → State 2 (89%) → State 4 (52%) → State 1 (63%) forms a secondary cycle that occasionally branches off from the main pairing.

---

## How does the strategy mix change innings-by-innings?

With each innings treated as its own sequence, this plot is a direct readout of the model's per-innings behaviour rather than an artefact of one long concatenated decode.

![Hidden-state mix per innings](hmm_innings_states.png)

Each bar is one innings (in chronological order, left to right), showing what fraction of that bowler's deliveries in that innings were decoded as State 0 (blue), State 1 (orange), State 2 (red), State 3 (purple) or State 4 (green). The number above each bar is how many deliveries that bowler bowled at this batter in that innings — many innings are single-figure samples, so individual bars should be read cautiously, but the overall pattern across many innings is informative.

- **M Morkel**: most innings show a mix of several states, with State 2 (over-the-wicket, red) and State 4 (his settled, highest-threat round-the-wicket state, green) both appearing as sizeable components in many innings rather than separate innings-level modes — a few innings are decoded as almost entirely State 2 (e.g. innings 4 and 9) or almost entirely State 4 (innings 3 and 8), but most mix several states within the innings, consistent with **wicket-position choice being a within-innings tactical variable** for Morkel.
- **SCJ Broad**: a handful of innings (e.g. 0, 3, 12, 26 and the last innings) are decoded as **almost entirely State 3 (purple, his absorbing high-chance/no-wicket mode)** — exactly as expected from a self-transition probability of 1.0: once he goes there for an innings, he never leaves. Most other innings show a mix of States 0/2/4 (blue/red/green), with State 4 (his highest-threat state, 13% wicket + 7% chance) a recurring, often-large component — and one striking innings (around 27) is decoded as **almost entirely State 1 (orange, his long swing-away containment mode)**.
- **MA Starc**: many innings are decoded as **almost entirely State 1 (orange, his largest, near-absorbing broad mix)** or **almost entirely State 0 (blue, the tight in-swinger-length opening)**, echoing how dominant these two states are as starting states (26/28 innings combined). States 3/4 (purple/green, his high-threat pocket) appear as small slices in several innings, and **one short innings (around innings 9, 6 balls) is decoded almost entirely as State 2 (red, the rare 100%-O0 state)** — exactly the kind of innings a 2-state model would have folded into "the broad mix" without comment.
- **PJ Cummins**: State 3 (purple, his tightest/most-converting default) and State 0 (blue, his main "edges without converting" state) are present in most innings as the two largest components, consistent with them being his two largest-occupancy states overall and State 3 dominating innings starts (20/24). State 4 (green, the "always goes for runs" state) appears as a modest slice in many innings rather than concentrated in a few — consistent with its short ~1.1-ball dwell meaning it crops up briefly throughout, rather than defining whole innings.

---

## Home vs away

The dataset records whether the bowling team was playing at home, away, or (rarely) on neutral ground. This field is blank for all the England-vs-Australia (Ashes) matches, so for those we derive it from `Ground Country`: a ground in England or Wales counts as "home" for England and "away" for Australia, and vice versa for grounds in Australia.

![State mix and outcomes: home vs away](hmm_home_away.png)

The top row shows the State 0–4 split for home vs away deliveries; the bottom row shows the overall wicket rate for home vs away (not split by state, since the per-state-per-venue samples get very small).

- **Wicket rate is higher at home for three of the four bowlers** (Morkel 2.3% vs 1.3%, Broad 3.3% vs 2.5%, Starc 4.0% vs 2.7%; Cummins is the exception at 1.4% home vs 2.1% away). This matches the general expectation that home conditions and crowd support tend to favour the bowling side.
- **Morkel's home sample (n=43, essentially one innings) leans heavily on the round-the-wicket transient pair, States 1 and 3** (30% + 33% = 63% home vs 18% + 20% = 38% away), and away spends much more time in the over-the-wicket State 2 (30% vs 9% home). State 4 (his highest-threat settled state, 7%) is similar in both (19% home vs 17% away), so it isn't what's driving the gap — and with only n=43 at home, this should be read cautiously.
- **Broad's highest-threat state (State 4, 13% wicket + 7% chance) is more than twice as common at home (21%) than away (9%)**, while the absorbing, wicket-free-but-chance-heavy State 3 is much *less* common at home (8% vs 19% away) — both point in the direction of a higher home wicket rate, consistent with what's observed (3.3% vs 2.5%). Away also leans heavily on State 0 (56% vs 27% home), his largest but comparatively tame state.
- **Starc's home deliveries lean heavily towards State 0 (45% vs 4% away)** — his tight, no-chance/no-wicket opening mode — but home also carries more of his high-threat pocket (States 2/3/4 combined: 5%+7%+9%=21% home vs 0%+5%+5%=10% away). Away is dominated by State 1 (86% vs 33% home, his largest, low-threat broad mix). So the higher home wicket rate (4.0% vs 2.7%) seems to come from spending more time in the small high-threat states (2/3/4) rather than from State 0 itself.
- **Cummins' state mix is fairly similar home and away** (State 3, his tightest/default state, is 30% home vs 32% away; State 0, his "edges without converting" state, is 27% home vs 30% away) — but State 4, the "always goes for runs" state, is somewhat more common away (12% vs 8% home), and his *away* wicket rate is higher (2.1% vs 1.4%). With a broadly similar state mix, the away wicket-rate edge looks like it comes mostly from within-state differences rather than a wholesale change of "strategy".

---

## Caveats

- **5 states is a deliberate over-fit, by choice.** BIC prefers 2 states for every bowler, by a wide margin (e.g. Morkel: 1657.0 at 2 states vs 2089.0 at 5). The 2-state model is the statistically "safer" choice, but it mostly just recovers the over/round-the-wicket split. 5 states was chosen here to dig for finer structure underneath that split — some of it (e.g. the high-threat states for Morkel, Broad, Starc and Cummins) is genuinely interesting, but it should be treated as **hypothesis-generating, not confirmed** — especially the transient states with ~1-ball expected dwell times, which a model with this many parameters can fit to noise.
- **Sample size**: 350–560 balls per bowler, split across 15–35 innings, is small for HMM fitting — many innings contribute only a handful of observations to the per-innings decoding, and this gets more acute the more states are added.
- **The "strategies reset each innings" assumption is itself a modelling choice**, not a certainty — international bowlers do carry plans and form across a match or series (e.g. "this batter struggled against X last time, do it again"). Treating innings as fully independent is a reasonable simplification, but innings-to-innings continuity is not literally zero.
- **The states are descriptive, not causal.** The HMM finds statistical regimes in the sequence; it doesn't know *why* the regime changed (new ball, declaration tactics, a different batter on strike, fatigue, etc.).
- **Label switching**: hidden states from EM have no inherent ordering. States were sorted by an "aggression score" (wicket rate ×10 + runs rate) purely so State 0...State 4 mean roughly the same thing (low to high aggression) across bowlers, but the absolute values are not directly comparable across bowlers (each HMM was fit independently).
- **Shared clustering with `analysis.md`**: the ball clusters feeding the HMM are identical to the over/round-the-wicket-split clusters described and named in `analysis.md`, so cluster names (O0, R1, ...) mean the same thing in both documents.
- **Chances**: deliveries where `Ball Events` records an edge or a catch chance, but no wicket, are their own outcome category alongside Dot/Runs/Wicket (5–15 per bowler, 1.1%–3.9% of deliveries). This is common enough to noticeably shift some states' fitted emission probabilities (e.g. Broad's State 3 and State 4, Starc's States 3/4), but the underlying counts are still small (single digits per state), so the *exact* percentages should be read as indicative rather than precise — the per-state "+N chance" counts are the more reliable number.
- **Starc's O0 (Length, out-swinger)** contains a small number of deliveries with an anomalous (physically implausible) positive Drop Angle — a Hawk-Eye tracking artefact flagged in `analysis.md`, not a genuine delivery type. It now sits inside State 2 (2.9% of deliveries) and shouldn't be over-interpreted.
