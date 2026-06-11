import pandas as pd
import numpy as np
from hmmlearn.hmm import CategoricalHMM
import matplotlib.pyplot as plt

import cluster_defs as CD

df = pd.read_excel("Test_Wicket_Match_Ups.xlsx")

# ── Same per-bowler delivery-type clusters as analysis.py ────────────────────
# The "ball cluster" feeding the HMM is the *same* over/round-split clustering
# used for the delivery-type description in analysis.md (see cluster_defs.py),
# so cluster names/labels (O0, R1, "In-swinger", ...) mean the same thing in
# both documents. The alphabet size per bowler is the total cluster count
# across both sides.
features = CD.features
best_k = {bowler: CD.total_k(bowler) for bowler in CD.side_config}

extra_cols = [
    "Bowler Name", "Match ID", "Match Start Date", "Innings ID", "Over No", "Ball No",
    "Is Wicket", "Bowler Runs Conceded", "Home/Away", "Fielder Action",
]

# ── Home/away: fill gaps in the Ashes matches via Ground Country ─────────────
# Team 1/2 Home/Away is blank for all England-vs-Australia matches, even
# though Ground Country is populated. For these, the bowling team is "Home"
# if the ground is in their own country (England/Wales count as home for
# England, per the 2009 Cardiff Ashes Test).
def bowling_home_away(row):
    if row["Bowling Team Name"] == row["Team 1 Name"]:
        ha = row["Team 1 Home/Away"]
    else:
        ha = row["Team 2 Home/Away"]
    if pd.notna(ha):
        return ha
    team, ground = row["Bowling Team Name"], row["Ground Country"]
    if team == "England" and ground in ("England", "Wales"):
        return "Home"
    if team == ground:
        return "Home"
    return "Away"

df["Home/Away"] = df.apply(bowling_home_away, axis=1)

df_clean = df[features + extra_cols].dropna(subset=features).copy()
df_clean = df_clean[df_clean["Release Speed"] >= 75].copy()

# ── Outcome category: 0=Dot, 1=Runs off bat, 2=Chance, 3=Wicket ──────────────
# A "chance" is a delivery where a wicket should have happened but didn't -
# a dropped catch, missed run-out or keeper error (Is Wicket is False for
# all of these). These are rare (0-4 per bowler) but worth distinguishing
# from an ordinary non-wicket ball.
chance_actions = {"Dropped Catch", "Run Out Chance", "Keeper Error"}

def outcome_cat(row):
    if row["Is Wicket"]:
        return 3
    if row["Fielder Action"] in chance_actions:
        return 2
    if row["Bowler Runs Conceded"] == 0:
        return 0
    return 1

df_clean["Outcome"] = df_clean.apply(outcome_cat, axis=1)
outcome_labels = ["Dot", "Runs", "Chance", "Wicket"]

# ── Per-bowler: cluster, order chronologically, fit HMM ──────────────────────
results = {}

for bowler, k in best_k.items():
    # Delivery-type cluster: same over/round-split clustering as analysis.py,
    # giving a single global cluster index 0..k-1 (see cluster_defs.py).
    sub = CD.build_clusters(df_clean, bowler)
    sub["Cluster"] = sub["GlobalCluster"]

    # Chronological order
    sub = sub.sort_values(["Match Start Date", "Match ID", "Innings ID", "Over No", "Ball No"])
    sub = sub.reset_index(drop=True)

    # Combined observed symbol = cluster * 4 + outcome
    n_outcomes = 4
    sub["Symbol"] = sub["Cluster"] * n_outcomes + sub["Outcome"]
    n_symbols = k * n_outcomes
    obs = sub["Symbol"].to_numpy().reshape(-1, 1)
    n_obs = len(obs)

    # Each innings is treated as its own independent sequence: the HMM resets
    # to its start distribution at the start of every innings, and no
    # transition is modelled across the boundary between two innings (whether
    # within the same match or across matches). Innings are contiguous blocks
    # in `sub` because it is sorted chronologically.
    lengths = sub.groupby(["Match ID", "Innings ID"], sort=False).size().to_numpy()
    assert lengths.sum() == n_obs

    # ── Choose number of hidden "strategy" states ─────────────────────────────
    # BIC is computed for n_states = 2..5 for reference, but we deliberately
    # use n_states=5 ("five strategies") regardless of what BIC prefers — BIC
    # consistently favours 2 states on this sample size, but 2 states mostly
    # just recovers the over/round-the-wicket split (see hmm_analysis.md
    # caveats). 5 states is used here to look for finer-grained strategy
    # structure within each side, at the cost of some states being supported
    # by very few deliveries.
    FORCE_N_STATES = 5
    bic_scores = {}
    fitted_models = {}
    for n_states in [2, 3, 4, 5]:
        best_ll, best_model = -np.inf, None
        for seed in range(15):
            model = CategoricalHMM(
                n_components=n_states, n_features=n_symbols,
                random_state=seed, n_iter=300, tol=1e-4,
            )
            model.fit(obs, lengths)
            ll = model.score(obs, lengths)
            if ll > best_ll:
                best_ll, best_model = ll, model
        n_params = (n_states - 1) + n_states * (n_states - 1) + n_states * (n_symbols - 1)
        bic = -2 * best_ll + n_params * np.log(n_obs)
        bic_scores[n_states] = bic
        fitted_models[n_states] = (best_model, best_ll)

    chosen_n = FORCE_N_STATES
    model, ll = fitted_models[chosen_n]

    # ── Decode hidden states (per-match, via lengths), relabel by "aggression score" ──
    states_raw = model.predict(obs, lengths)
    emission = model.emissionprob_  # (n_states, n_symbols)

    p_outcome_given_state = np.zeros((chosen_n, n_outcomes))
    p_cluster_given_state = np.zeros((chosen_n, k))
    for s in range(chosen_n):
        for sym in range(n_symbols):
            c, o = sym // n_outcomes, sym % n_outcomes
            p_cluster_given_state[s, c] += emission[s, sym]
            p_outcome_given_state[s, o] += emission[s, sym]

    aggression = (p_outcome_given_state[:, 3] * 10 + p_outcome_given_state[:, 2] * 5
                  + p_outcome_given_state[:, 1])
    order = np.argsort(aggression)
    relabel = {old: new for new, old in enumerate(order)}
    states = np.array([relabel[s] for s in states_raw])

    p_cluster_given_state = p_cluster_given_state[order]
    p_outcome_given_state = p_outcome_given_state[order]
    transmat = model.transmat_[np.ix_(order, order)]
    startprob = model.startprob_[order]

    sub["State"] = states

    print(f"\n{'='*60}\n{bowler} (n={n_obs}, k_clusters={k})\n{'='*60}")
    print("BIC by n_states:", {ns: round(b, 1) for ns, b in bic_scores.items()},
          f"-> using n_states={chosen_n} (fixed; BIC would prefer "
          f"{min(bic_scores, key=bic_scores.get)})")
    print("Transition matrix (relabelled, low->high aggression):")
    print(np.round(transmat, 3))
    print("Start probabilities:", np.round(startprob, 3))
    print("P(outcome | state):")
    for s in range(chosen_n):
        print(f"  State {s}: " +
              ", ".join(f"{outcome_labels[o]}={p_outcome_given_state[s,o]:.2f}" for o in range(n_outcomes)))
    print("P(cluster | state):")
    for s in range(chosen_n):
        print(f"  State {s}: " + ", ".join(f"C{c}={p_cluster_given_state[s,c]:.2f}" for c in range(k)))
    print("State occupancy (% of deliveries):")
    occ = sub["State"].value_counts(normalize=True).sort_index()
    for s, p in occ.items():
        dwell = 1 / (1 - transmat[s, s]) if transmat[s, s] < 1 else np.inf
        print(f"  State {s}: {p*100:.1f}%  (expected dwell ~{dwell:.1f} balls)")

    # Which state does each innings start in? (uses startprob_, fit per-innings)
    innings_starts = sub["State"].to_numpy()[np.cumsum(lengths) - lengths]
    print(f"Innings starting in each state: " +
          ", ".join(f"State {s}={np.sum(innings_starts==s)}" for s in range(chosen_n)) +
          f"  (out of {len(lengths)} innings)")

    results[bowler] = dict(
        sub=sub, k=k, chosen_n=chosen_n, bic_scores=bic_scores,
        transmat=transmat, startprob=startprob, lengths=lengths,
        p_outcome=p_outcome_given_state, p_cluster=p_cluster_given_state,
        ll=ll,
    )

# ── Figure 1: BIC vs n_states ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
fig.suptitle("BIC vs number of hidden strategy states (n_states=5 used despite BIC minimum)",
              fontsize=12, fontweight="bold")
for ax, bowler in zip(axes, best_k):
    bic_scores = results[bowler]["bic_scores"]
    ns = list(bic_scores.keys())
    vals = list(bic_scores.values())
    ax.plot(ns, vals, marker="o", color="steelblue")
    chosen = results[bowler]["chosen_n"]
    bic_best = min(bic_scores, key=bic_scores.get)
    ax.axvline(bic_best, color="seagreen", linestyle=":", alpha=0.6, label=f"BIC min n={bic_best}")
    ax.axvline(chosen, color="darkorange", linestyle="--", alpha=0.6, label=f"used n={chosen}")
    ax.set_title(bowler, fontweight="bold")
    ax.set_xlabel("n_states")
    ax.set_xticks(ns)
    if bowler == "M Morkel":
        ax.set_ylabel("BIC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("hmm_state_selection.png", dpi=150, bbox_inches="tight")
print("\nSaved hmm_state_selection.png")

# ── Figure 2: hidden state timelines ─────────────────────────────────────────
state_colors = ["#2196F3", "#FF9800", "#F44336", "#9C27B0", "#4CAF50"]

fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=False)
fig.suptitle("Decoded hidden 'strategy' state through each bowler's deliveries (chronological)",
              fontsize=12, fontweight="bold")
for ax, bowler in zip(axes, best_k):
    sub = results[bowler]["sub"]
    chosen_n = results[bowler]["chosen_n"]
    lengths = results[bowler]["lengths"]
    x = np.arange(len(sub))
    for s in range(chosen_n):
        mask = sub["State"].to_numpy() == s
        ax.scatter(x[mask], sub["State"].to_numpy()[mask], s=4, color=state_colors[s], label=f"State {s}")
    wkt_x = x[sub["Is Wicket"].to_numpy()]
    ax.scatter(wkt_x, sub["State"].to_numpy()[sub["Is Wicket"].to_numpy()],
               marker="*", s=120, color="black", zorder=5, label="Wicket")
    # Mark innings boundaries (each HMM sequence resets here)
    for boundary in np.cumsum(lengths)[:-1]:
        ax.axvline(boundary, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_title(bowler, fontweight="bold", loc="left", fontsize=10)
    ax.set_yticks(range(chosen_n))
    ax.set_ylabel("State")
    ax.set_xlim(0, len(sub))
    if bowler == "M Morkel":
        ax.legend(fontsize=8, loc="upper right", ncol=chosen_n + 1)
ax.set_xlabel("Delivery index (chronological)")
plt.tight_layout()
plt.savefig("hmm_timelines.png", dpi=150, bbox_inches="tight")
print("Saved hmm_timelines.png")

# ── Figure 3: emission profiles P(cluster|state) and P(outcome|state) ────────
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
fig.suptitle("Emission profiles per hidden state", fontsize=13, fontweight="bold")
for col, bowler in enumerate(best_k):
    res = results[bowler]
    chosen_n, k = res["chosen_n"], res["k"]

    cnames = CD.cluster_names(bowler)
    short_labels = [cnames[c][0] for c in range(k)]

    ax = axes[0, col]
    im = ax.imshow(res["p_cluster"], cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_yticks(range(chosen_n))
    ax.set_yticklabels([f"State {s}" for s in range(chosen_n)], fontsize=8)
    ax.set_title(bowler, fontweight="bold", fontsize=10)
    for s in range(chosen_n):
        for c in range(k):
            v = res["p_cluster"][s, c]
            ax.text(c, s, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 else "black")
    if col == 0:
        ax.set_ylabel("P(ball cluster | state)")

    ax2 = axes[1, col]
    im2 = ax2.imshow(res["p_outcome"], cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(len(outcome_labels)))
    ax2.set_xticklabels(outcome_labels, fontsize=8)
    ax2.set_yticks(range(chosen_n))
    ax2.set_yticklabels([f"State {s}" for s in range(chosen_n)], fontsize=8)
    for s in range(chosen_n):
        for o in range(len(outcome_labels)):
            v = res["p_outcome"][s, o]
            ax2.text(o, s, f"{v:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if v > 0.6 else "black")
    if col == 0:
        ax2.set_ylabel("P(outcome | state)")

plt.tight_layout()
plt.savefig("hmm_emissions.png", dpi=150, bbox_inches="tight")
print("Saved hmm_emissions.png")

# ── Figure 4: histogram of delivery (cluster) types per decoded state ────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)
fig.suptitle("Delivery types bowled within each decoded hidden state",
              fontsize=13, fontweight="bold")
for ax, bowler in zip(axes, best_k):
    res = results[bowler]
    sub, chosen_n, k = res["sub"], res["chosen_n"], res["k"]
    cnames = CD.cluster_names(bowler)
    short_labels = [cnames[c][0] for c in range(k)]

    counts = (
        sub.groupby(["State", "Cluster"]).size()
        .unstack(fill_value=0)
        .reindex(columns=range(k), fill_value=0)
    )
    proportions = counts.div(counts.sum(axis=1), axis=0)

    bar_width = 0.8 / chosen_n
    x = np.arange(k)
    for s in range(chosen_n):
        ax.bar(x + s * bar_width, proportions.loc[s], width=bar_width,
               color=state_colors[s], label=f"State {s} (n={counts.loc[s].sum()})")

    ax.set_xticks(x + bar_width * (chosen_n - 1) / 2)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_title(bowler, fontweight="bold", fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    if bowler == "M Morkel":
        ax.set_ylabel("Share of state's deliveries")

plt.tight_layout()
plt.savefig("hmm_cluster_histograms.png", dpi=150, bbox_inches="tight")
print("Saved hmm_cluster_histograms.png")

# ── Figure 5: state mix per innings, in chronological order ──────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
fig.suptitle("Hidden-state mix per innings (chronological)", fontsize=13, fontweight="bold")
for ax, bowler in zip(axes, best_k):
    res = results[bowler]
    sub, chosen_n = res["sub"], res["chosen_n"]

    # sub is already sorted chronologically; preserve that innings order
    innings_order = sub[["Match ID", "Innings ID"]].drop_duplicates().to_numpy()
    innings_index = {tuple(mi): i for i, mi in enumerate(innings_order)}
    sub["InningsIdx"] = list(
        map(innings_index.get, zip(sub["Match ID"], sub["Innings ID"]))
    )

    counts = (
        sub.groupby(["InningsIdx", "State"]).size()
        .unstack(fill_value=0)
        .reindex(columns=range(chosen_n), fill_value=0)
        .reindex(range(len(innings_order)), fill_value=0)
    )
    proportions = counts.div(counts.sum(axis=1), axis=0)
    n_balls = counts.sum(axis=1)

    bottom = np.zeros(len(innings_order))
    for s in range(chosen_n):
        ax.bar(proportions.index, proportions[s], bottom=bottom, width=0.85,
               color=state_colors[s], label=f"State {s}")
        bottom += proportions[s].to_numpy()

    # annotate number of balls bowled in each innings
    for i, n in enumerate(n_balls):
        if n > 0:
            ax.text(i, 1.02, str(n), ha="center", va="bottom", fontsize=6, color="gray")

    ax.set_title(bowler, fontweight="bold", loc="left", fontsize=10)
    ax.set_xlim(-0.6, len(innings_order) - 0.4)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("State share")
    if bowler == "M Morkel":
        ax.legend(fontsize=8, loc="upper right", ncol=chosen_n)
axes[-1].set_xlabel("Innings number (chronological; label = balls bowled in that innings)")
plt.tight_layout()
plt.savefig("hmm_innings_states.png", dpi=150, bbox_inches="tight")
print("Saved hmm_innings_states.png")

# ── Figure 6: state mix and outcomes, home vs away ────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("Hidden-state mix and outcomes: home vs away", fontsize=13, fontweight="bold")
for col, bowler in enumerate(best_k):
    res = results[bowler]
    sub, chosen_n = res["sub"], res["chosen_n"]

    # -- Top row: state share by home/away --
    ax = axes[0, col]
    state_counts = (
        sub.groupby(["Home/Away", "State"]).size()
        .unstack(fill_value=0)
        .reindex(index=["Home", "Away"], fill_value=0)
        .reindex(columns=range(chosen_n), fill_value=0)
    )
    state_props = state_counts.div(state_counts.sum(axis=1), axis=0)
    n_per_group = state_counts.sum(axis=1)

    bottom = np.zeros(2)
    for s in range(chosen_n):
        ax.bar(state_props.index, state_props[s], bottom=bottom, width=0.6,
               color=state_colors[s], label=f"State {s}")
        bottom += state_props[s].to_numpy()
    for i, n in enumerate(n_per_group):
        ax.text(i, 1.02, f"n={n}", ha="center", va="bottom", fontsize=8, color="gray")
    ax.set_title(bowler, fontweight="bold", fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.5, 1])
    if col == 0:
        ax.set_ylabel("State share")
        ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.08))

    # -- Bottom row: wicket rate by home/away (overall, not per-state) --
    ax2 = axes[1, col]
    wkt_rate = sub.groupby("Home/Away")["Is Wicket"].mean().reindex(["Home", "Away"], fill_value=0)
    ax2.bar(wkt_rate.index, wkt_rate.values * 100, width=0.6, color="#888888")
    for i, v in enumerate(wkt_rate.values):
        ax2.text(i, v * 100, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_ylim(0, wkt_rate.values.max() * 100 * 1.4 + 1)
    if col == 0:
        ax2.set_ylabel("Wicket rate (%)")

plt.tight_layout()
plt.savefig("hmm_home_away.png", dpi=150, bbox_inches="tight")
print("Saved hmm_home_away.png")

print("\n=== Home vs away breakdown ===")
for bowler in best_k:
    sub = results[bowler]["sub"]
    chosen_n = results[bowler]["chosen_n"]
    print(f"\n{bowler}:")
    state_props = (
        sub.groupby(["Home/Away", "State"]).size().unstack(fill_value=0)
        .reindex(index=["Home", "Away"], fill_value=0)
        .reindex(columns=range(chosen_n), fill_value=0)
    )
    state_props = state_props.div(state_props.sum(axis=1), axis=0)
    wkt_rate = sub.groupby("Home/Away")["Is Wicket"].mean().reindex(["Home", "Away"], fill_value=0)
    n = sub.groupby("Home/Away").size().reindex(["Home", "Away"], fill_value=0)
    for venue in ["Home", "Away"]:
        states_str = ", ".join(f"State {s}={state_props.loc[venue, s]*100:.0f}%" for s in range(chosen_n))
        print(f"  {venue} (n={n[venue]}): {states_str}, wicket rate={wkt_rate[venue]*100:.1f}%")
