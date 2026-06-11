import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi

import cluster_defs as CD

df = pd.read_excel("../data/Test_Wicket_Match_Ups.xlsx")

# ── EDA ──────────────────────────────────────────────────────────────────────

print("=== Dataset shape ===")
print(df.shape)

print("\n=== Match-up structure ===")
print("Unique matches:", df["Match ID"].nunique())
print("Bowler deliveries:")
print(df["Bowler Name"].value_counts())
print("Batter deliveries:")
print(df["Batter Name"].value_counts())

print("\n=== Bowling Length ===")
print(df["Bowling Length"].value_counts())

print("\n=== Bowling Line ===")
print(df["Bowling Line"].value_counts())

print("\n=== Bowling Movement ===")
print(df["Bowling Movement"].value_counts())

print("\n=== Ball Age stats ===")
print(df["Ball Age (Balls)"].describe())

print("\n=== Wicket summary ===")
print("Total wickets:", df["Is Wicket"].sum())
wickets = df[df["Is Wicket"] == True]
print("By bowler:")
print(wickets["Bowler Name"].value_counts())
print("How out:")
print(wickets["How Out"].value_counts())
print("Bowling Length:")
print(wickets["Bowling Length"].value_counts())

# ── Per-bowler clustering ─────────────────────────────────────────────────────

# Nine delivery-property features, bat-hand adjusted where relevant.
# Excluded: stumps height (trajectory endpoint); release angle (corr 0.56 with
# drop angle); speed loss (corr 0.57 with bounce angle); release height
# (std 0.09 m — negligible variation); landing X/Y (wides only); intercept
# metrics (43% coverage).
features = CD.features
radar_labels = CD.radar_labels

df_clean = df[
    features
    + [
        "Bowler Name", "Batter Name", "Bowling Length", "Bowling Line",
        "Bowling Movement", "Is Wicket", "How Out", "Shot Type",
        "Bowler Style", "Ball Age (Balls)",
    ]
].dropna(subset=features).copy()

# Remove 7 tracking errors (release speed < 75 mph)
df_clean = df_clean[df_clean["Release Speed"] >= 75].copy()
print(f"\n=== Rows available for clustering: {len(df_clean)} / {len(df)} ===")

# ── Over/round-the-wicket split, then per-side clustering ────────────────────
# Release Width sign marks which side of the stumps the ball was released
# from (over vs. round the wicket). Splitting on this *before* clustering
# guarantees every resulting cluster is homogeneous in this dimension — no
# cluster mixes over- and round-the-wicket deliveries. "Over" = the side each
# bowler used for the majority of their deliveries (their stock position);
# "Round" = the minority side. k per side is chosen via silhouette score, or
# fixed at 1 if a side is too small to subdivide (Starc round=8, Cummins
# round=3 — both already flagged as anomalies/variations in earlier passes).
# Note: Morkel and Broad bowl to a left-handed batter, so the over/round
# labelling of Release Width sign is flipped relative to Starc and Cummins
# (who bowl to a right-handed batter).
# side_config and expert_labels live in cluster_defs.py, shared with
# hmm_analysis.py so that "cluster C" means the same thing everywhere.
side_config = CD.side_config
expert_labels = CD.expert_labels
side_subset = CD.side_subset

print("\n=== Silhouette scores per bowler/side ===")
for bowler, sides in side_config.items():
    sub = df_clean[df_clean["Bowler Name"] == bowler]
    print(f"\n{bowler}:")
    for side_name, (sign, chosen_k) in sides.items():
        side_sub = side_subset(sub, sign)
        print(f"  {side_name} ({sign}, n={len(side_sub)}):")
        if len(side_sub) < 10:
            print("    too small to subdivide -> k=1")
            continue
        X = StandardScaler().fit_transform(side_sub[features])
        for k in range(2, 6):
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)
            marker = " <-- chosen" if k == chosen_k else ""
            print(f"    k={k}: {sil:.4f}{marker}")

print("\n=== Generating cluster selection plots ===")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle('Choosing k per bowler/side (silhouette)', fontsize=14, fontweight='bold', y=1.01)

for col, (bowler, sides) in enumerate(side_config.items()):
    sub = df_clean[df_clean["Bowler Name"] == bowler]
    for row, (side_name, (sign, chosen_k)) in enumerate(sides.items()):
        ax = axes[row, col]
        side_sub = side_subset(sub, sign)
        if row == 0:
            ax.set_title(bowler, fontweight="bold")
        if len(side_sub) < 10:
            ax.text(0.5, 0.5, f"{side_name}\nn={len(side_sub)}\ntoo small to subdivide\n(k=1)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        X = StandardScaler().fit_transform(side_sub[features])
        k_range = range(2, 6)
        sils = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = km.fit_predict(X)
            sils.append(silhouette_score(X, labels))
        ax.plot(list(k_range), sils, marker="o", color="darkorange")
        ax.set_xlabel("k")
        ax.set_xticks(list(k_range))
        ax.grid(True, alpha=0.3)
        ax.axvline(chosen_k, color="darkorange", linestyle="--", alpha=0.5, label=f"chosen k={chosen_k}")
        ax.legend(fontsize=8)
        ax.set_ylabel(f"{side_name} (n={len(side_sub)})\nSilhouette")

plt.tight_layout()
plt.savefig("cluster_selection.png", dpi=150, bbox_inches="tight")
print("Saved cluster_selection.png")


def build_clusters(bowler):
    """Return the bowler's deliveries with Side/Cluster/Label/GlobalCluster/
    ExpertName columns (see cluster_defs.build_clusters)."""
    return CD.build_clusters(df_clean, bowler)


print("\n\n=== Per-bowler cluster results ===")
for bowler, sides in side_config.items():
    sub = build_clusters(bowler)
    print(f"\n{'='*60}")
    print(f"{bowler}  (Over k={sides['Over'][1]}, Round k={sides['Round'][1]})")
    print("=" * 60)

    for side_name, (sign, k) in sides.items():
        for c in range(k):
            csub = sub[(sub["Side"] == side_name) & (sub["Cluster"] == c)]
            label = side_name[0] + str(c)
            w = csub["Is Wicket"].sum()
            print(f"\n  -- {side_name} {label} (n={len(csub)}, wickets={w}, {w/len(csub)*100:.1f}%) --")
            print("  Centroid:")
            print(csub[features].mean().round(2).to_string())
            print("  Bowling Length:")
            print(csub["Bowling Length"].value_counts().head(4).to_string())
            print("  Bowling Movement:")
            print(csub["Bowling Movement"].value_counts().head(4).to_string())
            print(f"  Mean ball age: {csub['Ball Age (Balls)'].mean():.0f} balls")
            if w > 0:
                print("  How Out:")
                print(csub[csub["Is Wicket"] == True]["How Out"].value_counts().to_string())
                print("  Shot Type:")
                print(csub[csub["Is Wicket"] == True]["Shot Type"].value_counts().to_string())

# ── Radar charts ──────────────────────────────────────────────────────────────
print("\n=== Generating radar charts ===")

mms = MinMaxScaler()
mms.fit(df_clean[features].values)

colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#795548", "#607D8B"]
n_feat = len(features)
angles = [n / n_feat * 2 * pi for n in range(n_feat)] + [0]  # close polygon

fig, axes = plt.subplots(1, 4, figsize=(18, 5), subplot_kw=dict(polar=True))
fig.suptitle("Cluster radar charts — normalised delivery properties", fontsize=13, fontweight="bold")

side_colors = {
    "Over": ["#1f77b4", "#2ca02c", "#17becf", "#3a86ff", "#06d6a0"],
    "Round": ["#e07a5f", "#9c27b0", "#f4a261", "#e63946"],
}

for ax, bowler in zip(axes, side_config):
    sub = build_clusters(bowler)

    ax.set_title(bowler, fontweight="bold", pad=15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, size=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.50", "0.75"], size=6)
    ax.grid(True, alpha=0.3)

    patches = []
    for side_name, (sign, k) in side_config[bowler].items():
        for c in range(k):
            csub = sub[(sub["Side"] == side_name) & (sub["Cluster"] == c)]
            w = csub["Is Wicket"].sum()
            wkt_pct = w / len(csub) * 100
            color = side_colors[side_name][c]
            centroid_norm = mms.transform(csub[features].mean().values.reshape(1, -1))[0]
            vals = list(centroid_norm) + [centroid_norm[0]]
            ax.plot(angles, vals, color=color, linewidth=2)
            ax.fill(angles, vals, color=color, alpha=0.12)
            cluster_name = expert_labels[bowler][(side_name, c)]
            label = f"{side_name[0]}{c}: {cluster_name} (n={len(csub)}, {wkt_pct:.1f}% wkt)"
            patches.append(mpatches.Patch(color=color, label=label))
    ax.legend(handles=patches, loc="upper right",
              bbox_to_anchor=(1.45, 1.15), fontsize=7)

plt.tight_layout()
plt.savefig("radar_charts.png", dpi=150, bbox_inches="tight")
print("Saved radar_charts.png")

# ── Centroid heatmaps ─────────────────────────────────────────────────────────
print("\n=== Generating centroid heatmaps ===")

col_labels = [
    'Speed\n(mph)', 'Length\n(m)', 'Line\n(bat adj)', 'Swing°\n(bat adj)',
    'Dev°\n(bat adj)', 'Drop\nAngle°', 'Bounce\nAngle°', 'Release\nWidth',
]

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
fig.suptitle('Cluster centroids — per-feature heatmap (within-bowler z-score)',
             fontsize=13, fontweight='bold')

for ax, bowler in zip(axes, side_config):
    sub = build_clusters(bowler)
    bowler_scaler = StandardScaler().fit(sub[features])

    rows, row_labels = [], []
    for side_name, (sign, k) in side_config[bowler].items():
        for c in range(k):
            csub = sub[(sub["Side"] == side_name) & (sub["Cluster"] == c)]
            w = csub["Is Wicket"].sum()
            cluster_name = expert_labels[bowler][(side_name, c)]
            rows.append(csub[features].mean().values)
            row_labels.append(f"{side_name[0]}{c}: {cluster_name}  n={len(csub)}\n{w/len(csub)*100:.1f}% wkt")

    centroid_real = np.array(rows)
    centroid_z = bowler_scaler.transform(centroid_real)
    n_rows = len(rows)

    vmin, vmax = -2, 2
    im = ax.imshow(centroid_z, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=0, ha="center")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(bowler, fontweight="bold", fontsize=10)

    for i in range(n_rows):
        for j in range(len(features)):
            norm_val = (centroid_z[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            text_color = "white" if norm_val > 0.6 else "black"
            ax.text(j, i, f"{centroid_real[i, j]:.1f}", ha="center", va="center",
                    fontsize=6.5, color=text_color)

fig.colorbar(im, ax=axes[-1], label="Within-bowler z-score", shrink=0.8)
plt.tight_layout()
plt.savefig("centroid_heatmaps.png", dpi=150, bbox_inches="tight")
print("Saved centroid_heatmaps.png")
