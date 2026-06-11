"""Shared delivery-type clustering definitions, used by both analysis.py
(the EDA / cluster description) and hmm_analysis.py (the strategy model),
so that "cluster C" means the same thing in every figure and document.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Nine delivery-property features, bat-hand adjusted where relevant.
features = [
    "Release Speed",
    "Bounce Length",
    "Bounce Line (Bat Hand Adjusted)",
    "Swing (Deg) (Bat Hand Adjusted)",
    "Deviation (Deg) (Bat Hand Adjusted)",
    "Drop Angle (Deg)",
    "Bounce Angle (Deg)",
    "Release Width",
]

radar_labels = [
    "Speed", "Length", "Line\n(+ leg)", "Swing\n(+ in)",
    "Deviation\n(+ in)", "Drop\nAngle", "Bounce\nAngle",
    "Release\nWidth",
]

col_labels = [
    'Speed\n(mph)', 'Length\n(m)', 'Line\n(bat adj)', 'Swing°\n(bat adj)',
    'Dev°\n(bat adj)', 'Drop\nAngle°', 'Bounce\nAngle°', 'Release\nWidth',
]

# Over/round-the-wicket split, then per-side k-means. Release Width sign
# marks which side of the stumps the ball was released from. "Over" = the
# side each bowler used for the majority of their deliveries (their stock
# position); "Round" = the minority side. k per side chosen via silhouette
# score (see analysis.py). Morkel and Broad bowl to a left-handed batter, so
# the over/round labelling of Release Width sign is flipped relative to
# Starc and Cummins (right-handed batter).
side_config = {
    "M Morkel":   {"Over": ("neg", 2), "Round": ("pos", 4)},
    "SCJ Broad":  {"Over": ("neg", 2), "Round": ("pos", 3)},
    "MA Starc":   {"Over": ("pos", 5), "Round": ("neg", 1)},
    "PJ Cummins": {"Over": ("neg", 3), "Round": ("pos", 1)},
}

# Expert-provided names for each (bowler, side, local cluster index).
expert_labels = {
    "M Morkel": {
        ("Over", 0): "Short", ("Over", 1): "Good length",
        ("Round", 0): "In-swinger", ("Round", 1): "Full outside off",
        ("Round", 2): "Out-swinger", ("Round", 3): "Bouncer",
    },
    "SCJ Broad": {
        ("Over", 0): "Short ball/bouncer", ("Over", 1): "Good length, angle across",
        ("Round", 0): "Back of length, nip-away", ("Round", 1): "Full and straight",
        ("Round", 2): "Good length, big swing away",
    },
    "MA Starc": {
        ("Over", 0): "Length, out-swinger", ("Over", 1): "Length, seam in",
        ("Over", 2): "Length, in-swinger", ("Over", 3): "Full in-swinger",
        ("Over", 4): "Bouncer", ("Round", 0): "Round-the-wicket length",
    },
    "PJ Cummins": {
        ("Over", 0): "Good swinger", ("Over", 1): "Angle-in",
        ("Over", 2): "Bouncer", ("Round", 0): "Round-the-wicket bouncer",
    },
}


def side_subset(sub, sign):
    return sub[sub["Release Width"] > 0] if sign == "pos" else sub[sub["Release Width"] <= 0]


def total_k(bowler):
    """Total number of clusters across both sides (used as the HMM alphabet size)."""
    return sum(k for _, k in side_config[bowler].values())


def cluster_names(bowler):
    """Map global cluster index (0..total_k-1) -> (Label, ExpertName), e.g.
    0 -> ("O0", "Short"). Global indices run through Over's clusters first,
    then Round's, in side_config order."""
    names = {}
    offset = 0
    for side_name, (sign, k) in side_config[bowler].items():
        for c in range(k):
            names[offset + c] = (f"{side_name[0]}{c}", expert_labels[bowler][(side_name, c)])
        offset += k
    return names


def build_clusters(df_clean, bowler):
    """Return the bowler's deliveries (all original df_clean columns retained)
    with added Side / Cluster (local, per side) / Label (e.g. "O0") /
    GlobalCluster (0..total_k-1) / ExpertName columns."""
    sub_all = df_clean[df_clean["Bowler Name"] == bowler].copy()
    pieces = []
    offset = 0
    for side_name, (sign, k) in side_config[bowler].items():
        side_sub = side_subset(sub_all, sign).copy()
        if k == 1:
            side_sub["Cluster"] = 0
        else:
            X = StandardScaler().fit_transform(side_sub[features])
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            side_sub["Cluster"] = km.fit_predict(X)
        side_sub["Side"] = side_name
        side_sub["Label"] = side_name[0] + side_sub["Cluster"].astype(str)
        side_sub["GlobalCluster"] = offset + side_sub["Cluster"]
        side_sub["ExpertName"] = side_sub["Cluster"].map(
            lambda c, sn=side_name: expert_labels[bowler][(sn, c)]
        )
        offset += k
        pieces.append(side_sub)
    return pd.concat(pieces).sort_index()
