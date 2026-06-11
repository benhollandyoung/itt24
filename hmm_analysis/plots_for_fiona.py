import sys
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kmeans_analysis"))
import cluster_defs as CD

# Reuses the data prep, clustering and per-innings HMM fits from hmm_analysis.py
# (importing re-runs that script, regenerating its figures too).
import hmm_analysis as H

INNINGS_PER_PAGE = 6
OUT_DIR = "../Plots for Fiona/With state colouring"
os.makedirs(OUT_DIR, exist_ok=True)

print("\n=== Generating per-innings delivery-type/state plots for Fiona ===")

for bowler, k in H.best_k.items():
    res = H.results[bowler]
    sub = res["sub"]
    chosen_n = res["chosen_n"]
    names = CD.cluster_names(bowler)
    y_labels = [f"{names[c][0]} {names[c][1]}" for c in range(k)]

    innings_order = sub[["Match ID", "Innings ID"]].drop_duplicates().to_numpy()
    n_innings = len(innings_order)
    n_pages = math.ceil(n_innings / INNINGS_PER_PAGE)

    fname = f"{OUT_DIR}/Plots for Fiona_{bowler}"
    with PdfPages(f"{fname}.pdf") as pdf:
        for p in range(n_pages):
            chunk = innings_order[p * INNINGS_PER_PAGE:(p + 1) * INNINGS_PER_PAGE]
            n_rows = len(chunk)

            fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.4 * n_rows + 1.4), squeeze=False)
            fig.suptitle(
                f"{bowler} — delivery type vs decoded state, ball-by-ball per innings (page {p + 1} of {n_pages})",
                fontsize=14, fontweight="bold", y=0.995,
            )
            state_patches = [mpatches.Patch(color=H.state_colors[s], label=f"State {s}")
                              for s in range(chosen_n)]
            fig.legend(handles=state_patches + [plt.Line2D([0], [0], marker="*", color="w",
                       markerfacecolor="black", markersize=12, label="Wicket")],
                       loc="upper right", ncol=chosen_n + 1, fontsize=9, bbox_to_anchor=(1.0, 0.99))

            for row, (m, inn) in enumerate(chunk):
                i = p * INNINGS_PER_PAGE + row
                ax = axes[row, 0]
                msub = sub[(sub["Match ID"] == m) & (sub["Innings ID"] == inn)].reset_index(drop=True)
                date = pd.to_datetime(msub["Match Start Date"].iloc[0]).date()

                xs = np.arange(len(msub))
                clusters = msub["Cluster"].to_numpy()
                states = msub["State"].to_numpy()
                for s in range(chosen_n):
                    mask = states == s
                    ax.scatter(xs[mask], clusters[mask], color=H.state_colors[s], s=30, zorder=3)

                wkt_mask = msub["Is Wicket"].to_numpy()
                if wkt_mask.any():
                    ax.scatter(xs[wkt_mask], clusters[wkt_mask], marker="*", s=120,
                               color="black", zorder=5)

                ax.set_yticks(range(k))
                ax.set_yticklabels(y_labels, fontsize=8)
                ax.set_ylim(-0.5, k - 0.5)
                ax.set_xlim(-0.5, max(len(msub) - 0.5, 0.5))
                ax.grid(axis="y", alpha=0.2)
                ax.set_title(f"Innings {i + 1} — {date} (Match ID {m}, Innings ID {inn}, n={len(msub)})",
                              fontsize=10, loc="left")
                if row == n_rows - 1:
                    ax.set_xlabel("Ball number in innings (chronological)", fontsize=10)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            if p == 0:
                plt.savefig(f"{fname}_page1.png", dpi=130, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved {fname}.pdf ({n_pages} pages) / preview png")
