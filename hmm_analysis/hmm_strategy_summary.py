import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from math import pi
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kmeans_analysis"))
import cluster_defs as CD

# Reuses the data prep, clustering and per-innings HMM fits from hmm_analysis.py
# (importing re-runs that script, regenerating its figures too).
import hmm_analysis as H

radar_labels = CD.radar_labels
col_labels = CD.col_labels
cluster_colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

mms = MinMaxScaler().fit(H.df_clean[H.features].values)
n_feat = len(H.features)
angles = [a / n_feat * 2 * pi for a in range(n_feat)] + [0]

INNINGS_PER_PAGE = 6

print("\n=== Generating per-bowler strategy summary PDFs ===")

for bowler, k in H.best_k.items():
    res = H.results[bowler]
    sub = res["sub"]
    chosen_n = res["chosen_n"]
    # Cluster 0..k-1 here is the same over/round-split clustering as
    # analysis.py (see cluster_defs.py), so the expert names apply directly.
    names = CD.cluster_names(bowler)  # {c: (label e.g. "O0", expert name)}
    label_of = lambda c: f"{names[c][0]} {names[c][1]}"
    tick_labels = [f"{names[c][0]}\n{names[c][1]}" for c in range(k)]
    # Short codes only, for the cramped per-innings bar-chart x-axis (page 2+);
    # full names are given in the page-1 legend/heatmap.
    short_labels = [names[c][0] for c in range(k)]

    centroids = sub.groupby("Cluster")[H.features].mean()
    sizes = sub.groupby("Cluster").size()
    wkt = sub.groupby("Cluster")["Is Wicket"].mean()

    innings_order = sub[["Match ID", "Innings ID"]].drop_duplicates().to_numpy()
    n_innings = len(innings_order)

    fname = f"hmm_summary_{bowler.replace(' ', '_')}"
    with PdfPages(f"{fname}.pdf") as pdf:

        # ── Page 1: cluster overview (radar + centroid heatmap) ──────────────
        fig1 = plt.figure(figsize=(14, 8))
        fig1.suptitle(
            f"{bowler} — delivery-type clusters",
            fontsize=17, fontweight="bold", y=0.98,
        )
        fig1.subplots_adjust(top=0.80, bottom=0.18, wspace=0.35)

        ax_radar = fig1.add_subplot(1, 2, 1, polar=True)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(radar_labels, size=10)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_yticks([0.25, 0.5, 0.75])
        ax_radar.set_yticklabels(["0.25", "0.50", "0.75"], size=8)
        ax_radar.grid(True, alpha=0.3)
        ax_radar.set_title("Cluster shapes (normalised)", fontsize=13, pad=18)

        patches = []
        for c in range(k):
            norm = mms.transform(centroids.loc[c].values.reshape(1, -1))[0]
            vals = list(norm) + [norm[0]]
            color = cluster_colors[c % len(cluster_colors)]
            ax_radar.plot(angles, vals, color=color, linewidth=2.5)
            ax_radar.fill(angles, vals, color=color, alpha=0.12)
            patches.append(mpatches.Patch(
                color=color, label=f"{label_of(c)} (n={sizes[c]}, {wkt[c]*100:.1f}% wkt)"
            ))
        ax_radar.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.1),
                         ncol=2, fontsize=10, frameon=False)

        ax_heat = fig1.add_subplot(1, 2, 2)
        bowler_scaler = StandardScaler().fit(sub[H.features])
        centroid_z = bowler_scaler.transform(centroids.values)
        vmin, vmax = -2, 2
        im = ax_heat.imshow(centroid_z, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
        ax_heat.xaxis.tick_top()
        ax_heat.xaxis.set_label_position("top")
        ax_heat.set_xticks(range(len(H.features)))
        ax_heat.set_xticklabels(col_labels, fontsize=9)
        ax_heat.set_yticks(range(k))
        ax_heat.set_yticklabels(
            [f"{label_of(c)}\nn={sizes[c]}, {wkt[c]*100:.1f}% wkt" for c in range(k)], fontsize=9
        )
        ax_heat.set_title("Cluster centroids (within-bowler z-score)", fontsize=13, pad=42)
        for i in range(k):
            for j in range(len(H.features)):
                norm_val = (centroid_z[i, j] - vmin) / (vmax - vmin)
                text_color = "white" if norm_val > 0.6 else "black"
                ax_heat.text(j, i, f"{centroids.values[i, j]:.1f}", ha="center", va="center",
                              fontsize=8, color=text_color)
        fig1.colorbar(im, ax=ax_heat, shrink=0.7, label="z-score", pad=0.02)

        pdf.savefig(fig1, bbox_inches="tight")
        plt.savefig(f"{fname}_page1.png", dpi=130, bbox_inches="tight")
        plt.close(fig1)

        # ── Pages 2+: per-innings strategy mix, a few innings per page ───────
        bar_width = 0.8 / chosen_n
        x = np.arange(k)
        n_pages = math.ceil(n_innings / INNINGS_PER_PAGE)

        for p in range(n_pages):
            chunk = innings_order[p * INNINGS_PER_PAGE:(p + 1) * INNINGS_PER_PAGE]
            n_rows = len(chunk)

            fig2 = plt.figure(figsize=(14, 2.6 * n_rows + 1.6))
            fig2.suptitle(
                f"{bowler} — per-innings strategy mix (page {p + 1} of {n_pages})",
                fontsize=15, fontweight="bold", y=0.99,
            )
            axes = fig2.subplots(n_rows, 2, gridspec_kw={"width_ratios": [3, 1.3]})
            if n_rows == 1:
                axes = axes.reshape(1, 2)
            fig2.subplots_adjust(top=0.90, hspace=0.65, wspace=0.18)

            # Page-level legend (state colours) and a note on the wicket marker
            state_patches = [mpatches.Patch(color=H.state_colors[s], label=f"State {s}")
                              for s in range(chosen_n)]
            fig2.legend(handles=state_patches, loc="upper center", bbox_to_anchor=(0.5, 0.965),
                        ncol=chosen_n, fontsize=11)
            fig2.text(0.985, 0.965, "★ = wicket", ha="right", va="center", fontsize=10)

            for row, (m, inn) in enumerate(chunk):
                i = p * INNINGS_PER_PAGE + row
                ax, ax2 = axes[row, 0], axes[row, 1]
                msub = sub[(sub["Match ID"] == m) & (sub["Innings ID"] == inn)].reset_index(drop=True)
                date = pd.to_datetime(msub["Match Start Date"].iloc[0]).date()

                # Left: delivery-type mix within each decoded state
                for s in range(chosen_n):
                    ssub = msub[msub["State"] == s]
                    n_s = len(ssub)
                    if n_s > 0:
                        props = ssub["Cluster"].value_counts(normalize=True).reindex(range(k), fill_value=0)
                    else:
                        props = pd.Series(0.0, index=range(k))
                    ax.bar(x + s * bar_width, props.values, width=bar_width,
                           color=H.state_colors[s], label=f"State {s} (n={n_s})")
                ax.set_xticks(x + bar_width * (chosen_n - 1) / 2)
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels(["0", "0.5", "1"], fontsize=9)
                ax.set_ylabel(f"Innings {i+1}\n{date}\nMatch ID {m}\nInnings ID {inn}",
                               fontsize=10, rotation=0, ha="right",
                               va="center", labelpad=12)
                ax.tick_params(axis="x", which="both", length=0)
                ax.set_xticklabels(short_labels, fontsize=9)
                if row == 0:
                    ax.set_title("Delivery-type mix per state "
                                  "(see page 1 for cluster names)", fontsize=12, pad=10)

                # Right: ball-by-ball decoded state, in chronological order (this innings only)
                xs = np.arange(len(msub))
                states_arr = msub["State"].to_numpy()
                for s in range(chosen_n):
                    mask = states_arr == s
                    ax2.scatter(xs[mask], states_arr[mask], s=14, color=H.state_colors[s])
                wkt_mask = msub["Is Wicket"].to_numpy()
                if wkt_mask.any():
                    ax2.scatter(xs[wkt_mask], states_arr[wkt_mask], marker="*", s=90,
                                color="black", zorder=5)
                ax2.set_yticks(range(chosen_n))
                ax2.set_ylim(-0.5, chosen_n - 0.5)
                ax2.set_xlim(-0.5, max(len(msub) - 0.5, 0.5))
                ax2.tick_params(axis="both", labelsize=9)
                ax2.set_xlabel("Ball number in innings", fontsize=9)
                if row == 0:
                    ax2.set_title("Ball-by-ball state (this innings)", fontsize=12, pad=10)

            pdf.savefig(fig2, bbox_inches="tight")
            if p == 0:
                plt.savefig(f"{fname}_page2.png", dpi=130, bbox_inches="tight")
            plt.close(fig2)

    print(f"Saved {fname}.pdf ({1 + n_pages} pages) / preview pngs")
