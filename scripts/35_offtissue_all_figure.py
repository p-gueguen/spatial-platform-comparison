#!/usr/bin/env python3
"""Figure: all six platforms on the Visium v1 off-tissue axis (34_offtissue_all_platforms.py)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/srv/GT/analysis/pgueguen/spatial_platform_comparison"
OUT  = f"{ROOT}/outputs"; FIG = f"{OUT}/figs"; DIF = f"{OUT}/diffusion"
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,
                     "figure.dpi":140,"savefig.dpi":140,"font.family":"DejaVu Sans"})

A  = json.load(open(f"{DIF}/_offtissue_all.json"))
V1 = json.load(open(f"{DIF}/_v1_control.json"))

LAB = {"stdxenium_breast":"Xenium 313-plex","prime5k_breast":"Xenium Prime 5K",
       "wta_breast":"Atera 18k","visiumhd65_breast_8um":"Visium HD 6.5mm",
       "visiumhd11_breast_8um":"Visium HD 11mm","stratamap_breast":"Illumina StrataMap",
       "visiumv1_breast_s1":"Visium v1 · s1","visiumv1_breast_s2":"Visium v1 · s2"}
CLR = {"stdxenium_breast":"#1b9e77","prime5k_breast":"#d95f02","wta_breast":"#7570b3",
       "visiumhd65_breast_8um":"#e7298a","visiumhd11_breast_8um":"#66a61e",
       "stratamap_breast":"#2e6faf","visiumv1_breast_s1":"#B2182B","visiumv1_breast_s2":"#D6604D"}
MODERN = ["stdxenium_breast","prime5k_breast","wta_breast",
          "visiumhd65_breast_8um","visiumhd11_breast_8um","stratamap_breast"]
V1KEYS = [k for k in ("visiumv1_breast_s1","visiumv1_breast_s2")
          if k in V1 and "error" not in V1[k]]
OK = [k for k in MODERN if k in A and "error" not in A[k]]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8),
                         gridspec_kw=dict(width_ratios=[1.05, 1.0, 0.95]))

# ---- 1. profiles, v1 on top of the six
ax = axes[0]
for k in V1KEYS:
    v = V1[k]
    ax.plot(v["profile_r_um"], np.array(v["profile_y"]) * 100, "o-", ms=5, lw=2.6,
            color=CLR[k], label=LAB[k], zorder=6)
for k in OK:
    v = A[k]
    ax.plot(v["profile_r_um"], np.array(v["profile_y"]) * 100, "s--", ms=4, lw=1.5,
            color=CLR[k], label=LAB[k], zorder=3)
ax.set_yscale("log")
ax.set_xlabel("distance beyond the tissue edge (µm)")
ax.set_ylabel("counts, as % of the in-tissue level")
ax.set_title("Signal recovered beyond the tissue", fontweight="bold", fontsize=12)
ax.legend(fontsize=7.5, frameon=False, ncol=2)

# ---- 2. first-ring level, with the threshold-sensitivity range as an error bar
ax = axes[1]
keys = V1KEYS + OK
xs = np.arange(len(keys))
sens = A.get("_threshold_sensitivity", {})
for x, k in zip(xs, keys):
    v = (V1[k] if k in V1KEYS else A[k])
    val = v["offtissue_level_at_first_ring"] * 100
    ax.bar([x], [val], color=CLR[k], edgecolor="white", linewidth=0.7,
           hatch="" if k in V1KEYS else "//")
    lo = hi = val
    if k in sens:
        good = [q * 100 for q in (sens[k].get("level") or {}).values() if q is not None]
        if len(good) > 1:
            lo, hi = min(good), max(good)
            ax.plot([x, x], [lo, hi], color="#222", lw=1.4, zorder=6)
            ax.plot([x], [hi], marker="_", ms=8, color="#222", zorder=6)
            ax.plot([x], [lo], marker="_", ms=8, color="#222", zorder=6)
    ax.text(x, max(hi, val) + 1.0, f"{val:.0f}", ha="center", va="bottom", fontsize=8.5,
            fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([LAB[k].replace(" · ", "\n") for k in keys],
                                      rotation=30, ha="right", fontsize=8)
ax.set_ylabel("% of the in-tissue level")
ax.set_title("Leak into the first 100 µm beyond the tissue",
             fontweight="bold", fontsize=11.5)
h = [plt.Rectangle((0,0),1,1,fc="#999",ec="white"),
     plt.Rectangle((0,0),1,1,fc="#999",ec="white",hatch="//"),
     plt.Line2D([],[],color="#222",lw=1.4)]
ax.legend(h, ["Visium v1 (vendor in_tissue flag)",
              "modern platforms (data-derived mask)",
              "range over 3 mask thresholds"], fontsize=7.5, frameon=False)

# ---- 3. one-sidedness vs each platform's own null
ax = axes[2]
ys = np.arange(len(keys))
for y, k in zip(ys, keys):
    v = (V1[k] if k in V1KEYS else A[k])
    # range over the three mask thresholds: for the four platforms with no vendor tissue flag
    # this is wide enough that no directional ranking against v1 survives, which is the point
    dr = [q for q in ((sens.get(k) or {}).get("drift") or {}).values() if q]
    if len(dr) > 1:
        ax.plot([min(dr), max(dr)], [y, y], lw=5, color=CLR[k], alpha=0.30,
                solid_capstyle="butt", zorder=3)
    ax.plot([v["drift_mag"]], [y], "o", ms=8, color=CLR[k], zorder=5)
    if v.get("drift_null_p99") is not None:
        ax.plot([v["drift_null_p99"]], [y], "|", ms=14, mew=2.0, color="#444", zorder=6)
for vv in V1KEYS:
    ax.axvline(V1[vv]["drift_mag"], color="#B2182B", ls=":", lw=1.2, zorder=2)
ax.plot([], [], "|", ms=14, mew=2.0, color="#444",
        label="99th pct of that section's\nown shuffled null")
ax.plot([], [], lw=5, color="#999", alpha=0.30, label="range over 3 mask thresholds")
ax.plot([], [], color="#B2182B", ls=":", lw=1.2, label="Visium v1 values")
ax.set_yticks(ys); ax.set_yticklabels([LAB[k].replace(" · ", " ") for k in keys], fontsize=8)
ax.set_xlabel("one-sidedness of the leak,  |D|")
ax.set_title("Is the leak directional?\n(bands straddle v1: not rankable)",
             fontweight="bold", fontsize=11)
ax.set_xscale("log")
ax.legend(fontsize=7, frameon=False, loc="lower left")

fig.tight_layout()
fig.savefig(f"{FIG}/fig_offtissue_all.png", bbox_inches="tight")
plt.close(fig)
print("fig: fig_offtissue_all")

fr = lambda k: (V1[k] if k in V1KEYS else A[k])["offtissue_level_at_first_ring"] * 100
dm = lambda k: (V1[k] if k in V1KEYS else A[k])["drift_mag"]
print(f"  v1 first-ring {np.mean([fr(k) for k in V1KEYS]):.1f}%  vs  "
      f"six modern {np.mean([fr(k) for k in OK]):.1f}% "
      f"(range {min(fr(k) for k in OK):.1f}-{max(fr(k) for k in OK):.1f}%)")
print(f"  v1 |D| {np.mean([dm(k) for k in V1KEYS]):.3f}  vs  "
      f"six modern {np.mean([dm(k) for k in OK]):.3f} "
      f"(range {min(dm(k) for k in OK):.3f}-{max(dm(k) for k in OK):.3f})")
