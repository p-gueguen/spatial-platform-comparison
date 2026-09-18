#!/usr/bin/env python3
"""Figure for the Visium v1 real-data positive control (32_visium_v1_control.py)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/srv/GT/analysis/pgueguen/spatial_platform_comparison"
OUT  = f"{ROOT}/outputs"; FIG = f"{OUT}/figs"; DIF = f"{OUT}/diffusion"
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,
                     "figure.dpi":140,"savefig.dpi":140,"font.family":"DejaVu Sans"})

R = json.load(open(f"{DIF}/_v1_control.json"))
LAB = {"visiumv1_breast_s1":"Visium v1 · breast s1", "visiumv1_breast_s2":"Visium v1 · breast s2",
       "visiumhd65_breast_8um":"Visium HD 6.5mm", "visiumhd11_breast_8um":"Visium HD 11mm"}
CLR = {"visiumv1_breast_s1":"#B2182B", "visiumv1_breast_s2":"#D6604D",
       "visiumhd65_breast_8um":"#e7298a", "visiumhd11_breast_8um":"#66a61e"}
ORDER = [k for k in LAB if k in R and "error" not in R[k]]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6),
                         gridspec_kw=dict(width_ratios=[1.1, 0.75, 0.95]))

# --- 1. the decay profiles
ax = axes[0]
for k in ORDER:
    v = R[k]
    r = np.array(v["profile_r_um"]); y = np.array(v["profile_y"]) * 100
    v1 = v["kind"] == "v1"
    ax.plot(r, y, "o-" if v1 else "s--", ms=5, lw=2.2 if v1 else 1.5, color=CLR[k],
            label=LAB[k], zorder=5 if v1 else 3)
ax.set_yscale("log")
ax.set_xlabel("distance from the nearest tissue-containing spot (µm)")
ax.set_ylabel("counts, as % of the in-tissue level")
ax.set_title("Signal recovered where there is NO tissue",
             fontweight="bold", fontsize=12)
ax.legend(fontsize=8.5, frameon=False)
ax.text(0.98, 0.95, "anything above zero here\ngot there by moving",
        transform=ax.transAxes, fontsize=8.5, color="#555", ha="right", va="top", style="italic")

# --- 2. the level in the first ring outside tissue
ax = axes[1]
xs = np.arange(len(ORDER))
vals = [R[k]["offtissue_level_at_first_ring"] * 100 for k in ORDER]
ax.bar(xs, vals, color=[CLR[k] for k in ORDER], edgecolor="white", linewidth=0.7)
for x, v in zip(xs, vals):
    ax.text(x, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([LAB[k].replace(" · ", "\n") for k in ORDER],
                                      rotation=22, ha="right", fontsize=8.5)
ax.set_ylabel("% of the in-tissue level")
ax.set_ylim(0, max(vals) * 1.25)
ax.set_title("Leak into the adjacent\noff-tissue ring", fontweight="bold", fontsize=11.5)

# --- 3. one-sidedness against each platform's own null
ax = axes[2]
ys = np.arange(len(ORDER))
for y, k in zip(ys, ORDER):
    v = R[k]
    ax.plot([0, v["drift_mag"]], [y, y], lw=1.2, color=CLR[k], alpha=0.5)
    ax.plot([v["drift_mag"]], [y], "o", ms=8, color=CLR[k],
            label=None, zorder=5)
    ax.plot([v["drift_null_p99"]], [y], "|", ms=14, mew=2.0, color="#444", zorder=6)
ax.plot([], [], "|", ms=14, mew=2.0, color="#444", label="99th pct of that section's\nown shuffled null")
ax.set_yticks(ys); ax.set_yticklabels([LAB[k].replace(" · ", " ") for k in ORDER], fontsize=8.5)
ax.set_xlabel("one-sidedness of the leak,  |D|")
ax.set_title("Is the leak directional?", fontweight="bold", fontsize=11.5)
ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.set_xlim(0, max(R[k]["drift_mag"] for k in ORDER) * 1.35)

fig.tight_layout()
fig.savefig(f"{FIG}/fig_v1_control.png", bbox_inches="tight")
plt.close(fig)
print("fig: fig_v1_control")

v1 = [R[k] for k in ORDER if R[k]["kind"] == "v1"]
hd = [R[k] for k in ORDER if R[k]["kind"] == "hd"]
print(f"  v1 first-ring {np.mean([v['offtissue_level_at_first_ring'] for v in v1])*100:.1f}% vs "
      f"HD {np.mean([v['offtissue_level_at_first_ring'] for v in hd])*100:.1f}%")
print(f"  v1 |D| {np.mean([v['drift_mag'] for v in v1]):.3f} vs "
      f"HD {np.mean([v['drift_mag'] for v in hd]):.3f}")
