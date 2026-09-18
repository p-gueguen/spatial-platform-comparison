#!/usr/bin/env python3
"""Figures for the diffusion / lateral spillover section (29_diffusion.py -> outputs/diffusion)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/srv/GT/analysis/pgueguen/spatial_platform_comparison"
OUT  = f"{ROOT}/outputs"; FIG = f"{OUT}/figs"; DIF = f"{OUT}/diffusion"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,
                     "figure.dpi":140,"savefig.dpi":140,"font.family":"DejaVu Sans"})

LAB = {"stdxenium_breast":"Xenium 313-plex","prime5k_breast":"Xenium Prime 5K",
       "wta_breast":"Atera 18k","visiumhd65_breast_8um":"Visium HD 6.5mm",
       "visiumhd11_breast_8um":"Visium HD 11mm","stratamap_breast":"Illumina StrataMap"}
CLR = {"stdxenium_breast":"#1b9e77","prime5k_breast":"#d95f02","wta_breast":"#7570b3",
       "visiumhd65_breast_8um":"#e7298a","visiumhd11_breast_8um":"#66a61e",
       "stratamap_breast":"#2e6faf"}
ORDER   = ["stdxenium_breast","prime5k_breast","wta_breast",
           "visiumhd65_breast_8um","visiumhd11_breast_8um","stratamap_breast"]
SOURCES = ["epithelial","immune","stromal"]
BIN_UM  = 8.0

MIN_CONTRAST = 1.5   # in-domain / far-field. Below this the radial fit is noise-dominated.

R = {}
for k in ORDER:
    p = f"{DIF}/{k}.json"
    if os.path.exists(p):
        R[k] = json.load(open(p))
KEYS = [k for k in ORDER if k in R]
print(f"[fig] {len(KEYS)} platforms with diffusion results: {KEYS}")


def gate(s):
    """Is this compartment's decay length interpretable at all? Returns (usable, reason)."""
    lam, con, nul = s.get("lambda_um"), s.get("contrast"), s.get("lambda_null_um")
    if lam is None:
        return False, "no decay could be fitted"
    if con is None or con < MIN_CONTRAST:
        return False, (f"domain contrast only {con:.2f}x, below the {MIN_CONTRAST}x floor"
                       if con else "domain contrast unmeasurable")
    if nul is not None and nul >= 0.5 * lam:
        return False, (f"the displaced-domain null returns {nul:.0f} um against {lam:.0f} um, "
                       f"so the fit is not tracking the domain")
    return True, None


def get(k, src, field):
    """Gated accessor: lambda-derived fields are withheld where the gate fails."""
    s = R[k]["sources"].get(src, {})
    if field in ("lambda_um", "aniso_ratio", "aniso_axis_deg", "lambda_sector_um",
                 "swap_lambda_um", "swap_lambda_ratio"):
        ok, _ = gate(s)
        if not ok:
            return None
    return s.get(field)


# ---------------------------------------------------------------- fig 1: lambda + profiles
fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9),
                         gridspec_kw=dict(width_ratios=[1.15, 1]))

ax = axes[0]
w = 0.26
bars = [v for k in KEYS for s in SOURCES for v in [get(k, s, "lambda_um")] if v]
swaps = [v for k in KEYS for s in SOURCES for v in [get(k, s, "swap_lambda_um")] if v]
top = max(bars + swaps) * 1.28 if (bars or swaps) else 100.0
for j, src in enumerate(SOURCES):
    xs = np.arange(len(KEYS)) + (j - 1) * w
    vals = [get(k, src, "lambda_um") for k in KEYS]
    ax.bar(xs, [v if v else 0 for v in vals], width=w, label=src,
           color=[CLR[k] for k in KEYS], alpha=[1.0, 0.62, 0.34][j],
           edgecolor="white", linewidth=0.6)
    for x, k, v in zip(xs, KEYS, vals):
        if v:
            ax.text(x, v + top * 0.015, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
        else:
            ax.text(x, top * 0.02, "withheld", ha="center", va="bottom", fontsize=6.5,
                    color="#999", rotation=90)
        sw = get(k, src, "swap_lambda_um")
        if sw:
            ax.plot([x], [sw], marker="_", ms=9, mew=1.6, color="#222", zorder=5)
        # the displaced-domain null is NOT gated, so it can sit far off scale (a null of
        # 1,179 um is exactly the signal that a measurement is meaningless). Clip it to the
        # top of the axis and mark it, rather than letting one outlier flatten every bar.
        nl = R[k]["sources"].get(src, {}).get("lambda_null_um")
        if nl:
            if nl <= top:
                ax.plot([x], [nl], marker="x", ms=5, mew=1.4, color="#c00", zorder=5)
            else:
                ax.plot([x], [top * 0.97], marker="^", ms=5, mew=0, color="#c00", zorder=5)
                ax.text(x, top * 0.90, f"{nl:.0f}", ha="center", va="top", fontsize=6,
                        color="#c00", rotation=90)
ax.axhline(BIN_UM, color="#444", ls=":", lw=1.2)
ax.text(len(KEYS) - 0.45, BIN_UM * 1.15, f"{BIN_UM:g} µm lattice floor",
        fontsize=8.5, color="#444", ha="right")
ax.set_ylim(0, top)
ax.set_xticks(np.arange(len(KEYS)))
ax.set_xticklabels([LAB[k] for k in KEYS], rotation=22, ha="right")
ax.set_ylabel("spillover decay length  λ  (µm)")
ax.set_title("How far a compartment's signature reaches outside it",
             fontweight="bold", fontsize=12)
leg1 = ax.legend(title="source compartment", fontsize=8.5, title_fontsize=8.5, frameon=False,
                 loc="upper left", ncol=3, columnspacing=1.0, handlelength=1.1)
ax.add_artist(leg1)
h = [plt.Line2D([], [], marker="_", ls="", color="#222", mew=1.6, ms=9),
     plt.Line2D([], [], marker="x", ls="", color="#c00", mew=1.4, ms=5),
     plt.Line2D([], [], marker="^", ls="", color="#c00", ms=5)]
ax.legend(h, ["marker halves swapped", "displaced-domain null",
              "null off scale (see number)"],
          fontsize=7.5, frameon=False, loc="upper right", handlelength=1.1)

# Right panel uses the IMMUNE source: it is the only compartment that passes the quality gates
# on every platform, so it is the only one where all six curves mean the same thing.
ax = axes[1]
for k in KEYS:
    s = R[k]["sources"].get("immune", {})
    if "profile_r_um" not in s or get(k, "immune", "lambda_um") is None:
        continue
    r = np.array(s["profile_r_um"]); p = np.array(s["profile_p"])
    m = p > 0
    ax.plot(r[m], p[m], "o-", ms=4, lw=1.6, color=CLR[k], label=LAB[k])
    lam = get(k, "immune", "lambda_um")
    if lam:
        rr = np.linspace(r[m].min(), r[m].max(), 50)
        ax.plot(rr, np.exp(-rr / lam) / np.exp(-r[m].min() / lam) * p[m][0],
                ls="--", lw=1.0, color=CLR[k], alpha=0.55)
ax.set_yscale("log")
ax.set_xlabel("distance outside the immune domain (µm)")
ax.set_ylabel("excess immune signal\n(fraction of in-domain level)")
ax.set_title("The measured decay, immune source\n(the one compartment measurable on all six)",
             fontweight="bold", fontsize=11.5)
ax.legend(fontsize=8.5, frameon=False)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_diffusion_lambda.png", bbox_inches="tight")
plt.close(fig)
print("fig: fig_diffusion_lambda")


# ---------------------------------------------------------------- fig 2: direction
fig = plt.figure(figsize=(13.6, 4.9))
axp = fig.add_subplot(1, 2, 1, projection="polar")
for k in KEYS:
    s = R[k]["sources"].get("immune", {})
    if get(k, "immune", "lambda_um") is None:
        continue
    ls = s.get("lambda_sector_um") or {}
    th, rv = [], []
    for i in range(8):
        v = ls.get(str(i))
        if v:
            th.append(np.deg2rad(i * 45 - 180 + 22.5)); rv.append(v)
    if len(rv) >= 6:
        th.append(th[0]); rv.append(rv[0])
        axp.plot(th, rv, "-o", ms=3, lw=1.5, color=CLR[k], label=LAB[k])
axp.set_title("λ by direction, immune source\n(a circle is isotropic leak)",
              fontweight="bold", fontsize=11.5, pad=18)
axp.set_theta_zero_location("E")
axp.tick_params(labelsize=8)
axp.legend(fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2)

# Right panel: the raw drift statistic against its own calibration curve. An all-zero bar chart
# would show that nothing was found without showing how far below detectability it sits, which is
# the part that makes a negative result worth anything.
ax = fig.add_subplot(1, 2, 2)
cal = json.load(open(f"{DIF}/_drift_calibration.json")) if \
    os.path.exists(f"{DIF}/_drift_calibration.json") else None
rows = [(k, s) for k in KEYS for s in SOURCES
        if R[k]["sources"].get(s, {}).get("drift_mag") is not None]
yy = np.arange(len(rows))
for y, (k, s) in zip(yy, rows):
    v = R[k]["sources"][s]["drift_mag"]
    ax.plot([1e-4, v], [y, y], lw=1.1, color=CLR[k], alpha=0.45)
    ax.plot([v], [y], "o", ms=6, color=CLR[k],
            alpha=[1.0, 0.66, 0.4][SOURCES.index(s)])
if cal:
    for off, mag in zip(cal["offset_um"], cal["drift_mag"]):
        if off == 0:
            continue
        ax.axvline(mag, color="#c00", ls="--", lw=1.0, alpha=0.75)
        ax.text(mag, len(rows) - 0.3, f" {off:g} µm", color="#c00", fontsize=8,
                rotation=90, va="top", ha="left")
ax.set_xscale("log")
ax.set_yticks(yy)
ax.set_yticklabels([f"{LAB[k]} · {s}" for k, s in rows], fontsize=7.5)
ax.set_xlabel("directional-bias statistic |D|   (log scale)\n"
              "red lines = |D| produced by a KNOWN one-sided offset of that size,\n"
              "injected into synthetic data", fontsize=9.5)
ax.set_title("Is the leak one-sided? (the Visium v1 signature)",
             fontweight="bold", fontsize=12)
ax.set_ylim(-0.8, len(rows) - 0.2)
ax.grid(axis="y", alpha=0.15)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_diffusion_direction.png", bbox_inches="tight")
plt.close(fig)
print("fig: fig_diffusion_direction")


# ---------------------------------------------------------------- summary table to json
summary = {"_gate": dict(min_contrast=MIN_CONTRAST,
                         rule="lambda and the anisotropy/swap values derived from it are withheld "
                              "unless the fit converged, the in-domain/far-field contrast is at "
                              "least min_contrast, and the displaced-domain null is below half the "
                              "fitted value. Drift, contrast and off-domain share are NOT gated: "
                              "they need no fit.")}
for k in KEYS:
    summary[k] = dict(off_cell_frac=R[k].get("off_cell_frac"),
                      off_cell_note=R[k].get("off_cell_note"))
    for src in SOURCES:
        s = R[k]["sources"].get(src, {})
        ok, why = gate(s)
        summary[k][src] = {f: (get(k, src, f) if f in
                               ("lambda_um", "aniso_ratio", "aniso_axis_deg",
                                "swap_lambda_um", "swap_lambda_ratio") else s.get(f))
                           for f in
                           ("lambda_um", "lambda_null_um", "swap_lambda_um", "swap_lambda_ratio",
                            "aniso_ratio", "aniso_axis_deg", "drift_equiv_um", "drift_deg",
                            "drift_mag", "drift_p", "contrast", "in_domain_frac", "ambient_frac",
                            "n_source_bins", "lambda_fit_r2",
                            "offdomain_signal_frac_gt16um", "offdomain_signal_frac_gt40um")}
        summary[k][src]["lambda_usable"] = bool(ok)
        summary[k][src]["lambda_withheld_because"] = why
        summary[k][src]["lambda_raw_um"] = s.get("lambda_um")
        if "error" in s:
            summary[k][src]["error"] = s["error"]
        print(f"  {k:24s} {src:11s} lambda={'OK' if ok else 'WITHHELD: ' + str(why)}")
json.dump(summary, open(f"{DIF}/_summary.json", "w"), indent=2)
print("wrote", f"{DIF}/_summary.json")
