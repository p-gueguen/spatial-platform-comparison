#!/usr/bin/env python3
"""39_immune_rescue_figure.py - Figure 28: RCTD on vendor vs Proseg segmentation, like-for-like.

Reads outputs/rctd_matched/summary.csv (40_rctd_matched_rerun.py --summarise). Every bar is the same
1.5 x 1.5 mm window per platform, reference genes only, one threshold scale. The earlier version of this
figure hardcoded numbers from 36_rctd_gpu_benchmark.py, which compared a section-wide vendor subset
with a Proseg window and scaled RCTD's thresholds with each input's feature count.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = "/srv/GT/analysis/pgueguen/spatial_platform_comparison"
ROWS = [("atera_vendor_fixed", "Atera\nvendor"), ("atera_proseg_fixed", "Atera\nProseg"),
        ("sm_vendor_fixed", "StrataMap G1\nvendor"), ("sm_proseg_fixed", "StrataMap G1\nProseg")]


def main(out=f"{ROOT}/outputs/figs/28_rctd_immune_rescue.png"):
    s = pl.read_csv(f"{ROOT}/outputs/rctd_matched/summary.csv")
    d = {r["condition"]: r for r in s.iter_rows(named=True)}
    rows = [(k, lab) for k, lab in ROWS if k in d]
    x = np.arange(len(rows)); labs = [lab for _, lab in rows]
    get = lambda col: np.array([100 * d[k][col] for k, _ in rows])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)

    sing, dbl, rej = get("singlet"), get("doublet"), get("reject")
    ax[0].bar(x, sing, 0.6, label="singlet", color="#2ca02c")
    ax[0].bar(x, dbl, 0.6, bottom=sing, label="doublet", color="#ff7f0e")
    ax[0].bar(x, rej, 0.6, bottom=sing + dbl, label="reject", color="#d62728")
    for i, r in enumerate(rej): ax[0].text(i, sing[i] + dbl[i] + r / 2, f"{r:.1f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax[0].set_ylabel("% of cells"); ax[0].set_ylim(0, 100); ax[0].set_xticks(x, labs)
    ax[0].set_title("A  RCTD spot class, same window per platform"); ax[0].legend(fontsize=8, loc="lower right")

    mal, stro, endo, imm = get("malignant_of_singlets"), get("stroma_of_singlets"), get("endothelial_of_singlets"), get("immune_of_singlets")
    b = np.zeros(len(rows))
    for v, lab, col in [(mal, "malignant / epithelial", "#8c564b"), (stro, "stroma / mural", "#17becf"),
                        (endo, "endothelial", "#bcbd22"), (imm, "immune", "#9467bd")]:
        ax[1].bar(x, v, 0.6, bottom=b, label=lab, color=col); b = b + v
    for i, v in enumerate(imm): ax[1].text(i, b[i] + 1.5, f"immune {v:.1f}%", ha="center", fontsize=8)
    ax[1].set_ylabel("% of singlets"); ax[1].set_ylim(0, 110); ax[1].set_xticks(x, labs)
    ax[1].set_title("B  lineage of singlets"); ax[1].legend(fontsize=8, loc="lower right")
    fig.text(0.5, -0.02, "One 1.5 x 1.5 mm window per platform; different specimens and fixation (Atera FFPE, StrataMap fresh-frozen). "
             "CELLxGENE Census poly-A reference.", ha="center", fontsize=8, color="#555")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
