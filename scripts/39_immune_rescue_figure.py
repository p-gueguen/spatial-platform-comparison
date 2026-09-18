#!/usr/bin/env python3
"""
39_immune_rescue_figure.py
Generate Figure 28: RCTD deconvolution, ambient tumor soup, and immune rescue.
Contrasts 10x Atera FFPE against Illumina StrataMap (Grades 1, 2, 3) and Proseg-dediffused data.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def generate_figure(output_path="outputs/figs/28_rctd_immune_rescue.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=200)

    datasets = ['Atera FFPE\n(Std)', 'StrataMap G1\n(Std)', 'StrataMap G2\n(Std)', 'StrataMap G3\n(Std)', 'StrataMap G1\n(Proseg)']
    x = np.arange(len(datasets))

    # Panel A: Spot Classification (Singlet vs Reject vs Doublet)
    ax = axes[0, 0]
    singlets = [33.56, 44.40, 37.44, 55.88, 76.36]
    rejects = [8.80, 55.60, 62.52, 43.98, 21.16]
    doublets = [57.64, 0.00, 0.04, 0.14, 2.48]

    b1 = ax.bar(x, singlets, width=0.55, label='Singlets (Usable)', color='#2ca02c', alpha=0.85)
    b2 = ax.bar(x, doublets, width=0.55, bottom=singlets, label='Doublets', color='#ff7f0e', alpha=0.85)
    bottom_rej = np.array(singlets) + np.array(doublets)
    b3 = ax.bar(x, rejects, width=0.55, bottom=bottom_rej, label='Rejects (Mixture Noise)', color='#d62728', alpha=0.85)

    ax.set_ylabel('Spot Class Fraction (%)', fontsize=11, fontweight='bold')
    ax.set_title('A: RCTD Spot Classification (Whole Transcriptome)', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=True, fontsize=9, loc='upper right')

    for i, (s, r) in enumerate(zip(singlets, rejects)):
        ax.text(i, s/2, f'{s:.1f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)
        if r > 10:
            ax.text(i, 100 - r/2, f'{r:.1f}%\nRej', ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    # Panel B: Major Lineage Distribution among Singlets
    ax = axes[0, 1]
    malignant = [41.3, 87.8, 96.3, 89.9, 72.3]
    stroma = [31.3, 7.5, 1.8, 3.3, 18.9]
    endothelial = [9.5, 2.8, 0.3, 0.5, 5.6]
    immune = [16.0, 1.1, 1.5, 6.2, 2.2]

    b_mal = ax.bar(x, malignant, width=0.55, label='Malignant / Epithelial', color='#8c564b', alpha=0.85)
    b_str = ax.bar(x, stroma, width=0.55, bottom=malignant, label='Fibroblast / Stroma', color='#17becf', alpha=0.85)
    b_end = ax.bar(x, endothelial, width=0.55, bottom=np.array(malignant)+np.array(stroma), label='Endothelial', color='#bcbd22', alpha=0.85)
    b_imm = ax.bar(x, immune, width=0.55, bottom=np.array(malignant)+np.array(stroma)+np.array(endothelial), label='Immune Compartment', color='#9467bd', alpha=0.85)

    ax.set_ylabel('Lineage Share in Singlets (%)', fontsize=11, fontweight='bold')
    ax.set_title('B: Unmasking Stroma & Immune from Ambient Tumor Soup', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=True, fontsize=8.5, loc='upper right')

    # Panel C: Immune Subsets (T/NK vs Myeloid)
    ax = axes[1, 0]
    width = 0.35
    t_nk = [10.49, 0.09, 0.05, 0.14, 0.05]
    myeloid = [4.65, 0.90, 1.39, 5.73, 1.96]

    r1 = ax.bar(x - width/2, t_nk, width, label='T / NK Singlet %', color='#1f77b4', alpha=0.85)
    r2 = ax.bar(x + width/2, myeloid, width, label='Myeloid Singlet %', color='#ff7f0e', alpha=0.85)

    ax.set_ylabel('% of Confident Singlets', fontsize=11, fontweight='bold')
    ax.set_title('C: Immune Compartment Singlet Recovery', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9.5)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=True, fontsize=9.5)

    for i, (t, m) in enumerate(zip(t_nk, myeloid)):
        ax.text(i - width/2, t + 0.2, f'{t:.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1f77b4')
        ax.text(i + width/2, m + 0.2, f'{m:.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#d95f02')
    ax.set_ylim(0, 12.5)

    # Panel D: Ambient Soup Stripping (EPCAM in T-cell candidates)
    ax = axes[1, 1]
    labels_d = ['Standard\nStrataMap G1', 'Proseg\nDe-diffused', 'Proseg +\nSPLIT Purified']
    cd3_counts = [5, 119, 119]
    epcam_in_t = [100.0, 36.8, 0.3]  # % of T cells contaminated with EPCAM (>0)

    ax_twin = ax.twinx()
    p1 = ax.bar(np.arange(3) - 0.18, cd3_counts, width=0.35, color='#386cb0', alpha=0.85, label='CD3D+ Cells (Count)')
    p2 = ax_twin.bar(np.arange(3) + 0.18, epcam_in_t, width=0.35, color='#e41a1c', alpha=0.75, label='% T Cells with EPCAM Spillover')

    ax.set_ylabel('Number of CD3D>=2 Cells Recovered', fontsize=10.5, fontweight='bold', color='#386cb0')
    ax_twin.set_ylabel('% T Cells with Detectable EPCAM (Spillover)', fontsize=10.5, fontweight='bold', color='#e41a1c')
    ax.set_title('D: Immune Rescue & Ambient Soup Elimination', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels_d, fontsize=10)
    ax.set_ylim(0, 140)
    ax_twin.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    for i, (c, e) in enumerate(zip(cd3_counts, epcam_in_t)):
        ax.text(i - 0.18, c + 3, f'{c}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#386cb0')
        ax_twin.text(i + 0.18, e + 2, f'{e:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#e41a1c')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    generate_figure()
