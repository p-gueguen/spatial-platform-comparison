#!/usr/bin/env python3
"""Atera vs CosMx WTx (breast FFPE) whole-transcriptome imaging head-to-head.
Three small multiples, each on its own scale (never a shared axis):
  A) per-cell depth (median genes/cell, transcripts/cell)
  B) molecule-for-molecule sensitivity on the shared gene set (mean tx / gene / cell)
  C) background / specificity: negative-probe + decoding-error normalised rate (log, lower=cleaner)
Fixed entity colours: Atera #7570B3, CosMx #A6761D (validated pair, CVD-safe both themes)."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,
                     "savefig.dpi":140,"font.family":"DejaVu Sans"})
ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"; FIG=f"{OUT}/figs"
A=json.load(open(f"{OUT}/per_dataset/wta_breast.json"))
C=json.load(open(f"{OUT}/per_dataset/cosmx_breast.json"))
ATC="#7570B3"; COC="#A6761D"

# Atera decoding-error (codeword) normalised rate, computed like CosMx falsecode_norm_rate
per_gene_A = A["total_gex_transcripts"]/A["n_gex_genes"]/A["n_units"]
atera_cw_norm = (A["negctrl_codeword_counts"]/A["n_negctrl_codewords"]/A["n_units"])/per_gene_A

fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(13.2,4.7))

# ---- A: per-cell depth ----
grp=["genes / cell","transcripts / cell"]; x=np.arange(2); w=0.38
av=[A["median_genes_per_cell"],A["median_transcripts_per_cell"]]
cv=[C["median_genes_per_cell"],C["median_transcripts_per_cell"]]
a1.bar(x-w/2,av,w,color=ATC,edgecolor="white")
a1.bar(x+w/2,cv,w,color=COC,edgecolor="white")
for xi,v in zip(x-w/2,av): a1.text(xi,v,f"{v:,.0f}",ha="center",va="bottom",fontsize=9)
for xi,v in zip(x+w/2,cv): a1.text(xi,v,f"{v:,.0f}",ha="center",va="bottom",fontsize=9)
a1.set_xticks(x); a1.set_xticklabels(grp); a1.set_ylabel("median per cell")
a1.set_title("Per-cell depth",fontweight="bold",fontsize=12)
a1.set_ylim(0,max(av+cv)*1.18)

# ---- B: molecule-for-molecule on shared genes ----
nsh=C["shared_with_atera_n_genes"]
bv=[C["shared_atera_mean_tx_per_gene_per_cell_atera"],C["shared_atera_mean_tx_per_gene_per_cell_cosmx"]]
a2.bar([0,1],bv,color=[ATC,COC],width=0.6,edgecolor="white")
for xi,v in zip([0,1],bv): a2.text(xi,v,f"{v:.3f}",ha="center",va="bottom",fontsize=9.5)
a2.set_xticks([0,1]); a2.set_xticklabels(["Atera","CosMx"])
a2.set_ylabel("mean transcripts / gene / cell")
a2.set_title(f"Same molecules, {nsh:,} shared genes",fontweight="bold",fontsize=12)
a2.set_ylim(0,max(bv)*1.20)

# ---- C: background / specificity (log, lower = cleaner) ----
grp3=["negative probe\n(nonspecific)","decoding error\n(unused codeword)"]; x3=np.arange(2)
aR=[A["negctrl_norm_rate"],atera_cw_norm]
cR=[C["negctrl_norm_rate"],C["falsecode_norm_rate"]]
a3.bar(x3-w/2,aR,w,color=ATC,edgecolor="white")
a3.bar(x3+w/2,cR,w,color=COC,edgecolor="white")
a3.set_yscale("log")
for xi,v in zip(x3-w/2,aR): a3.text(xi,v,f"{v:.4f}",ha="center",va="bottom",fontsize=8.3)
for xi,v in zip(x3+w/2,cR): a3.text(xi,v,f"{v:.3f}",ha="center",va="bottom",fontsize=8.3)
a3.set_xticks(x3); a3.set_xticklabels(grp3,fontsize=9)
a3.set_ylabel("normalised rate  vs  per-gene signal")
a3.set_title("Background (lower = cleaner)",fontweight="bold",fontsize=12)
a3.set_ylim(min(aR)*0.4, max(cR)*3)

leg=[Patch(facecolor=ATC,label="Atera  (10x, 18,028 genes)"),
     Patch(facecolor=COC,label="CosMx WTx  (Bruker, 18,942 genes)")]
fig.legend(handles=leg,loc="upper center",ncol=2,frameon=False,fontsize=10.5,bbox_to_anchor=(0.5,1.02))
fig.suptitle("Whole-transcriptome imaging head-to-head: Atera vs CosMx WTx (breast FFPE)",
             fontweight="bold",fontsize=13,y=1.10)
fig.tight_layout(rect=(0,0,1,0.98))
fig.savefig(f"{FIG}/13_atera_vs_cosmx.png",bbox_inches="tight"); plt.close(fig)
print("fig 13 written")
print(f"  depth: Atera {av}  CosMx {cv}")
print(f"  per-gene shared({nsh}): Atera {bv[0]:.4f}  CosMx {bv[1]:.4f}")
print(f"  bg negprobe: Atera {aR[0]:.5f}  CosMx {cR[0]:.4f} ; codeword: Atera {aR[1]:.5f}  CosMx {cR[1]:.4f}")
print(f"  concordance Spearman Atera-vs-CosMx: {C['concordance_spearman_vs_atera']:.3f}")
