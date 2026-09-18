#!/usr/bin/env python3
"""StrataMap on the breadth-depth frontier: panel breadth (genes) vs per-gene sensitivity
(transcripts/mm2/gene on the 186 shared genes), both recomputed & segmentation-free.
StrataMap (Illumina, recomputed from the raw 1um-SBC matrix) sits top-right - highest on both."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,
                     "savefig.dpi":140,"font.family":"DejaVu Sans"})
ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"; FIG=f"{OUT}/figs"
D=json.load(open(f"{OUT}/bundle.json"))["datasets"]
sm=json.load(open(f"{OUT}/per_dataset/stratamap_breast.json"))

pts=[  # label, genes, tx/mm2/gene(186), colour, prep, highlight
 ("Xenium 313",           D["stdxenium_breast"]["n_gex_genes"], D["stdxenium_breast"]["L1_transcripts_per_mm2_per_gene"], "#1B9E77","FFPE",0),
 ("Xenium Prime 5K",      D["prime5k_breast"]["n_gex_genes"],   D["prime5k_breast"]["L1_transcripts_per_mm2_per_gene"],   "#D95F02","FFPE",0),
 ("Atera",                D["wta_breast"]["n_gex_genes"],       D["wta_breast"]["L1_transcripts_per_mm2_per_gene"],       "#7570B3","FFPE",0),
 ("Visium HD 6.5 mm",     D["visiumhd65_breast_8um"]["n_gex_genes"], D["visiumhd65_breast_8um"]["L1_transcripts_per_mm2_per_gene"], "#E7298A","FFPE",0),
 ("Visium HD 11 mm",      D["visiumhd11_breast_8um"]["n_gex_genes"], D["visiumhd11_breast_8um"]["L1_transcripts_per_mm2_per_gene"], "#66A61E","FFPE",0),
 ("Illumina StrataMap",   sm["n_gex_genes"],                    sm["L1_transcripts_per_mm2_per_gene"],                    "#2E6FAF","fresh-frozen",1),
]
fig,ax=plt.subplots(figsize=(8.8,5.6))
for lab,g,y,c,prep,hl in pts:
    ax.scatter(g,y,s=(320 if hl else 150),c=c,edgecolor=("#12385f" if hl else "white"),
               linewidth=(2.0 if hl else 1.2),zorder=(5 if hl else 3),marker=("*" if hl else "o"))
# labels with small offsets to avoid the HD overlap
off={"Visium HD 6.5 mm":(1.05,0.72),"Visium HD 11 mm":(1.05,1.18),"Illumina StrataMap":(0.60,1.16),
     "Atera":(1.06,1.0),"Xenium 313":(0.42,1.0),"Xenium Prime 5K":(1.06,1.0)}
for lab,g,y,c,prep,hl in pts:
    dx,dy=off.get(lab,(1.06,1.0))
    ax.annotate(f"{lab}\n{y:,.0f} tx/mm²/gene · {g:,} genes"+(f"\n({prep})" if hl else ""),
                (g,y),xytext=(g*dx,y*dy),fontsize=(9.5 if hl else 8.3),
                fontweight=("700" if hl else "400"),color=("#12385f" if hl else "#33454c"),
                ha=("right" if dx<1 else "left"),va="center")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("panel breadth  (gene-expression genes, log)")
ax.set_ylabel("per-gene sensitivity  (transcripts / mm² / gene, 186 shared genes, log)")
ax.set_title("StrataMap breaks the breadth-depth frontier",fontweight="bold",fontsize=13)
ax.annotate("higher = more sensitive per gene",xy=(0.02,0.97),xycoords="axes fraction",
            fontsize=8.5,color="#8A9AA0",va="top")
ax.annotate("broader panel  →",xy=(0.98,0.02),xycoords="axes fraction",fontsize=8.5,color="#8A9AA0",ha="right")
ax.set_xlim(200,120000); ax.set_ylim(80,6000)
fig.tight_layout(); fig.savefig(f"{FIG}/14_stratamap_frontier.png",bbox_inches="tight"); plt.close(fig)
print("fig 14 written")
for lab,g,y,c,prep,hl in pts: print(f"  {lab:20s} genes={g:6d}  tx/mm2/gene={y:.0f}")
