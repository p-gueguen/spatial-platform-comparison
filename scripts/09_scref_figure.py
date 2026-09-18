#!/usr/bin/env python3
"""Dissociated scRNA-seq sensitivity reference vs whole-transcriptome spatial, same tissue (breast).
Two panels: median genes/cell AND median UMIs (counts)/cell. Whole-tx methods only
(Xenium/Prime panels are gene-capped, not comparable here)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,
                     "savefig.dpi":140,"font.family":"DejaVu Sans"})
ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"; FIG=f"{OUT}/figs"
D=json.load(open(f"{OUT}/bundle.json"))["datasets"]
sc=json.load(open(f"{OUT}/scref.json"))
ff=json.load(open(f"{OUT}/per_dataset/visiumhd65ff_breast_seg.json"))

# (label, genes/cell, UMIs/cell, color, group, fresh-frozen?)
bars=[]
for a in ["10x 3' v3","10x 5' v2","Smart-seq2","flex"]:
    if a in sc:
        bars.append((sc[a]["label"], sc[a]["median_genes_per_cell"], sc[a]["median_umis_per_cell"], "#9AA7AD","scRNA",False))
# spatial: FFPE platforms + the fresh-frozen StrataMap (hatched, so prep stays visible)
bars.append(("Atera (FFPE)", D["wta_breast"]["median_genes_per_cell"], D["wta_breast"]["median_transcripts_per_cell"], "#7570B3","sp",False))
bars.append(("Visium HD 11 mm (FFPE)", D["visiumhd11_breast_seg"]["median_genes_per_cell"], D["visiumhd11_breast_seg"]["median_transcripts_per_cell"], "#66A61E","sp",False))
bars.append(("StrataMap (FF)", D["stratamap_breast"]["median_genes_per_cell"], D["stratamap_breast"]["median_transcripts_per_cell"], "#2E6FAF","sp",True))

labs=[b[0] for b in bars]; genes=[b[1] for b in bars]; umis=[b[2] for b in bars]; cols=[b[3] for b in bars]; ffs=[b[5] for b in bars]
n_sc=sum(1 for b in bars if b[4]=="scRNA"); x=np.arange(len(bars))
fig,(a1,a2)=plt.subplots(1,2,figsize=(13.0,4.8))
for ax,vals,ylab,logy in [(a1,genes,"median genes detected per cell",False),
                          (a2,umis,"median counts (UMIs / transcripts) per cell",True)]:
    bb=ax.bar(x,vals,color=cols,edgecolor="white")
    for b,ff in zip(bb,ffs):
        if ff: b.set_hatch("///"); b.set_edgecolor("white")
    if logy: ax.set_yscale("log")
    for xi,v in zip(x,vals): ax.text(xi,v,f"{v:,.0f}",ha="center",va="bottom",fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labs,rotation=26,ha="right",fontsize=8.5)
    ax.set_ylabel(ylab)
    ax.axvspan(-0.5,n_sc-0.5,color="#9AA7AD",alpha=0.10)
    yt=ax.get_ylim()[1]
    ax.text(n_sc/2-0.5, yt*(0.9 if logy else 0.97),"dissociated scRNA-seq",ha="center",fontsize=8.5,color="#5B6B72")
    ax.text((n_sc+len(bars)-1)/2, yt*(0.9 if logy else 0.97),"spatial / cell",ha="center",fontsize=8.5,color="#5B6B72")
from matplotlib.patches import Patch
a1.legend(handles=[Patch(facecolor="#bbbbbb",label="FFPE"),Patch(facecolor="#bbbbbb",hatch="///",edgecolor="white",label="fresh-frozen")],
          loc="upper left",fontsize=8,frameon=False)
a1.set_title("Genes per cell",fontweight="bold",fontsize=12)
a2.set_title("Counts (UMIs) per cell",fontweight="bold",fontsize=12)
fig.suptitle("Per-cell sensitivity: dissociated scRNA-seq vs whole-transcriptome spatial (breast)",
             fontweight="bold",fontsize=13,y=1.03)
fig.tight_layout(); fig.savefig(f"{FIG}/12_scref_vs_spatial.png",bbox_inches="tight"); plt.close(fig)
print("fig 12 (genes + UMIs) written")
for b in bars: print(f"  {b[0]:32s} genes={b[1]:.0f}  UMIs={b[2]:.0f}")
