#!/usr/bin/env python3
"""Per-cell shared-gene depth: median transcripts per CELL on the 186 shared marker genes,
across the cell-based platforms. The counterpoint to per-AREA sensitivity - targeted panels
concentrate more reads on specific genes per cell, even where whole-tx wins per mm2.
Fresh-frozen bars hatched (prep lifts per-cell depth)."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,
                     "savefig.dpi":140,"font.family":"DejaVu Sans"})
ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"; FIG=f"{OUT}/figs"
D=json.load(open(f"{OUT}/bundle.json"))["datasets"]
L1N=len(open(f"{OUT}/shared_genes_L1.txt").read().split())   # flagship core, now 304

# (key, label, colour, fresh-frozen?). Per-cell shared total is EXTENSIVE in gene-set size, so only
# platforms carrying the full flagship core appear; Prime 5K (187/304) is excluded (it is in L2).
rows=[
 ("stdxenium_breast","Xenium 313 (targeted)","#1b9e77",False),
 ("wta_breast","Atera (whole-tx)","#7570b3",False),
 ("stratamap_breast","StrataMap (whole-tx)","#2e6faf",True),
 ("visiumhd11_breast_seg","Visium HD 11mm cell","#a6761d",False),
 ("visiumhd65ff_breast_seg","Visium HD 6.5mm cell","#e7298a",True),
]
rows=[(k,l,c,ff,D[k]["L1_median_transcripts_per_cell"]) for k,l,c,ff in rows]
rows.sort(key=lambda r:-r[4])                       # descending
labs=[f"{l}{' (FF)' if ff else ''}" for _,l,_,ff,_ in rows]
vals=[v for *_,v in rows]; cols=[c for _,_,c,_,_ in rows]; ffs=[ff for _,_,_,ff,_ in rows]

fig,ax=plt.subplots(figsize=(8.6,4.8))
x=np.arange(len(rows))
bars=ax.bar(x,vals,color=cols,edgecolor="white",linewidth=0.8)
for b,ff in zip(bars,ffs):
    if ff: b.set_hatch("///"); b.set_edgecolor("#ffffff")
for xi,v in zip(x,vals): ax.text(xi,v,f"{v:.0f}",ha="center",va="bottom",fontsize=10,fontweight="600")
ax.set_xticks(x); ax.set_xticklabels(labs,rotation=22,ha="right",fontsize=9.5)
ax.set_ylabel(f"median transcripts per cell\n({L1N} shared marker genes)")
ax.set_title("Per-cell depth on specific genes: targeted panels concentrate the reads",
             fontweight="bold",fontsize=12)
ax.set_ylim(0,max(vals)*1.15)
ax.legend(handles=[Patch(facecolor="#bbbbbb",label="FFPE"),
                   Patch(facecolor="#bbbbbb",hatch="///",edgecolor="white",label="fresh-frozen")],
          loc="upper right",fontsize=9,frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/15_percell_sharedgene.png",bbox_inches="tight"); plt.close(fig)
print("fig 15 written:", [(l,v) for l,v in zip(labs,vals)])
