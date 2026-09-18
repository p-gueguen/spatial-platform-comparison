#!/usr/bin/env python3
"""Unit-matched 8um-bin comparison figure: all FFPE platforms gridded onto identical 8um bins,
NO segmentation. Two panels: breadth (median genes/bin) and per-gene depth (median shared-186 tx/bin)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,
                     "savefig.dpi":140,"font.family":"DejaVu Sans"})
ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"; FIG=f"{OUT}/figs"
D=json.load(open(f"{OUT}/bundle.json"))["datasets"]
L1N=len(open(f"{OUT}/shared_genes_L1.txt").read().split())   # flagship core, now 304
LAB={"stdxenium_breast":"Xenium 313","prime5k_breast":"Prime 5K","wta_breast":"Atera 18k",
     "visiumhd65_breast_8um":"Visium HD 6.5","visiumhd11_breast_8um":"Visium HD 11",
     "stratamap_breast":"StrataMap (FF)"}
CLR={"stdxenium_breast":"#1B9E77","prime5k_breast":"#D95F02","wta_breast":"#7570B3",
     "visiumhd65_breast_8um":"#E7298A","visiumhd11_breast_8um":"#66A61E","stratamap_breast":"#2E6FAF"}
ORDER=["stdxenium_breast","prime5k_breast","wta_breast","visiumhd65_breast_8um","visiumhd11_breast_8um","stratamap_breast"]
FF={"stratamap_breast"}   # fresh-frozen (rest FFPE) -> hatch

def get(name):
    """Return (median_genes_per_bin, median_shared186_tx_per_bin) on identical 8um bins."""
    bp=f"{OUT}/bin8um/{name}.json"
    if os.path.exists(bp):                       # imaging: binned from raw transcripts
        b=json.load(open(bp)); return b["median_genes_per_bin"], b["L1_median_tx_per_bin"]
    d=D[name]                                    # Visium HD: native 8um bins
    return d["median_genes_per_cell"], d["L1_median_transcripts_per_cell"]

genes=[get(n)[0] for n in ORDER]; shared=[max(get(n)[1],0.3) for n in ORDER]  # floor for log vis
cols=[CLR[n] for n in ORDER]; labs=[LAB[n] for n in ORDER]; x=np.arange(len(ORDER))

def _hatch(bars):
    for n,b in zip(ORDER,bars):
        if n in FF: b.set_hatch("///"); b.set_edgecolor("white")
fig,(a1,a2)=plt.subplots(1,2,figsize=(11.5,4.6))
_hatch(a1.bar(x,genes,color=cols,edgecolor="white")); a1.set_yscale("log")
a1.set_title("Breadth on identical 8 µm bins",fontweight="bold",fontsize=12)
a1.set_ylabel("median genes per 8 µm bin (log)")
for xi,v in zip(x,genes): a1.text(xi,v,f"{v:.0f}",ha="center",va="bottom",fontsize=9)
_hatch(a2.bar(x,shared,color=cols,edgecolor="white")); a2.set_yscale("log")
a2.set_title("Per-gene depth on identical 8 µm bins",fontweight="bold",fontsize=12)
a2.set_ylabel(f"median transcripts / 8 µm bin\n({L1N} shared genes, log)")
for xi,v,raw in zip(x,shared,[get(n)[1] for n in ORDER]): a2.text(xi,v,f"{raw:.0f}",ha="center",va="bottom",fontsize=9)
for a in (a1,a2):
    a.set_xticks(x); a.set_xticklabels(labs,rotation=22,ha="right")
fig.suptitle("Same 8 µm grid, no segmentation: breadth vs targeted depth",fontweight="bold",fontsize=13,y=1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/11_bin8um_unitmatched.png",bbox_inches="tight"); plt.close(fig)
print("fig 11_bin8um_unitmatched written")
for n in ORDER: print(f"  {LAB[n]:16s} genes/bin={get(n)[0]:.0f}  shared-186 tx/bin={get(n)[1]:.1f}")
