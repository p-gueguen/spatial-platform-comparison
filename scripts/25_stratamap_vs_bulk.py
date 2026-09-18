#!/usr/bin/env python3
"""StrataMap (poly-A in-situ) vs bulk breast RNA-seq: does the in-situ poly-A capture recapitulate
the bulk gene-abundance distribution? Bulk = HPA consensus nTPM for 'breast' (protein-coding).
NB unit caveat: bulk is length-normalised nTPM, StrataMap is raw molecule counts, so compare the
SHAPE of the curves (concentration across genes), not exact positions."""
import gzip, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,
                     "savefig.dpi":140,"font.family":"DejaVu Sans"})
ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"; FIG=f"{OUT}/figs"

# bulk breast: HPA consensus nTPM (protein-coding by construction)
bulk=[]
with open(f"{ROOT}/data/rna_tissue_consensus.tsv") as fh:
    next(fh)
    for line in fh:
        p=line.rstrip("\n").split("\t")
        if p[2]=="breast" and float(p[3])>0: bulk.append(float(p[3]))
bulk=np.array(sorted(bulk, reverse=True))

# StrataMap protein-coding pseudobulk counts
pc_ensg=set(open(f"{ROOT}/data/stratamap_breast/pc_ensg.txt").read().split())
pc_sym=set()
with gzip.open(f"{ROOT}/data/stratamap_breast/grade3/Raw_Matrix_Files/features.tsv.gz","rt") as fh:
    for line in fh:
        q=line.rstrip("\n").split("\t")
        if q[0].split(".")[0] in pc_ensg and len(q)>1: pc_sym.add(q[1].upper())
sm=[]
for line in open(f"{OUT}/pseudobulk/stratamap_breast.csv"):
    if line.startswith("gene,"): continue
    g,c=line.rstrip().split(",")
    if g in pc_sym and float(c)>0: sm.append(float(c))
sm=np.array(sorted(sm, reverse=True))

SERIES=[("StrataMap (poly-A in-situ)", sm, "#2E6FAF"),
        ("Bulk breast RNA-seq (HPA)", bulk, "#222222")]

fig,(a,b)=plt.subplots(1,2,figsize=(11.6,4.9))
stats={}
for name,arr,col in SERIES:
    tot=arr.sum(); N=len(arr)
    share=arr/tot*100.0; rankpct=np.arange(1,N+1)/N*100.0
    cum=np.cumsum(arr)/tot*100.0
    n90=int(np.searchsorted(cum,90.0))+1; n50=int(np.searchsorted(cum,50.0))+1
    stats[name]=dict(detected=N,n50=n50,n90=n90,top=share[0])
    ls="-" if "StrataMap" in name else "--"
    a.plot(rankpct, share, color=col, lw=2.2, ls=ls, label=name)
    b.plot(rankpct, cum,   color=col, lw=2.2, ls=ls, label=name)
    b.plot(n90/N*100.0, 90.0, "o", color=col, ms=7, zorder=5)

a.set_yscale("log"); a.set_xlabel("gene rank (% of detected genes, most → least expressed)")
a.set_ylabel("share of total signal per gene (%, log)")
a.set_title("Sorted per-gene abundance",fontweight="bold",fontsize=12); a.legend(fontsize=9)
b.axhline(90,color="#888",ls=":",lw=1.0)
b.set_xlabel("top X% of detected genes"); b.set_ylabel("cumulative % of signal")
b.set_title("Concentration (Lorenz): dots = genes carrying 90%",fontweight="bold",fontsize=12)
b.set_xlim(0,100); b.set_ylim(0,101); b.legend(fontsize=9,loc="lower right")
fig.suptitle("StrataMap recapitulates the bulk RNA-seq gene-abundance distribution",fontweight="bold",fontsize=13,y=1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/25_stratamap_vs_bulk.png",bbox_inches="tight"); plt.close(fig)
json.dump(stats, open(f"{OUT}/stratamap_vs_bulk_stats.json","w"), indent=2)
print("fig 25 written")
for k,v in stats.items(): print(f"  {k:28s} detected={v['detected']}  90%={v['n90']}  50%={v['n50']}  top={v['top']:.1f}%")
