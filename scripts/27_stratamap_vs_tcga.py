#!/usr/bin/env python3
"""StrataMap vs bulk TCGA-BRCA: (1) multi-sample bulk SPREAD (per-sample gene-abundance Lorenz band),
(2) DOWNSAMPLE StrataMap counts to the median TCGA library size and ask whether, at matched depth, its
gene-abundance distribution falls inside the bulk cloud. TCGA from Xena GDC hub as log2(count+1);
reversed to counts. Protein-coding only; TCGA primary tumour (-01) only. StrataMap counts are 1um SBC
features, not reads - the depth match is on COUNT budget, caveat noted in the report."""
import gzip, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,"savefig.dpi":140,"font.family":"DejaVu Sans"})
ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"; FIG=f"{OUT}/figs"
rng=np.random.RandomState(0)

# protein-coding ENSG (version-stripped) from the StrataMap PC list
pc=set(x.split(".")[0] for x in open(f"{ROOT}/data/stratamap_breast/pc_ensg.txt").read().split())

# ---- TCGA-BRCA: load log2(count+1) -> counts, PC genes, primary-tumour columns ----
print("[1] loading TCGA-BRCA star_counts ...", flush=True)
f=gzip.open(f"{ROOT}/data/tcga_brca_star_counts.tsv.gz","rt")
hdr=f.readline().rstrip("\n").split("\t")[1:]
tum=[i for i,s in enumerate(hdr) if s[13:15]=="01"]   # -01 = primary solid tumour
print(f"    {len(hdr)} samples, {len(tum)} primary-tumour", flush=True)
rows=[]
for line in f:
    p=line.rstrip("\n").split("\t")
    if p[0].split(".")[0] in pc:
        v=np.array(p[1:],dtype=np.float32)[tum]
        rows.append(2.0**v - 1.0)   # reverse log2(count+1)
f.close()
T=np.vstack(rows)   # genes x tumour-samples (PC), counts
print(f"    TCGA PC matrix {T.shape} (genes x tumours)", flush=True)
libsize=T.sum(0)                       # per-sample total PC counts
D=int(np.median(libsize)); print(f"    median TCGA-BRCA PC library size = {D:,}", flush=True)

def lorenz(counts, grid):
    c=np.sort(counts[counts>0])[::-1]; cum=np.cumsum(c)/c.sum()*100.0
    x=np.arange(1,len(c)+1)/len(c)*100.0
    return np.interp(grid, x, cum), int(np.searchsorted(cum,90.0))+1, len(c)
grid=np.linspace(0,100,201)
# per-sample TCGA Lorenz band + genes-for-90 distribution
Lc=np.zeros((T.shape[1],len(grid))); n90=np.zeros(T.shape[1]); ndet=np.zeros(T.shape[1])
for j in range(T.shape[1]):
    Lc[j],n90[j],ndet[j]=lorenz(T[:,j],grid)
band_lo,band_md,band_hi=np.percentile(Lc,[10,50,90],axis=0)

# ---- StrataMap PC pseudobulk counts, full + downsampled to D ----
sm={}
# map StrataMap symbols->counts, then need ENSG? pseudobulk is by SYMBOL; use PC-symbol set instead
pc_sym=set()
with gzip.open(f"{ROOT}/data/stratamap_breast/grade3/Raw_Matrix_Files/features.tsv.gz","rt") as fh:
    for line in fh:
        q=line.rstrip("\n").split("\t")
        if q[0].split(".")[0] in pc and len(q)>1: pc_sym.add(q[1].upper())
smc=[]
for line in open(f"{OUT}/pseudobulk/stratamap_breast.csv"):
    if line.startswith("gene,"): continue
    g,c=line.rstrip().split(",")
    if g in pc_sym and float(c)>0: smc.append(float(c))
smc=np.array(smc); tot=smc.sum()
print(f"[2] StrataMap PC pseudobulk: {len(smc)} genes, {tot:,.0f} total counts -> downsample to {D:,}", flush=True)
probs=smc/tot
ds=rng.multinomial(D, probs).astype(np.float64)     # downsampled to TCGA depth
L_full,n90_full,ndet_full=lorenz(smc,grid)
L_ds,n90_ds,ndet_ds=lorenz(ds,grid)

# ---- figure ----
fig,(a,b)=plt.subplots(1,2,figsize=(12.6,5.0))
a.fill_between(grid,band_lo,band_hi,color="#999",alpha=0.28,label="TCGA-BRCA bulk (10-90% band)")
a.plot(grid,band_md,color="#444",lw=1.8,ls="--",label="TCGA-BRCA median")
a.plot(grid,L_ds,color="#2E6FAF",lw=2.4,label=f"StrataMap, downsampled to {D/1e6:.0f}M")
a.plot(grid,L_full,color="#2E6FAF",lw=1.4,ls=":",label="StrataMap, full depth")
a.axhline(90,color="#bbb",ls=":",lw=1)
a.set_xlabel("top X% of detected genes"); a.set_ylabel("cumulative % of signal")
a.set_title("Depth-matched: StrataMap vs the TCGA-BRCA bulk cloud",fontweight="bold",fontsize=12)
a.set_xlim(0,100); a.set_ylim(0,101); a.legend(fontsize=8.6,loc="lower right")
# genes-for-90 distribution
b.hist(n90,bins=40,color="#999",alpha=0.6,label=f"TCGA-BRCA samples (n={len(n90)})")
b.axvline(n90_ds,color="#2E6FAF",lw=2.4,label=f"StrataMap downsampled ({n90_ds:,})")
b.axvline(np.median(n90),color="#444",lw=1.6,ls="--",label=f"TCGA median ({int(np.median(n90)):,})")
b.set_xlabel("genes carrying 90% of signal"); b.set_ylabel("TCGA samples")
b.set_title("Usable-gene count: StrataMap vs bulk spread",fontweight="bold",fontsize=12); b.legend(fontsize=8.6)
fig.suptitle("StrataMap vs bulk TCGA-BRCA (depth-matched, protein-coding)",fontweight="bold",fontsize=13,y=1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/27_stratamap_vs_tcga.png",bbox_inches="tight"); plt.close(fig)

stats=dict(tcga_n_tumours=int(T.shape[1]), tcga_median_libsize=D,
           tcga_n90_median=int(np.median(n90)), tcga_n90_lo=int(np.percentile(n90,10)), tcga_n90_hi=int(np.percentile(n90,90)),
           tcga_ndet_median=int(np.median(ndet)),
           stratamap_downsampled_n90=n90_ds, stratamap_downsampled_ndet=ndet_ds,
           stratamap_full_n90=n90_full, stratamap_full_ndet=ndet_full)
json.dump(stats, open(f"{OUT}/stratamap_vs_tcga_stats.json","w"), indent=2)
print("fig 27 written"); print(json.dumps(stats,indent=2))
