#!/usr/bin/env python3
"""Gene-abundance distribution across genes (answers 'why 18k genes when bulk sees 10-15k usable?').
For each platform: per-gene total count (pseudobulk rowsum), sorted. Two normalised-axis panels:
 A) sorted per-gene share of total transcripts (log y) vs gene rank (% of detected genes)
 B) Lorenz cumulative: % of transcripts carried by the top X% of genes; marks the 90%-of-signal point.
StrataMap restricted to protein-coding (its 60k GENCODE features include ~31k noncoding)."""
import gzip, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,
                     "savefig.dpi":140,"font.family":"DejaVu Sans"})
ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"; FIG=f"{OUT}/figs"

PLAT=["stdxenium_breast","prime5k_breast","wta_breast","visiumhd65_breast_8um","cosmx_breast","stratamap_breast"]
LAB={"stdxenium_breast":"Xenium 313 (targeted)","prime5k_breast":"Xenium Prime 5K","wta_breast":"Atera (WTx)",
     "visiumhd65_breast_8um":"Visium HD 6.5 (probe)","cosmx_breast":"CosMx (WTx)","stratamap_breast":"StrataMap (poly-A, PC)"}
CLR={"stdxenium_breast":"#1B9E77","prime5k_breast":"#D95F02","wta_breast":"#7570B3",
     "visiumhd65_breast_8um":"#E7298A","cosmx_breast":"#8B5E3C","stratamap_breast":"#2E6FAF"}

# protein-coding symbol set for StrataMap
pc_ensg=set(open(f"{ROOT}/data/stratamap_breast/pc_ensg.txt").read().split())
pc_sym=set()
with gzip.open(f"{ROOT}/data/stratamap_breast/grade3/Raw_Matrix_Files/features.tsv.gz","rt") as fh:
    for line in fh:
        p=line.rstrip("\n").split("\t")
        if p[0].split(".")[0] in pc_ensg and len(p)>1: pc_sym.add(p[1].upper())

def load_counts(name):
    c=[]
    pcf = (name=="stratamap_breast")
    for line in open(f"{OUT}/pseudobulk/{name}.csv"):
        if line.startswith("gene,"): continue
        g,v=line.rstrip().split(",")
        if pcf and g not in pc_sym: continue
        v=float(v)
        if v>0: c.append(v)
    return np.array(sorted(c, reverse=True))

fig,(a,b,c)=plt.subplots(1,3,figsize=(17.5,5.1))
stats={}; viol=[]; vcol=[]; vlab=[]
for n in PLAT:
    ct=load_counts(n); tot=ct.sum(); N=len(ct)
    share=ct/tot*100.0                    # per-gene % of total
    rankpct=np.arange(1,N+1)/N*100.0      # gene rank as % of detected genes
    cum=np.cumsum(ct)/tot*100.0           # cumulative % of transcripts
    n90=int(np.searchsorted(cum,90.0))+1
    n50=int(np.searchsorted(cum,50.0))+1
    stats[n]=dict(detected=N, n50=n50, n90=n90, top=share[0])
    a.plot(rankpct, share, color=CLR[n], lw=2.0, label=LAB[n])
    b.plot(rankpct, cum,   color=CLR[n], lw=2.0, label=LAB[n])
    b.plot(n90/N*100.0, 90.0, "o", color=CLR[n], ms=6, zorder=5)
    viol.append(np.log10(ct/tot*1e6))     # per-gene CPM (depth-normalised), log10 -> distribution
    vcol.append(CLR[n]); vlab.append(LAB[n].split(" (")[0])

a.set_yscale("log"); a.set_xlabel("gene rank  (% of detected genes, most → least expressed)")
a.set_ylabel("share of total transcripts per gene (%, log)")
a.set_title("Sorted per-gene abundance: a few genes dominate",fontweight="bold",fontsize=11.5)
a.legend(fontsize=8.0,loc="upper right")

b.axhline(90,color="#888",ls="--",lw=1.0)
b.set_xlabel("top X% of detected genes"); b.set_ylabel("cumulative % of transcripts")
b.set_title("Concentration (Lorenz): dots = genes carrying 90% of signal",fontweight="bold",fontsize=11.5)
b.set_xlim(0,100); b.set_ylim(0,101); b.legend(fontsize=8.0,loc="lower right")

# violin: distribution of per-gene abundance (log10 CPM), one per platform
parts=c.violinplot(viol, showextrema=False, showmedians=True, widths=0.85)
for body,col in zip(parts["bodies"],vcol):
    body.set_facecolor(col); body.set_edgecolor("#333"); body.set_alpha(0.75)
parts["cmedians"].set_color("#111"); parts["cmedians"].set_linewidth(1.4)
c.set_xticks(range(1,len(vlab)+1)); c.set_xticklabels(vlab,rotation=25,ha="right",fontsize=8.5)
c.set_ylabel("per-gene abundance  (log10 CPM)")
c.set_title("Distribution of per-gene abundance (violin)",fontweight="bold",fontsize=11.5)
fig.suptitle("How transcripts distribute across genes — “detected” ≫ usable",fontweight="bold",fontsize=13,y=1.03)
fig.tight_layout(); fig.savefig(f"{FIG}/24_gene_abundance.png",bbox_inches="tight"); plt.close(fig)

json.dump(stats, open(f"{OUT}/gene_abundance_stats.json","w"), indent=2)
print("fig 24_gene_abundance written")
for n in PLAT:
    s=stats[n]; print(f"  {LAB[n]:26s} detected={s['detected']:6d}  50%={s['n50']:5d}  90%={s['n90']:5d}  top-gene={s['top']:.1f}%")
