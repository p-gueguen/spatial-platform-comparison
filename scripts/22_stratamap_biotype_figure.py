#!/usr/bin/env python3
"""Why does the StrataMap matrix have ~62k genes? Because poly(A) sequencing is mapped against the
FULL Ensembl/GENCODE gene annotation, not a probe panel. Only ~1/3 are protein-coding."""
import gzip, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,
                     "savefig.dpi":140,"font.family":"DejaVu Sans"})
R="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; FIG=f"{R}/outputs/figs"

bt={}
for line in open(f"{R}/data/stratamap_breast/ensg_biotype.tsv"):
    p=line.rstrip("\n").split("\t")
    if len(p)==2: bt[p[0].split(".")[0]]=p[1]
ens=[""]
with gzip.open(f"{R}/data/stratamap_breast/grade3/Raw_Matrix_Files/features.tsv.gz","rt") as fh:
    for line in fh: ens.append(line.split("\t",1)[0].split(".")[0])
n=len(ens)-1
pb=np.zeros(n+1)
for line in open(f"{R}/data/stratamap_breast/grade3_pseudobulk_rows.txt"):
    if line.startswith("#"): continue
    a,b=line.split(); pb[int(a)]=float(b)

# collapse to readable classes
def cls(b):
    if b=="protein_coding": return "protein-coding"
    if b=="lncRNA": return "lncRNA"
    if b=="(not in GTF)": return "newer annotation*"
    if "pseudogene" in b: return "pseudogenes"
    if b in ("miRNA","snRNA","snoRNA","misc_RNA","scaRNA","rRNA","rRNA_pseudogene","sRNA","scRNA","vault_RNA","ribozyme"):
        return "small / misc RNA"
    if b.startswith(("IG_","TR_")): return "IG / TR segments"
    if b=="TEC": return "TEC (to be confirmed)"
    return "other"
cnt=collections.Counter(); det=collections.Counter()
for i in range(1,n+1):
    c=cls(bt.get(ens[i],"(not in GTF)")); cnt[c]+=1
    if pb[i]>0: det[c]+=1
order=[k for k,_ in cnt.most_common()]
tot=sum(cnt.values())

fig,(a1,a2)=plt.subplots(1,2,figsize=(13,4.9),gridspec_kw={"width_ratios":[1.35,1]})
y=np.arange(len(order))[::-1]
inm=[cnt[k] for k in order]; dt=[det[k] for k in order]
cols=["#2E6FAF" if k=="protein-coding" else "#B9C6D2" for k in order]
a1.barh(y,inm,color=cols,edgecolor="white",height=0.72,label="in matrix")
a1.barh(y,dt,color=["#12385f" if k=="protein-coding" else "#8697A8" for k in order],
        edgecolor="white",height=0.36,label="detected (>0 counts)")
for yi,k,v in zip(y,order,inm):
    a1.text(v+600,yi,f"{v:,}  ({100*v/tot:.0f}%)",va="center",fontsize=9)
a1.set_yticks(y); a1.set_yticklabels(order,fontsize=10)
a1.set_xlabel("gene models"); a1.set_xlim(0,max(inm)*1.35)
a1.set_title(f"The {tot:,} 'genes' are a full annotation, not a panel",fontweight="bold",fontsize=12)
a1.legend(frameon=False,fontsize=9,loc="lower right")

# right: the comparison that matters
labs=["StrataMap\nmatrix\n(annotation)","StrataMap\nprotein-coding","Atera\npanel","Visium HD\nprobe set","Xenium\npanel"]
vals=[tot, cnt["protein-coding"], 18028, 18085, 313]
cl=["#B9C6D2","#2E6FAF","#7570B3","#66A61E","#1B9E77"]
a2.bar(np.arange(5),vals,color=cl,edgecolor="white")
a2.set_yscale("log")
for xi,v in zip(np.arange(5),vals): a2.text(xi,v,f"{v:,}",ha="center",va="bottom",fontsize=9.5,fontweight="600")
a2.set_xticks(np.arange(5)); a2.set_xticklabels(labs,fontsize=8.6)
a2.set_ylabel("genes (log)")
a2.set_title("Compare like with like",fontweight="bold",fontsize=12)
fig.suptitle("Why the StrataMap matrix has ~62,000 genes",fontweight="bold",fontsize=13.5,y=1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/17_stratamap_biotypes.png",bbox_inches="tight")
print("fig 17 written")
for k in order: print(f"  {k:22s} {cnt[k]:6,}  detected {det[k]:6,}")
