#!/usr/bin/env python3
"""Figures for the spatial platform comparison Artifact (saved as PNG to outputs/figs/)."""
import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = "/srv/GT/analysis/pgueguen/spatial_platform_comparison"
OUT  = f"{ROOT}/outputs"; FIG = f"{OUT}/figs"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,
                     "figure.dpi":140,"savefig.dpi":140,"font.family":"DejaVu Sans"})

D = json.load(open(f"{OUT}/bundle.json"))["datasets"]
L1N = len(open(f"{OUT}/shared_genes_L1.txt").read().split())   # flagship core (whole-tx), now 304
L2N = len(open(f"{OUT}/shared_genes_L2.txt").read().split())   # high-plex L2, now 4843

LAB = {"stdxenium_breast":"Xenium 313-plex","prime5k_breast":"Xenium Prime 5K",
       "wta_breast":"Atera 18k","visiumhd65_breast_8um":"Visium HD 6.5mm (bin)",
       "visiumhd11_breast_8um":"Visium HD 11mm (bin)","visiumhd11_breast_seg":"Visium HD 11mm (cell)",
       "prime5k_cervical":"Prime 5K (cervix)","wta_cervical":"Atera (cervix)",
       "stratamap_breast":"Illumina StrataMap"}
CLR = {"stdxenium_breast":"#1b9e77","prime5k_breast":"#d95f02","wta_breast":"#7570b3",
       "visiumhd65_breast_8um":"#e7298a","visiumhd11_breast_8um":"#66a61e",
       "visiumhd11_breast_seg":"#a6761d","prime5k_cervical":"#d95f02","wta_cervical":"#7570b3",
       "stratamap_breast":"#2e6faf"}
BREAST5 = ["stdxenium_breast","prime5k_breast","wta_breast","visiumhd65_breast_8um","visiumhd11_breast_8um"]
# StrataMap is native on the segmentation-free axes (per-area, per-gene, breadth) but not the
# per-cell/bin ones (raw 1um-SBC matrix). PA = the platforms comparable on per-area/per-gene.
BREAST_PA = BREAST5 + ["stratamap_breast"]

def bar(names, vals, title, ylab, fname, log=True, fmt="{:.0f}", figsize=(7,4.2)):
    fig,ax=plt.subplots(figsize=figsize)
    x=np.arange(len(names)); cols=[CLR[n] for n in names]
    b=ax.bar(x,vals,color=cols,edgecolor="white",linewidth=0.6)
    if log: ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([LAB[n] for n in names],rotation=22,ha="right")
    ax.set_ylabel(ylab); ax.set_title(title,fontweight="bold",fontsize=12)
    for xi,v in zip(x,vals):
        if v and v==v: ax.text(xi,v,fmt.format(v),ha="center",va="bottom",fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}.png",bbox_inches="tight"); plt.close(fig)
    print("fig:",fname)

# 1) per-gene sensitivity on shared L1 genes (area-normalised) -- HEADLINE (StrataMap native)
# Intensive per-gene metric (tx/mm²/gene): comparable across panels of different size, so Prime 5K
# is shown on the 187 flagship-core genes it carries (it is not part of the 304-gene core itself).
bar(BREAST_PA,[D[n].get("L1_transcripts_per_mm2_per_gene",np.nan) for n in BREAST_PA],
    f"Per-gene sensitivity on the {L1N}-gene flagship core","transcripts / mm² / gene (log)",
    "01_pergene_sensitivity_L1",fmt="{:.0f}")

# 2) median transcripts per cell/bin, PROTEIN-CODING genes only.
# All panel platforms are protein-coding-oriented already; StrataMap is whole-transcriptome, so its
# per-cell total is restricted to protein-coding here (2925) to match - counting all 61,906 GENCODE
# features (incl. ~31k noncoding) gave 3109, only +6%, since noncoding genes are many but low-count.
_f2=BREAST5+["stratamap_breast"]
_f2vals=[D[n].get("median_transcripts_per_cell_pc", D[n]["median_transcripts_per_cell"]) for n in _f2]
bar(_f2,_f2vals,
    "Total molecular yield per cell (protein-coding genes)","median transcripts per cell/bin (log)",
    "02_fullpanel_tx_per_cell",fmt="{:.0f}")

# 3) transcripts per area (full panel, StrataMap native)
bar(BREAST_PA,[D[n]["transcripts_per_mm2"] for n in BREAST_PA],
    "Transcript density (full panel)","transcripts / mm² (log)",
    "03_tx_per_mm2",fmt="{:.2g}")

# 4) breadth vs per-gene depth trade-off (scatter, StrataMap native + highlighted)
fig,ax=plt.subplots(figsize=(6.8,6.8))
_xs=[D[n]["n_gex_genes"] for n in BREAST_PA]
_ys=[D[n].get("L1_transcripts_per_mm2_per_gene",np.nan) for n in BREAST_PA]
_ys=[v for v in _ys if v==v]
for n in BREAST_PA:
    x=D[n]["n_gex_genes"]; y=D[n].get("L1_transcripts_per_mm2_per_gene",np.nan)
    hl = n=="stratamap_breast"
    ax.scatter(x,y,s=(320 if hl else 170),color=CLR[n],edgecolor=("#12385f" if hl else "k"),
               linewidth=(2.0 if hl else 1.0),marker=("*" if hl else "o"),zorder=(5 if hl else 3))
    # points cluster near x~18k; label the whole-tx cluster to the LEFT so nothing runs off the right edge
    right_cluster = x>=4000
    dx,dy=((-12,8) if hl else (-10,6)) if right_cluster else (10,6)
    ax.annotate(LAB[n]+(" ★" if hl else ""),(x,y),textcoords="offset points",xytext=(dx,dy),
                fontsize=(10 if hl else 9),fontweight=("700" if hl else "400"),
                ha=("right" if right_cluster else "left"),color=("#12385f" if hl else "#222"))
ax.set_xscale("log"); ax.set_yscale("log")
# generous limits so the <2-decade breadth spread and the labels both have room (no thin strip)
ax.set_xlim(min(_xs)*0.45, max(_xs)*3.2)
ax.set_ylim(min(_ys)*0.45, max(_ys)*2.3)
ax.set_xlabel("panel breadth (protein-coding genes detected, log)")
ax.set_ylabel("per-gene sensitivity\n(transcripts / mm² / gene, log)")
ax.set_title("Breadth vs per-gene depth: StrataMap breaks the frontier",fontweight="bold",fontsize=12)
# make the shared-gene basis explicit on the figure itself (breadth cannot inflate the y-axis)
ax.text(0.5, -0.155,
        "y-axis is per-gene (transcripts/mm²/gene), area-normalised → panel breadth cannot inflate it.\n"
        f"Whole-transcriptome comparators measured on the identical {L1N}-gene flagship core; Prime 5K on the 187 it carries.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.3, color="#5B6B72", style="italic")
fig.tight_layout(); fig.savefig(f"{FIG}/04_breadth_vs_depth.png",bbox_inches="tight"); plt.close(fig)
print("fig: 04_breadth_vs_depth")

# 5) specificity: negative-control probe fraction (imaging tiers only)
img=["stdxenium_breast","prime5k_breast","wta_breast"]
bar(img,[D[n]["negctrl_probe_frac_of_gex"]*100 for n in img],
    "Background / specificity (imaging platforms)","negative-control-probe counts (% of GEX)",
    "05_specificity_negctrl",log=True,fmt="{:.3f}%",figsize=(5.5,4.2))

# 6) concordance heatmaps
def heatmap_csv(path,title,fname):
    rows=list(csv.reader(open(path))); labels=rows[0][1:]
    M=np.array([[float(x) for x in r[1:]] for r in rows[1:]])
    LMAP={"stdxenium":"Xenium","prime5k":"Prime 5K","wta":"Atera","visiumhd65":"HD 6.5","visiumhd11":"HD 11","stratamap":"StrataMap"}
    sl=[LMAP.get(l.replace("_breast","").replace("_8um",""), l) for l in labels]
    fig,ax=plt.subplots(figsize=(0.9*len(sl)+2,0.9*len(sl)+1.6))
    im=ax.imshow(M,cmap="viridis",vmin=max(0.5,M.min()-0.02),vmax=1.0)
    ax.set_xticks(range(len(sl))); ax.set_xticklabels(sl,rotation=35,ha="right",fontsize=9)
    ax.set_yticks(range(len(sl))); ax.set_yticklabels(sl,fontsize=9)
    for i in range(len(sl)):
        for j in range(len(sl)):
            ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",
                    color="white" if M[i,j]<0.8 else "black",fontsize=9)
    ax.set_title(title,fontweight="bold",fontsize=12)
    fig.colorbar(im,ax=ax,shrink=0.8,label="Spearman r")
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}.png",bbox_inches="tight"); plt.close(fig); print("fig:",fname)
heatmap_csv(f"{OUT}/concordance_L1.csv",f"Cross-platform concordance ({L1N} shared genes)","06_concordance_L1")
heatmap_csv(f"{OUT}/concordance_L2.csv",f"Cross-platform concordance ({L2N} shared genes)","07_concordance_L2")

# 7) per-cell-type sensitivity heatmap (median shared-L1 tx per cell), whole-tx core tiers.
# Per-cell shared-gene total is EXTENSIVE in the gene-set size, so it is only comparable across
# platforms carrying the full core - Prime 5K (187/304) is excluded here (it is in the L2 track).
CORE_CELL=["stdxenium_breast","wta_breast","visiumhd65_breast_8um","visiumhd11_breast_8um"]
cts=["Epithelial/Tumor","T cell","B/Plasma","Myeloid","Fibroblast","Endothelial","Mast"]
mat=np.full((len(cts),len(CORE_CELL)),np.nan)
for j,n in enumerate(CORE_CELL):
    p=f"{OUT}/celltype/{n}.csv"
    if not os.path.exists(p): continue
    dd={r["celltype"]:r for r in csv.DictReader(open(p))}
    for i,ct in enumerate(cts):
        if ct in dd: mat[i,j]=float(dd[ct]["median_L1_tx_per_cell"])
fig,ax=plt.subplots(figsize=(7.5,4.8))
im=ax.imshow(mat,cmap="magma",norm=LogNorm(vmin=np.nanmin(mat[mat>0]) if np.any(mat>0) else 1,vmax=np.nanmax(mat)))
ax.set_xticks(range(len(CORE_CELL))); ax.set_xticklabels([LAB[n] for n in CORE_CELL],rotation=22,ha="right",fontsize=9)
ax.set_yticks(range(len(cts))); ax.set_yticklabels(cts,fontsize=9)
for i in range(len(cts)):
    for j in range(len(CORE_CELL)):
        if mat[i,j]==mat[i,j]: ax.text(j,i,f"{mat[i,j]:.0f}",ha="center",va="center",color="white",fontsize=8)
ax.set_title("Per-cell-type sensitivity (median shared-gene tx / cell)",fontweight="bold",fontsize=12)
fig.colorbar(im,ax=ax,shrink=0.8,label=f"transcripts/cell ({L1N} genes, log)")
fig.tight_layout(); fig.savefig(f"{FIG}/08_celltype_sensitivity.png",bbox_inches="tight"); plt.close(fig); print("fig: 08_celltype_sensitivity")

# 8) breast vs cervical (Prime5K & WTA): per-gene sensitivity on shared genes
fig,ax=plt.subplots(figsize=(6.5,4.2))
groups=["Xenium Prime 5K","Atera"]; x=np.arange(2); w=0.36
br=[D["prime5k_breast"]["L2_transcripts_per_mm2_per_gene"],D["wta_breast"]["L2_transcripts_per_mm2_per_gene"]]
cv=[D["prime5k_cervical"]["L2_transcripts_per_mm2_per_gene"],D["wta_cervical"]["L2_transcripts_per_mm2_per_gene"]]
ax.bar(x-w/2,br,w,label="breast",color="#4477aa"); ax.bar(x+w/2,cv,w,label="cervix",color="#cc6677")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("transcripts / mm² / gene (4842 genes, log)")
ax.set_title("Breast vs cervix (matched high-plex platforms)",fontweight="bold",fontsize=12); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/09_breast_vs_cervix.png",bbox_inches="tight"); plt.close(fig); print("fig: 09_breast_vs_cervix")

print("ALL FIGURES DONE ->", FIG)
