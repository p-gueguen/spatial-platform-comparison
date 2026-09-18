#!/usr/bin/env python3
"""Per-cell transcript-count distribution (\"reads inside cells\") per sample - violin per platform.
Cell-based samples read from their h5 cell matrices; StrataMap per-cell array read from a .npy dumped
by 17_stratamap_percell_fast.py (run separately, heavy). GEX features only for imaging."""
import os, json
import numpy as np, scipy.sparse as sp, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"; FIG=f"{OUT}/figs"
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,"savefig.dpi":140,"font.family":"DejaVu Sans"})
rng=np.random.RandomState(0)

def percell_h5(path, imaging):
    with h5py.File(path,"r") as f:
        g=f["matrix"]; data=g["data"][:]; indices=g["indices"][:]; indptr=g["indptr"][:]
        shape=tuple(int(x) for x in g["shape"][:])
        ftype=g["features/feature_type"][:].astype(str) if "features/feature_type" in g else np.array(["Gene Expression"]*shape[0])
    M=sp.csc_matrix((data,indices,indptr),shape=shape)  # features x cells
    gex = (ftype=="Gene Expression") if imaging else np.ones(shape[0],bool)
    tot=np.asarray(M[gex,:].sum(0)).ravel().astype(np.float64)
    return tot[tot>0]

# (key, label, colour, h5 path or None, imaging?)
SAMPLES=[
 ("Xenium 313","#1B9E77", f"{ROOT}/data/stdxenium_breast/cell_feature_matrix.h5", True),
 ("Xenium Prime 5K","#D95F02", f"{ROOT}/data/prime5k_breast/cell_feature_matrix.h5", True),
 ("Atera (WTx)","#7570B3", f"{ROOT}/data/wta_breast/cell_feature_matrix.h5", True),
 ("Visium HD 11mm cell","#66A61E", f"{ROOT}/data/visiumhd_11_breast/segmented_outputs/filtered_feature_cell_matrix.h5", False),
 ("Visium HD 6.5mm FF cell","#E7298A", f"{ROOT}/data/visiumhd65ff_breast_seg/segmented_outputs/filtered_feature_cell_matrix.h5", False),
]
SM_NPY=f"{OUT}/stratamap_percell_tx.npy"   # produced by patched 17
data=[]; labs=[]; cols=[]; med=[]
for lab,col,path,img in SAMPLES:
    print(f"[percell] {lab} ...",flush=True)
    tot=percell_h5(path,img)
    if len(tot)>30000: tot=rng.choice(tot,30000,replace=False)
    data.append(np.log10(tot)); labs.append(lab); cols.append(col); med.append(np.median(tot))
if os.path.exists(SM_NPY):
    tot=np.load(SM_NPY); tot=tot[tot>0]
    if len(tot)>30000: tot=rng.choice(tot,30000,replace=False)
    data.append(np.log10(tot)); labs.append("StrataMap (poly-A)"); cols.append("#2E6FAF"); med.append(np.median(tot))
    print("[percell] StrataMap included from npy",flush=True)
else:
    print("[percell] StrataMap npy not yet present - plotting without it",flush=True)

fig,ax=plt.subplots(figsize=(1.5*len(labs)+2,5.2))
parts=ax.violinplot(data,showextrema=False,showmedians=True,widths=0.85)
for b,c in zip(parts["bodies"],cols): b.set_facecolor(c); b.set_edgecolor("#333"); b.set_alpha(0.78)
parts["cmedians"].set_color("#111"); parts["cmedians"].set_linewidth(1.4)
for i,m in enumerate(med,1): ax.text(i,np.log10(m),f" {m:.0f}",va="center",fontsize=9,fontweight="600")
ax.set_xticks(range(1,len(labs)+1)); ax.set_xticklabels(labs,rotation=22,ha="right",fontsize=9.5)
ax.set_ylabel("transcripts per cell (log10)"); ax.set_title("Reads inside cells: per-cell transcript distribution per sample",fontweight="bold",fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIG}/26_reads_in_cells.png",bbox_inches="tight"); plt.close(fig)
json.dump({l:float(m) for l,m in zip(labs,med)}, open(f"{OUT}/reads_in_cells_medians.json","w"), indent=2)
print("fig 26 written:", list(zip(labs,[round(m) for m in med])))
