#!/usr/bin/env python3
"""StrataMap data sanity-check panel: does the data actually look like tissue, did segmentation
work, and are the per-cell distributions sane? (protein-coding genes; grade-3 breast demo)"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams.update({"font.size":11,"axes.grid":False,"savefig.dpi":140,"font.family":"DejaVu Sans"})
ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"; FIG=f"{OUT}/figs"
z=np.load(f"{OUT}/stratamap_viz.npz")
ctx,cgn,cxx,cyy=z["cell_tx"],z["cell_genes"],z["cell_x"],z["cell_y"]
btx,bxx,byy=z["bin_tx"],z["bin_x"],z["bin_y"]
det=ctx>0
print(f"cells {det.sum():,}  bins {(btx>0).sum():,}")

fig,ax=plt.subplots(2,2,figsize=(13.5,11))

# A) tissue: 8um-bin transcript density -> raster image
BIN=8.0
ix=((bxx-bxx.min())/BIN).astype(int); iy=((byy-byy.min())/BIN).astype(int)
img=np.zeros((iy.max()+1, ix.max()+1)); img[iy,ix]=btx
img=np.ma.masked_where(img<=0, img)
a=ax[0,0]
im=a.imshow(img, origin="lower", cmap="magma", norm=LogNorm(vmin=max(1,np.percentile(btx[btx>0],1)),
            vmax=np.percentile(btx[btx>0],99.5)),
            extent=[bxx.min(),bxx.max(),byy.min(),byy.max()], interpolation="nearest")
a.set_title("A. Tissue map: transcripts per 8 µm bin", fontweight="bold", fontsize=12)
a.set_xlabel("x (µm)"); a.set_ylabel("y (µm)"); a.set_aspect("equal")
fig.colorbar(im, ax=a, shrink=0.8, label="transcripts / bin (log)")

# B) segmented cells coloured by transcripts/cell (subsample for rendering)
rng=np.random.default_rng(0)
idx=np.flatnonzero(det);
if len(idx)>250_000: idx=rng.choice(idx,250_000,replace=False)
a=ax[0,1]
s=a.scatter(cxx[idx], cyy[idx], c=ctx[idx], s=0.6, cmap="viridis",
            norm=LogNorm(vmin=max(1,np.percentile(ctx[det],1)), vmax=np.percentile(ctx[det],99)),
            linewidths=0, rasterized=True)
a.set_title(f"B. Segmented cells ({det.sum():,}), coloured by transcripts/cell",fontweight="bold",fontsize=12)
a.set_xlabel("x (µm)"); a.set_ylabel("y (µm)"); a.set_aspect("equal")
fig.colorbar(s, ax=a, shrink=0.8, label="transcripts / cell (log)")

# C) per-cell distributions
a=ax[1,0]; a.grid(True, alpha=0.25, axis="y")
a.hist(ctx[det], bins=np.logspace(0,np.log10(ctx[det].max()),80), color="#2E6FAF", alpha=0.85, label="transcripts / cell")
a.hist(cgn[det], bins=np.logspace(0,np.log10(max(cgn[det].max(),2)),80), color="#D95F02", alpha=0.7, label="genes / cell")
a.set_xscale("log"); a.set_xlabel("per cell (log)"); a.set_ylabel("cells")
a.axvline(np.median(ctx[det]), color="#2E6FAF", ls="--", lw=1.5)
a.axvline(np.median(cgn[det]), color="#D95F02", ls="--", lw=1.5)
a.set_title(f"C. Per-cell distributions (median {np.median(ctx[det]):,.0f} tx / {np.median(cgn[det]):,.0f} genes)",
            fontweight="bold", fontsize=12)
a.legend(frameon=False, fontsize=9.5)

# D) saturation: genes vs transcripts per cell
a=ax[1,1]; a.grid(True, alpha=0.25)
hb=a.hexbin(ctx[det], cgn[det], gridsize=70, bins="log", cmap="Blues", mincnt=1,
            xscale="log", yscale="log")
a.set_xlabel("transcripts / cell (log)"); a.set_ylabel("genes detected / cell (log)")
a.set_title("D. Library complexity: genes vs transcripts per cell",fontweight="bold",fontsize=12)
fig.colorbar(hb, ax=a, shrink=0.8, label="cells (log)")

fig.suptitle("StrataMap data check - breast IDC grade 3 demo (protein-coding genes)",
             fontweight="bold", fontsize=14, y=0.995)
fig.tight_layout()
fig.savefig(f"{FIG}/16_stratamap_datacheck.png", bbox_inches="tight")
print("fig 16 written")
