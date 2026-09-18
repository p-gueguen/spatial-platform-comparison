#!/usr/bin/env python3
"""StrataMap per-cell aggregation (Grade3 breast): assign each 1um SBC to a segmented cell via
the Expanded-5um cell contours (rasterised label grid), then aggregate the raw matrix to per-cell
transcripts + genes. Adds median_transcripts_per_cell / median_genes_per_cell to the per_dataset json."""
import json, glob, subprocess
import numpy as np
import polars as pl
from skimage.draw import polygon as draw_polygon

ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"
W=f"{ROOT}/data/stratamap_breast/grade3"; RM=f"{W}/Raw_Matrix_Files"
cf=glob.glob(f"{W}/*Expanded_5um_cell_contour*local.csv")[0]

# --- grid extent (um) from SBC coords (nm) so every SBC maps in-bounds ---
print("[1] barcodes -> SBC coords (um)", flush=True)
bc=pl.read_csv(f"{RM}/barcodes.tsv.gz", has_header=False, new_columns=["sbc"])
parts=bc["sbc"].str.split(":")
ynm=parts.list.get(1).cast(pl.Float64).to_numpy()   # SBC:Y:X
xnm=parts.list.get(2).cast(pl.Float64).to_numpy()
sy=ynm/1000.0; sx=xnm/1000.0
c=pl.read_csv(cf)
xmin=int(np.floor(min(sx.min(), c["vertex_x"].min()))); xmax=int(np.ceil(max(sx.max(), c["vertex_x"].max())))
ymin=int(np.floor(min(sy.min(), c["vertex_y"].min()))); ymax=int(np.ceil(max(sy.max(), c["vertex_y"].max())))
Wd=xmax-xmin+1; Hd=ymax-ymin+1
print(f"    grid {Hd} x {Wd} um  ({Hd*Wd/1e6:.0f} Mpx)", flush=True)

# --- rasterise cell polygons -> label grid (cell_id per um pixel) ---
print("[2] rasterise 718k cell contours", flush=True)
label=np.zeros((Hd,Wd), dtype=np.int32)
cid=c["cell_id"].to_numpy(); vx=c["vertex_x"].to_numpy(); vy=c["vertex_y"].to_numpy()
# split rows into per-cell contiguous blocks (file is grouped by cell_id)
bnd=np.flatnonzero(np.diff(cid))+1; starts=np.r_[0,bnd]; ends=np.r_[bnd,len(cid)]
for s,e in zip(starts,ends):
    rr,cc=draw_polygon(vy[s:e]-ymin, vx[s:e]-xmin, shape=(Hd,Wd))
    label[rr,cc]=cid[s]
print(f"    labelled px: {(label>0).sum()/1e6:.0f} M", flush=True)

# --- assign each SBC to a cell ---
print("[3] assign SBCs -> cells", flush=True)
iy=np.rint(sy-ymin).astype(np.int64); ix=np.rint(sx-xmin).astype(np.int64)
np.clip(iy,0,Hd-1,out=iy); np.clip(ix,0,Wd-1,out=ix)
cell_of_sbc=label[iy,ix]                         # 0 = unassigned
del label,iy,ix,sy,sx,ynm,xnm
n_assigned=int((cell_of_sbc>0).sum())
print(f"    {n_assigned/1e6:.1f} M / {len(cell_of_sbc)/1e6:.1f} M SBCs assigned ({100*n_assigned/len(cell_of_sbc):.0f}%)", flush=True)

# --- stream matrix -> per-cell transcripts + distinct genes ---
print("[4] stream matrix.mtx.gz -> per-cell aggregation", flush=True)
ncell=int(cid.max())
cell_tx=np.zeros(ncell+1, dtype=np.int64)
keybuf=[]; PACK=100000
proc=subprocess.Popen(["bash","-c",f"pigz -dc {RM}/matrix.mtx.gz"], stdout=subprocess.PIPE)
reader=pl.read_csv_batched(proc.stdout, separator=" ", has_header=False, skip_rows=3,
                           new_columns=["g","s","c"], batch_size=40_000_000)
nb=0
while True:
    batches=reader.next_batches(1)
    if not batches: break
    df=batches[0]; nb+=1
    g=df["g"].to_numpy(); s=df["s"].to_numpy(); cnt=df["c"].to_numpy()
    cell=cell_of_sbc[s-1]
    m=cell>0
    cellm=cell[m].astype(np.int64); gm=g[m].astype(np.int64); cm=cnt[m].astype(np.int64)
    np.add.at(cell_tx, cellm, cm)
    keybuf.append(np.unique(cellm*PACK+gm))
    if nb%8==0:
        keybuf=[np.unique(np.concatenate(keybuf))]
        print(f"    batch {nb}: {sum(len(k) for k in keybuf)/1e6:.0f} M distinct (cell,gene) so far", flush=True)
keys=np.unique(np.concatenate(keybuf)); del keybuf
genes_per_cell=np.bincount(keys//PACK, minlength=ncell+1)
del keys

tx=cell_tx[1:]; gp=genes_per_cell[1:]
det=tx>0
res_upd=dict(
    percell_n_cells_total=int(ncell),
    percell_n_cells_detected=int(det.sum()),
    percell_sbc_assigned_frac=float(n_assigned/len(cell_of_sbc)),
    median_transcripts_per_cell=float(np.median(tx[det])),
    mean_transcripts_per_cell=float(tx[det].mean()),
    median_genes_per_cell=float(np.median(gp[det])),
)
p=f"{OUT}/per_dataset/stratamap_breast.json"; d=json.load(open(p)); d.update(res_upd)
json.dump(d, open(p,"w"), indent=2)
print("[done] per-cell metrics merged into stratamap_breast.json:")
for k,v in res_upd.items(): print(f"    {k:34s} {v}")
