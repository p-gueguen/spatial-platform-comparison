#!/usr/bin/env python3
"""Recompute StrataMap per-cell molecular yield restricted to PROTEIN-CODING genes.

The full-panel per-cell median (3109) in 17_stratamap_percell_fast.py sums ALL ~61,906 GENCODE
features, incl. ~30,891 detected noncoding genes that the targeted protein-coding panels (Xenium,
Atera, Visium HD) physically cannot see. That makes the "total molecular yield per cell" bar
apples-to-oranges. Here: identical cell segmentation + SBC assignment, but sum only protein-coding
columns. Also reproduce the all-gene total as a sanity check (must == 3109).

Lean: per-cell totals via streaming bincount (no giant buffers, no np.add.at). Transcripts only -
that is exactly what the plot shows (median transcripts per cell).
"""
import json, glob, gzip, subprocess
import numpy as np
import polars as pl
from skimage.draw import polygon as draw_polygon

ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"
W=f"{ROOT}/data/stratamap_breast/grade3"; RM=f"{W}/Raw_Matrix_Files"
PC=set(open(f"{ROOT}/data/stratamap_breast/pc_ensg.txt").read().split())
cf=glob.glob(f"{W}/*Expanded_5um_cell_contour*local.csv")[0]

# gene rows: ENSG (col0, version-stripped) -> protein-coding mask (1-indexed)
ensg=[""]
with gzip.open(f"{RM}/features.tsv.gz","rt") as fh:
    for line in fh:
        ensg.append(line.split("\t",1)[0].split(".")[0])
ngene=len(ensg)-1
pc_gene=np.zeros(ngene+1, dtype=bool)
for i in range(1,ngene+1):
    if ensg[i] in PC: pc_gene[i]=True
print(f"[0] {pc_gene.sum():,} / {ngene:,} gene rows are protein-coding", flush=True)

print("[1] barcodes -> coords", flush=True)
bc=pl.read_csv(f"{RM}/barcodes.tsv.gz", has_header=False, new_columns=["sbc"])
pr=bc["sbc"].str.split(":")
sy=pr.list.get(1).cast(pl.Float64).to_numpy()/1000.0
sx=pr.list.get(2).cast(pl.Float64).to_numpy()/1000.0
c=pl.read_csv(cf)
xmin=int(np.floor(min(sx.min(),c["vertex_x"].min()))); xmax=int(np.ceil(max(sx.max(),c["vertex_x"].max())))
ymin=int(np.floor(min(sy.min(),c["vertex_y"].min()))); ymax=int(np.ceil(max(sy.max(),c["vertex_y"].max())))
Hd=ymax-ymin+1; Wd=xmax-xmin+1

print(f"[2] rasterise contours ({Hd}x{Wd})", flush=True)
label=np.zeros((Hd,Wd),dtype=np.int32)
cid=c["cell_id"].to_numpy(); vx=c["vertex_x"].to_numpy(); vy=c["vertex_y"].to_numpy()
bnd=np.flatnonzero(np.diff(cid))+1
for s,e in zip(np.r_[0,bnd], np.r_[bnd,len(cid)]):
    rr,cc=draw_polygon(vy[s:e]-ymin, vx[s:e]-xmin, shape=(Hd,Wd)); label[rr,cc]=cid[s]

print("[3] assign SBCs -> cells", flush=True)
iy=np.clip(np.rint(sy-ymin).astype(np.int64),0,Hd-1); ix=np.clip(np.rint(sx-xmin).astype(np.int64),0,Wd-1)
cell_of_sbc=label[iy,ix]; del label,iy,ix,sy,sx
ncell=int(cid.max())

print("[4] stream matrix -> per-cell bincount (all + protein-coding)", flush=True)
txall=np.zeros(ncell+1, dtype=np.float64)   # all genes  -> reproduce 3109
txpc =np.zeros(ncell+1, dtype=np.float64)   # protein-coding only
proc=subprocess.Popen(["bash","-c",f"pigz -dc {RM}/matrix.mtx.gz"], stdout=subprocess.PIPE)
reader=pl.read_csv_batched(proc.stdout, separator=" ", has_header=False, skip_rows=3,
                           new_columns=["g","s","c"], batch_size=80_000_000)
nb=0
while True:
    b=reader.next_batches(1)
    if not b: break
    df=b[0]; nb+=1
    g=df["g"].to_numpy(); s=df["s"].to_numpy(); cnt=df["c"].to_numpy().astype(np.float64)
    cell=cell_of_sbc[s-1]; m=cell>0
    cm=cell[m]; qm=cnt[m]; gm=g[m]
    txall+=np.bincount(cm, weights=qm, minlength=ncell+1)
    pcm=pc_gene[gm]
    txpc+=np.bincount(cm[pcm], weights=qm[pcm], minlength=ncell+1)
    if nb%10==0: print(f"    batch {nb}", flush=True)

det=txall[1:]>0            # detected cells - SAME definition as the all-gene 3109
ta=txall[1:][det]; tp=txpc[1:][det]
med_all=float(np.median(ta)); med_pc=float(np.median(tp))
print(f"[5] detected cells: {det.sum():,}")
print(f"    median transcripts/cell  ALL genes        = {med_all:.0f}   (must match 3109)")
print(f"    median transcripts/cell  PROTEIN-CODING   = {med_pc:.0f}")
print(f"    protein-coding share of per-cell yield    = {100*med_pc/med_all:.1f}%")

p=f"{OUT}/per_dataset/stratamap_breast.json"; d=json.load(open(p))
d["median_transcripts_per_cell_pc"]=med_pc
d["median_transcripts_per_cell_allgenes_check"]=med_all
json.dump(d, open(p,"w"), indent=2)
print("[done] wrote median_transcripts_per_cell_pc ->", p)
