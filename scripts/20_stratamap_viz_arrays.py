#!/usr/bin/env python3
"""Recompute + SAVE the StrataMap per-cell and per-8um-bin arrays needed for visual QC plots
(earlier passes only kept medians). Fast path: polars batched read + scipy coo->csr.
Saves outputs/stratamap_viz.npz with per-cell tx/genes/centroids and per-8um-bin tx/coords."""
import glob, gzip, subprocess
import numpy as np
import polars as pl
import scipy.sparse as sp
from skimage.draw import polygon as draw_polygon

ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"
W=f"{ROOT}/data/stratamap_breast/grade3"; RM=f"{W}/Raw_Matrix_Files"
cf=glob.glob(f"{W}/*Expanded_5um_cell_contour*local.csv")[0]
NNZ=2_644_841_588; BIN_UM=8.0

PC=set(open(f"{ROOT}/data/stratamap_breast/pc_ensg.txt").read().split())
ens=[""]
with gzip.open(f"{RM}/features.tsv.gz","rt") as fh:
    for line in fh: ens.append(line.split("\t",1)[0].split(".")[0])
ngene=len(ens)-1
is_pc=np.zeros(ngene+1,bool)
for i in range(1,ngene+1):
    if ens[i] in PC: is_pc[i]=True

print("[1] SBC coords", flush=True)
bc=pl.read_csv(f"{RM}/barcodes.tsv.gz", has_header=False, new_columns=["sbc"])
pr=bc["sbc"].str.split(":")
sy=pr.list.get(1).cast(pl.Float64).to_numpy()/1000.0   # um
sx=pr.list.get(2).cast(pl.Float64).to_numpy()/1000.0
c=pl.read_csv(cf)
xmin=int(np.floor(min(sx.min(),c["vertex_x"].min()))); xmax=int(np.ceil(max(sx.max(),c["vertex_x"].max())))
ymin=int(np.floor(min(sy.min(),c["vertex_y"].min()))); ymax=int(np.ceil(max(sy.max(),c["vertex_y"].max())))
Hd=ymax-ymin+1; Wd=xmax-xmin+1

print("[2] rasterise contours + cell centroids", flush=True)
label=np.zeros((Hd,Wd),dtype=np.int32)
cid=c["cell_id"].to_numpy(); vx=c["vertex_x"].to_numpy(); vy=c["vertex_y"].to_numpy()
bnd=np.flatnonzero(np.diff(cid))+1; starts=np.r_[0,bnd]; ends=np.r_[bnd,len(cid)]
ncell=int(cid.max())
cx=np.zeros(ncell+1); cy=np.zeros(ncell+1)
for s,e in zip(starts,ends):
    rr,cc=draw_polygon(vy[s:e]-ymin, vx[s:e]-xmin, shape=(Hd,Wd)); label[rr,cc]=cid[s]
    cx[cid[s]]=vx[s:e].mean(); cy[cid[s]]=vy[s:e].mean()

print("[3] assign SBCs", flush=True)
iy=np.clip(np.rint(sy-ymin).astype(np.int64),0,Hd-1); ix=np.clip(np.rint(sx-xmin).astype(np.int64),0,Wd-1)
cell_of_sbc=label[iy,ix]; del label
# 8um bin id per SBC (compact)
by=np.floor(sy/BIN_UM).astype(np.int64); bx=np.floor(sx/BIN_UM).astype(np.int64)
NX=int(bx.max())+1
uniqb,binid=np.unique(by*NX+bx, return_inverse=True)
B=len(uniqb); bin_of_sbc=binid.astype(np.int64)
bin_y=(uniqb//NX)*BIN_UM; bin_x=(uniqb%NX)*BIN_UM
del by,bx,binid,uniqb,iy,ix,sy,sx

print("[4] stream matrix -> cell x gene csr + per-bin totals", flush=True)
row=np.empty(NNZ,dtype=np.int32); col=np.empty(NNZ,dtype=np.int32); dat=np.empty(NNZ,dtype=np.int64)
bin_tx=np.zeros(B,dtype=np.int64); off=0
proc=subprocess.Popen(["bash","-c",f"pigz -dc {RM}/matrix.mtx.gz"], stdout=subprocess.PIPE)
reader=pl.read_csv_batched(proc.stdout, separator=" ", has_header=False, skip_rows=3,
                           new_columns=["g","s","c"], batch_size=80_000_000)
nb=0
while True:
    b=reader.next_batches(1)
    if not b: break
    df=b[0]; nb+=1
    g=df["g"].to_numpy(); s=df["s"].to_numpy(); cnt=df["c"].to_numpy().astype(np.int64)
    bin_tx+=np.bincount(bin_of_sbc[s-1], weights=cnt, minlength=B).astype(np.int64)
    cell=cell_of_sbc[s-1]; m=(cell>0)&is_pc[g]          # protein-coding, inside a cell
    k=int(m.sum())
    row[off:off+k]=cell[m]; col[off:off+k]=g[m]; dat[off:off+k]=cnt[m]; off+=k
    if nb%10==0: print(f"    batch {nb}: {off/1e6:.0f}M", flush=True)

print("[5] coo -> csr", flush=True)
M=sp.coo_matrix((dat[:off],(row[:off],col[:off])), shape=(ncell+1,ngene+1)).tocsr()
del row,col,dat
cell_tx=np.asarray(M.sum(1)).ravel()[1:]
cell_genes=M.getnnz(axis=1)[1:]
np.savez_compressed(f"{OUT}/stratamap_viz.npz",
                    cell_tx=cell_tx, cell_genes=cell_genes, cell_x=cx[1:], cell_y=cy[1:],
                    bin_tx=bin_tx, bin_x=bin_x, bin_y=bin_y)
print("[done] cells:", (cell_tx>0).sum(), "bins:", (bin_tx>0).sum(),
      "median tx/cell:", np.median(cell_tx[cell_tx>0]), "median genes/cell:", np.median(cell_genes[cell_tx>0]))
