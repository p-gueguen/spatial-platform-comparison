#!/usr/bin/env python3
"""StrataMap per-cell aggregation + typing (Grade3 breast) - FAST version.
Bottleneck fix vs script 16: no np.add.at / np.unique. Collect masked (cell,gene,count) into
preallocated buffers during the polars (Rust) batched read, then ONE scipy coo->csr that sums
duplicates - csr row-sums = transcripts/cell, csr nnz-per-row = genes/cell, column slices = markers.
"""
import json, glob, gzip, subprocess
import numpy as np
import polars as pl
import scipy.sparse as sp
from skimage.draw import polygon as draw_polygon

ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"
W=f"{ROOT}/data/stratamap_breast/grade3"; RM=f"{W}/Raw_Matrix_Files"
cf=glob.glob(f"{W}/*Expanded_5um_cell_contour*local.csv")[0]
NNZ=2_644_841_588

MARKERS = {
    "Epithelial/Tumor": ["EPCAM","KRT8","KRT18","KRT19","KRT7","ELF3","CDH1"],
    "T cell":           ["CD3D","CD3E","CD3G","CD2","TRAC","CD8A","CD4","IL7R"],
    "B/Plasma":         ["MS4A1","CD79A","CD79B","IGHG1","MZB1","JCHAIN"],
    "Myeloid":          ["CD68","LYZ","ITGAX","CD14","C1QA","C1QC","CD163"],
    "Fibroblast":       ["COL1A1","COL1A2","DCN","LUM","PDGFRA","PDGFRB"],
    "Endothelial":      ["PECAM1","VWF","CLDN5","CDH5","EGFL7"],
    "Mast":             ["TPSAB1","CPA3","MS4A2"],
}
L1=set(open(f"{OUT}/shared_genes_L1.txt").read().split())
# HGNC symbol drift (must match 02_metrics.py) so GENCODE symbols match the legacy-symbol L1 core.
ALIAS={"KARS1":"KARS","LARS1":"LARS","NARS1":"NARS","QARS1":"QARS","WARS1":"WARS","CYRIA":"FAM49A"}
sym=[""]
with gzip.open(f"{RM}/features.tsv.gz","rt") as fh:
    for line in fh:
        p=line.rstrip("\n").split("\t"); s=(p[1] if len(p)>1 else p[0]).upper(); sym.append(ALIAS.get(s,s))
ngene=len(sym)-1

print("[1] barcodes -> coords + grid", flush=True)
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
n_assigned=int((cell_of_sbc>0).sum())
print(f"    {n_assigned/1e6:.1f}M / {len(cell_of_sbc)/1e6:.1f}M assigned ({100*n_assigned/len(cell_of_sbc):.0f}%)", flush=True)
ncell=int(cid.max())

print("[4] stream matrix -> masked buffers", flush=True)
row=np.empty(NNZ,dtype=np.int32); col=np.empty(NNZ,dtype=np.int32); dat=np.empty(NNZ,dtype=np.int64)
off=0
proc=subprocess.Popen(["bash","-c",f"pigz -dc {RM}/matrix.mtx.gz"], stdout=subprocess.PIPE)
reader=pl.read_csv_batched(proc.stdout, separator=" ", has_header=False, skip_rows=3,
                           new_columns=["g","s","c"], batch_size=80_000_000)
nb=0
while True:
    b=reader.next_batches(1)
    if not b: break
    df=b[0]; nb+=1
    g=df["g"].to_numpy(); s=df["s"].to_numpy(); cnt=df["c"].to_numpy()
    cell=cell_of_sbc[s-1]; m=cell>0; k=int(m.sum())
    row[off:off+k]=cell[m]; col[off:off+k]=g[m]; dat[off:off+k]=cnt[m]; off+=k
    if nb%10==0: print(f"    batch {nb}: {off/1e6:.0f}M masked entries", flush=True)
print(f"    total masked {off/1e6:.0f}M", flush=True)

print("[5] coo -> csr (sums duplicates)", flush=True)
M=sp.coo_matrix((dat[:off],(row[:off],col[:off])), shape=(ncell+1,ngene+1)).tocsr()
del row,col,dat
tx=np.asarray(M.sum(1)).ravel()[1:]
gp=M.getnnz(axis=1)[1:]
def col_sum(genes):
    cols=[i for i in range(1,ngene+1) if sym[i] in genes]
    return np.asarray(M[:,cols].sum(1)).ravel()[1:] if cols else np.zeros(ncell)
L1_tx=col_sum(L1)
det=tx>0
np.save(f"{OUT}/stratamap_percell_tx.npy", tx[det].astype(np.float32))   # per-cell totals for the reads-in-cells plot
print(f"[6] {int(det.sum()):,} detected cells", flush=True)

# marker-argmax typing (same as 02_metrics.py)
safe=np.maximum(tx,1).astype(np.float64); scores=[]; types=[]
for ct,mk in MARKERS.items():
    sc=col_sum(set(mk)).astype(np.float64)
    scores.append(np.log1p(sc/safe*1e4)); types.append(ct)
S=np.vstack(scores); Z=(S-S.mean(1,keepdims=True))/(S.std(1,keepdims=True)+1e-9)
lab=np.array(types)[Z.argmax(0)]; lab[tx<5]="Unassigned"
with open(f"{OUT}/celltype/stratamap_breast.csv","w") as fh:
    fh.write("celltype,n_cells,median_tx_per_cell,median_genes_per_cell,median_L1_tx_per_cell\n")
    for ct in list(MARKERS)+["Unassigned"]:
        s=lab==ct
        if s.sum(): fh.write(f"{ct},{int(s.sum())},{np.median(tx[s]):.3f},{np.median(gp[s]):.3f},{np.median(L1_tx[s]):.3f}\n")

upd=dict(percell_n_cells_total=int(ncell), percell_n_cells_detected=int(det.sum()),
    percell_sbc_assigned_frac=float(n_assigned/len(cell_of_sbc)),
    median_transcripts_per_cell=float(np.median(tx[det])), mean_transcripts_per_cell=float(tx[det].mean()),
    median_genes_per_cell=float(np.median(gp[det])), L1_median_transcripts_per_cell=float(np.median(L1_tx[det])),
    percell_note="SBCs aggregated into the demo Expanded-5um segmentation contours; types = marker-argmax (same as all platforms)")
p=f"{OUT}/per_dataset/stratamap_breast.json"; d=json.load(open(p)); d.update(upd); json.dump(d,open(p,"w"),indent=2)
print("[done]")
for k,v in upd.items():
    if k!="percell_note": print(f"    {k:32s} {v}")
for line in open(f"{OUT}/celltype/stratamap_breast.csv"): print("   ",line.rstrip())
