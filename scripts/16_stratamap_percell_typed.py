#!/usr/bin/env python3
"""StrataMap per-cell aggregation + cell typing (Grade3 breast), in ONE pass over the raw matrix.

The demo ships segmentation contours but NO cell types, so we type it ourselves with the SAME
marker-score argmax used for every other platform in 02_metrics.py.

Steps: rasterise the Expanded-5um cell contours -> 1um label grid -> assign each SBC to a cell ->
stream matrix.mtx.gz once, accumulating (a) per-cell transcripts, (b) distinct genes per cell,
(c) per-cell counts for the marker + shared-L1 genes (dense, small).
Outputs: per-cell metrics merged into per_dataset/stratamap_breast.json + celltype/stratamap_breast.csv
"""
import json, glob, gzip, subprocess
import numpy as np
import polars as pl
from skimage.draw import polygon as draw_polygon

ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"
W=f"{ROOT}/data/stratamap_breast/grade3"; RM=f"{W}/Raw_Matrix_Files"
cf=glob.glob(f"{W}/*Expanded_5um_cell_contour*local.csv")[0]

MARKERS = {   # identical to 02_metrics.py
    "Epithelial/Tumor": ["EPCAM","KRT8","KRT18","KRT19","KRT7","ELF3","CDH1"],
    "T cell":           ["CD3D","CD3E","CD3G","CD2","TRAC","CD8A","CD4","IL7R"],
    "B/Plasma":         ["MS4A1","CD79A","CD79B","IGHG1","MZB1","JCHAIN"],
    "Myeloid":          ["CD68","LYZ","ITGAX","CD14","C1QA","C1QC","CD163"],
    "Fibroblast":       ["COL1A1","COL1A2","DCN","LUM","PDGFRA","PDGFRB"],
    "Endothelial":      ["PECAM1","VWF","CLDN5","CDH5","EGFL7"],
    "Mast":             ["TPSAB1","CPA3","MS4A2"],
}
L1=set(open(f"{OUT}/shared_genes_L1.txt").read().split())

# --- gene symbols per matrix row (1-indexed) ---
sym=[""]
with gzip.open(f"{RM}/features.tsv.gz","rt") as fh:
    for line in fh:
        p=line.rstrip("\n").split("\t"); sym.append((p[1] if len(p)>1 else p[0]).upper())
ngene=len(sym)-1
marker_syms={g for gs in MARKERS.values() for g in gs}
sel_rows=np.array([i for i in range(1,ngene+1) if sym[i] in L1 or sym[i] in marker_syms], dtype=np.int64)
sel_pos=np.full(ngene+1,-1,dtype=np.int64); sel_pos[sel_rows]=np.arange(len(sel_rows))
is_L1=np.array([sym[i] in L1 for i in sel_rows])
print(f"[0] {len(sel_rows)} selected genes ({is_L1.sum()} shared-L1 + markers)", flush=True)

# --- SBC coords + grid ---
print("[1] barcodes -> SBC coords (um)", flush=True)
bc=pl.read_csv(f"{RM}/barcodes.tsv.gz", has_header=False, new_columns=["sbc"])
pr=bc["sbc"].str.split(":")
sy=pr.list.get(1).cast(pl.Float64).to_numpy()/1000.0   # SBC:Y:X (nm -> um)
sx=pr.list.get(2).cast(pl.Float64).to_numpy()/1000.0
c=pl.read_csv(cf)
xmin=int(np.floor(min(sx.min(), c["vertex_x"].min()))); xmax=int(np.ceil(max(sx.max(), c["vertex_x"].max())))
ymin=int(np.floor(min(sy.min(), c["vertex_y"].min()))); ymax=int(np.ceil(max(sy.max(), c["vertex_y"].max())))
Hd=ymax-ymin+1; Wd=xmax-xmin+1

# --- rasterise contours -> label grid ---
print(f"[2] rasterise contours onto {Hd}x{Wd} um grid", flush=True)
label=np.zeros((Hd,Wd), dtype=np.int32)
cid=c["cell_id"].to_numpy(); vx=c["vertex_x"].to_numpy(); vy=c["vertex_y"].to_numpy()
bnd=np.flatnonzero(np.diff(cid))+1; starts=np.r_[0,bnd]; ends=np.r_[bnd,len(cid)]
for s,e in zip(starts,ends):
    rr,cc=draw_polygon(vy[s:e]-ymin, vx[s:e]-xmin, shape=(Hd,Wd))
    label[rr,cc]=cid[s]

# --- assign SBCs -> cells ---
print("[3] assign SBCs -> cells", flush=True)
iy=np.clip(np.rint(sy-ymin).astype(np.int64),0,Hd-1); ix=np.clip(np.rint(sx-xmin).astype(np.int64),0,Wd-1)
cell_of_sbc=label[iy,ix]
del label,iy,ix,sy,sx
n_assigned=int((cell_of_sbc>0).sum())
print(f"    {n_assigned/1e6:.1f}M / {len(cell_of_sbc)/1e6:.1f}M SBCs in a cell ({100*n_assigned/len(cell_of_sbc):.0f}%)", flush=True)

# --- single streaming pass ---
print("[4] stream matrix once: tx/cell, genes/cell, marker+L1 counts", flush=True)
ncell=int(cid.max())
cell_tx=np.zeros(ncell+1, dtype=np.int64)
cellsel=np.zeros((ncell+1, len(sel_rows)), dtype=np.int32)   # ~718k x ~230
keybuf=[]; PACK=100000
proc=subprocess.Popen(["bash","-c",f"pigz -dc {RM}/matrix.mtx.gz"], stdout=subprocess.PIPE)
reader=pl.read_csv_batched(proc.stdout, separator=" ", has_header=False, skip_rows=3,
                           new_columns=["g","s","c"], batch_size=40_000_000)
nb=0
while True:
    b=reader.next_batches(1)
    if not b: break
    df=b[0]; nb+=1
    g=df["g"].to_numpy().astype(np.int64); s=df["s"].to_numpy(); cnt=df["c"].to_numpy().astype(np.int64)
    cell=cell_of_sbc[s-1].astype(np.int64)
    m=cell>0
    cellm=cell[m]; gm=g[m]; cm=cnt[m]
    np.add.at(cell_tx, cellm, cm)
    keybuf.append(np.unique(cellm*PACK+gm))
    sp=sel_pos[gm]; ms=sp>=0
    if ms.any():
        np.add.at(cellsel.ravel(), cellm[ms]*len(sel_rows)+sp[ms], cm[ms].astype(np.int32))
    if nb%8==0:
        keybuf=[np.unique(np.concatenate(keybuf))]
        print(f"    batch {nb}: {len(keybuf[0])/1e6:.0f}M distinct (cell,gene)", flush=True)
keys=np.unique(np.concatenate(keybuf)); del keybuf
genes_per_cell=np.bincount(keys//PACK, minlength=ncell+1); del keys

# --- per-cell metrics ---
tx=cell_tx[1:]; gp=genes_per_cell[1:]; sel=cellsel[1:]
det=tx>0
L1_tx=sel[:,is_L1].sum(1)
print(f"[5] {det.sum():,} cells with >=1 transcript", flush=True)

# --- cell typing: same marker-score argmax as 02_metrics.py ---
safe=np.maximum(tx,1).astype(np.float64)
scores=[]; types=[]
selsym=[sym[i] for i in sel_rows]
for ct,mk in MARKERS.items():
    cols=[j for j,g in enumerate(selsym) if g in mk]
    if not cols: continue
    sc=sel[:,cols].sum(1).astype(np.float64)
    scores.append(np.log1p(sc/safe*1e4)); types.append(ct)
S=np.vstack(scores)
Z=(S-S.mean(1,keepdims=True))/(S.std(1,keepdims=True)+1e-9)
lab=np.array(types)[Z.argmax(0)]
lab[tx<5]="Unassigned"

with open(f"{OUT}/celltype/stratamap_breast.csv","w") as fh:
    fh.write("celltype,n_cells,median_tx_per_cell,median_genes_per_cell,median_L1_tx_per_cell\n")
    for ct in list(MARKERS)+["Unassigned"]:
        s=lab==ct
        if s.sum()==0: continue
        fh.write(f"{ct},{int(s.sum())},{np.median(tx[s]):.3f},{np.median(gp[s]):.3f},{np.median(L1_tx[s]):.3f}\n")
print("[6] wrote celltype/stratamap_breast.csv", flush=True)

upd=dict(
    percell_n_cells_total=int(ncell), percell_n_cells_detected=int(det.sum()),
    percell_sbc_assigned_frac=float(n_assigned/len(cell_of_sbc)),
    median_transcripts_per_cell=float(np.median(tx[det])),
    mean_transcripts_per_cell=float(tx[det].mean()),
    median_genes_per_cell=float(np.median(gp[det])),
    L1_median_transcripts_per_cell=float(np.median(L1_tx[det])),
    percell_note="SBCs aggregated into the demo's Expanded-5um segmentation contours; cell types = marker-score argmax (same method as all other platforms)",
)
p=f"{OUT}/per_dataset/stratamap_breast.json"; d=json.load(open(p)); d.update(upd); json.dump(d, open(p,"w"), indent=2)
print("[done] per-cell + typing merged:")
for k,v in upd.items():
    if k!="percell_note": print(f"    {k:34s} {v}")
for line in open(f"{OUT}/celltype/stratamap_breast.csv"): print("   ", line.rstrip())
