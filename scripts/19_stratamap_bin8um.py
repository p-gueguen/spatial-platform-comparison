#!/usr/bin/env python3
"""StrataMap on the 8um-bin check: grid the raw 1um SBC matrix onto 8um bins (same units as the
imaging transcripts + Visium HD bins, no segmentation). Outputs bin8um/stratamap_breast.json with
median_genes_per_bin (protein-coding, panel-comparable) and L1_median_tx_per_bin (186 shared genes)."""
import json, glob, gzip, subprocess
import numpy as np
import polars as pl

ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"
W=f"{ROOT}/data/stratamap_breast/grade3"; RM=f"{W}/Raw_Matrix_Files"
L1=set(open(f"{OUT}/shared_genes_L1.txt").read().split())
PC=set(open(f"{ROOT}/data/stratamap_breast/pc_ensg.txt").read().split())
# HGNC symbol drift (must match 02_metrics.py) so GENCODE symbols match the legacy-symbol L1 core.
ALIAS={"KARS1":"KARS","LARS1":"LARS","NARS1":"NARS","QARS1":"QARS","WARS1":"WARS","CYRIA":"FAM49A"}
BIN_UM=8.0; PACK=100000

# gene rows -> is_pc / is_l1
sym=[""]; ens=[""]
with gzip.open(f"{RM}/features.tsv.gz","rt") as fh:
    for line in fh:
        p=line.rstrip("\n").split("\t"); ens.append(p[0].split(".")[0]); s=(p[1] if len(p)>1 else p[0]).upper(); sym.append(ALIAS.get(s,s))
ngene=len(sym)-1
is_pc=np.zeros(ngene+1,bool); is_l1=np.zeros(ngene+1,bool)
for i in range(1,ngene+1):
    if ens[i] in PC: is_pc[i]=True
    if sym[i] in L1: is_l1[i]=True
print(f"[0] {is_pc.sum()} PC rows, {is_l1.sum()} L1 rows", flush=True)

# SBC -> compact 8um bin id
print("[1] barcodes -> 8um bin id", flush=True)
bc=pl.read_csv(f"{RM}/barcodes.tsv.gz", has_header=False, new_columns=["sbc"])
pr=bc["sbc"].str.split(":")
by=np.floor(pr.list.get(1).cast(pl.Float64).to_numpy()/1000.0/BIN_UM).astype(np.int64)
bx=np.floor(pr.list.get(2).cast(pl.Float64).to_numpy()/1000.0/BIN_UM).astype(np.int64)
NX=int(bx.max())+1
raw=by*NX+bx
uniq,binid=np.unique(raw, return_inverse=True)   # compact 0..B-1
B=len(uniq); bin_of_sbc=binid.astype(np.int64)
del raw,by,bx,binid,uniq
print(f"    {B:,} occupied 8um bins", flush=True)

# stream matrix -> per-bin total tx, per-bin L1 tx, distinct PC (bin,gene)
print("[2] stream matrix", flush=True)
bin_tx=np.zeros(B,dtype=np.int64); bin_l1=np.zeros(B,dtype=np.int64); keybuf=[]
proc=subprocess.Popen(["bash","-c",f"pigz -dc {RM}/matrix.mtx.gz"], stdout=subprocess.PIPE)
reader=pl.read_csv_batched(proc.stdout, separator=" ", has_header=False, skip_rows=3,
                           new_columns=["g","s","c"], batch_size=80_000_000)
nb=0
while True:
    b=reader.next_batches(1)
    if not b: break
    df=b[0]; nb+=1
    g=df["g"].to_numpy(); s=df["s"].to_numpy(); cnt=df["c"].to_numpy().astype(np.int64)
    bn=bin_of_sbc[s-1]
    bin_tx+=np.bincount(bn, weights=cnt, minlength=B).astype(np.int64)
    ml=is_l1[g]
    if ml.any(): bin_l1+=np.bincount(bn[ml], weights=cnt[ml], minlength=B).astype(np.int64)
    mp=is_pc[g]
    keybuf.append(np.unique(bn[mp].astype(np.int64)*PACK+g[mp].astype(np.int64)))
    if nb%8==0: keybuf=[np.unique(np.concatenate(keybuf))]; print(f"    batch {nb}: {len(keybuf[0])/1e6:.0f}M distinct (bin,PCgene)", flush=True)
keys=np.unique(np.concatenate(keybuf)); del keybuf
pcgenes_per_bin=np.bincount(keys//PACK, minlength=B); del keys

occ=bin_tx>0
res=dict(name="stratamap_breast", unit="8um_bin", n_bins=int(occ.sum()),
         median_genes_per_bin=float(np.median(pcgenes_per_bin[occ])),
         median_tx_per_bin=float(np.median(bin_tx[occ])),
         L1_median_tx_per_bin=float(np.median(bin_l1[occ])),
         note="1um SBC matrix gridded to 8um bins; genes/bin = protein-coding")
json.dump(res, open(f"{OUT}/bin8um/stratamap_breast.json","w"), indent=2)
print("[done]", {k:res[k] for k in ["n_bins","median_genes_per_bin","median_tx_per_bin","L1_median_tx_per_bin"]})
