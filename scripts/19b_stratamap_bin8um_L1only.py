#!/usr/bin/env python3
"""Fast recompute of ONLY the L1-dependent StrataMap 8um-bin number (L1_median_tx_per_bin) on the
new 304-gene core. Skips 19's O(n log n) distinct-(bin,gene) accumulation - that step feeds
median_genes_per_bin, which is L1-independent and unchanged (1682), so we reuse it from the json.
Single matrix pass, two bincounts (~7 MB), no np.unique on 300M+ arrays."""
import json, gzip, subprocess
import numpy as np
import polars as pl

ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"
W=f"{ROOT}/data/stratamap_breast/grade3"; RM=f"{W}/Raw_Matrix_Files"
L1=set(open(f"{OUT}/shared_genes_L1.txt").read().split())
ALIAS={"KARS1":"KARS","LARS1":"LARS","NARS1":"NARS","QARS1":"QARS","WARS1":"WARS","CYRIA":"FAM49A"}
BIN_UM=8.0

# gene rows -> is_l1 (alias-normalised)
sym=[""]
with gzip.open(f"{RM}/features.tsv.gz","rt") as fh:
    for line in fh:
        p=line.rstrip("\n").split("\t"); s=(p[1] if len(p)>1 else p[0]).upper(); sym.append(ALIAS.get(s,s))
ngene=len(sym)-1
is_l1=np.zeros(ngene+1,bool)
for i in range(1,ngene+1):
    if sym[i] in L1: is_l1[i]=True
print(f"[0] {is_l1.sum()} L1 gene rows (target 304 distinct)", flush=True)

# SBC -> compact 8um bin id (identical to 19)
print("[1] barcodes -> 8um bin id", flush=True)
bc=pl.read_csv(f"{RM}/barcodes.tsv.gz", has_header=False, new_columns=["sbc"])
pr=bc["sbc"].str.split(":")
by=np.floor(pr.list.get(1).cast(pl.Float64).to_numpy()/1000.0/BIN_UM).astype(np.int64)
bx=np.floor(pr.list.get(2).cast(pl.Float64).to_numpy()/1000.0/BIN_UM).astype(np.int64)
NX=int(bx.max())+1
raw=by*NX+bx
uniq,binid=np.unique(raw, return_inverse=True)
B=len(uniq); bin_of_sbc=binid.astype(np.int64)
del raw,by,bx,binid,uniq
print(f"    {B:,} occupied 8um bins", flush=True)

# single matrix pass: per-bin total tx (occupancy + sanity) and per-bin L1 tx
print("[2] stream matrix (bincounts only)", flush=True)
bin_tx=np.zeros(B,dtype=np.int64); bin_l1=np.zeros(B,dtype=np.int64)
proc=subprocess.Popen(["bash","-c",f"pigz -dc {RM}/matrix.mtx.gz"], stdout=subprocess.PIPE)
reader=pl.read_csv_batched(proc.stdout, separator=" ", has_header=False, skip_rows=3,
                           new_columns=["g","s","c"], batch_size=120_000_000)
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
    if nb%4==0: print(f"    batch {nb}", flush=True)
proc.wait()

occ=bin_tx>0
med_tx=float(np.median(bin_tx[occ])); med_l1=float(np.median(bin_l1[occ]))
print(f"[3] occ bins={int(occ.sum()):,}  median_tx_per_bin={med_tx:.0f} (sanity vs 3658)  L1_median_tx_per_bin={med_l1:.0f}", flush=True)

p=f"{OUT}/bin8um/stratamap_breast.json"
d=json.load(open(p))
d["median_tx_per_bin"]=med_tx          # should match the prior 3658 (L1-independent)
d["L1_median_tx_per_bin"]=med_l1       # the recomputed 304-core number
d["n_bins"]=int(occ.sum())
# median_genes_per_bin left as-is (L1-independent; recomputed in full 19 only)
json.dump(d, open(p,"w"), indent=2)
print("[done] wrote", p)
