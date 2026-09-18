#!/usr/bin/env python3
"""Assemble StrataMap (Grade3 breast) segmentation-free metrics from the two streaming passes,
matching the pipeline's per_dataset json fields (per-area + per-gene on shared genes).
Per-cell metrics are NOT produced: the demo ships a raw 1um-SBC matrix, not a cell matrix."""
import json, gzip
import numpy as np
ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"
W=f"{ROOT}/data/stratamap_breast"
FEAT=f"{W}/grade3/Raw_Matrix_Files/features.tsv.gz"

# HGNC symbol drift (must match 02_metrics.py): map current symbols back to the legacy ones the
# shared-gene sets use, so StrataMap's GENCODE symbols intersect the flagship core correctly.
ALIAS={"KARS1":"KARS","LARS1":"LARS","NARS1":"NARS","QARS1":"QARS","WARS1":"WARS","CYRIA":"FAM49A"}
norm=lambda g: ALIAS.get(g, g)

# gene symbols per MTX row (1-indexed)
sym=[None]  # index 0 unused
with gzip.open(FEAT,"rt") as fh:
    for line in fh:
        p=line.rstrip("\n").split("\t")
        sym.append(norm(p[1].upper()) if len(p)>1 else norm(p[0].upper()))
n_genes=len(sym)-1

L1=set(open(f"{OUT}/shared_genes_L1.txt").read().split())
L2=set(open(f"{OUT}/shared_genes_L2.txt").read().split())

# protein-coding ENSG (GRCh38.p13 GTF) -> bound the 61,906 GENCODE-everything count to a
# panel-comparable protein-coding breadth (the ~18k probe/panel platforms are protein-coding).
PC=set(open(f"{W}/pc_ensg.txt").read().split())
ensg=[None]  # 1-indexed row -> ENSG (version stripped)
import gzip as _gz
with _gz.open(FEAT,"rt") as fh:
    for line in fh:
        ensg.append(line.split("\t",1)[0].split(".")[0])

# pseudobulk rows -> per-gene totals
pb=np.zeros(n_genes+1, dtype=np.int64); total=None
for line in open(f"{W}/grade3_pseudobulk_rows.txt"):
    if line.startswith("#TOTAL_TX"): total=int(line.split()[1]); continue
    if line.startswith("#"): continue
    r,c=line.split(); pb[int(r)]=int(c)
total_gex=float(pb.sum()) if total is None else float(total)
n_detected=int((pb>0).sum())

# protein-coding bound (panel-comparable breadth) + noncoding bonus
pc_rows=np.array([i for i in range(1,n_genes+1) if ensg[i] in PC])
pc_mask=np.zeros(n_genes+1,bool); pc_mask[pc_rows]=True
n_pc_annotation=int(len(pc_rows))
n_pc_detected=int(((pb>0)&pc_mask).sum())
n_noncoding_detected=n_detected-n_pc_detected
total_pc=float(pb[pc_mask].sum())

# area from 8um-occupied bins (consistent with Visium HD hd_bin area def)
A={}
for line in open(f"{W}/grade3_area.txt"):
    if line.startswith("#"): k,v=line[1:].split(); A[k]=float(v)
n_bin8=A["N_BIN8UM"]; n_sbc=int(A["N_SBC"])
area_mm2=n_bin8*64.0/1e6                      # each 8um bin = 64 um^2
bbox_mm2=((A["YMAX_NM"]-A["YMIN_NM"])*(A["XMAX_NM"]-A["XMIN_NM"]))/1e12  # nm^2 -> mm^2 (sanity)

def shared_block(shared):
    rows=[i for i in range(1,n_genes+1) if sym[i] in shared]
    tot=float(pb[rows].sum())
    ng=len({sym[i] for i in rows})   # distinct symbols (GENCODE PAR genes span 2 rows, e.g. IL3RA)
    return dict(n=ng, total=tot,
                tx_per_mm2=tot/area_mm2,
                tx_per_mm2_per_gene=tot/area_mm2/ng if ng else None)
b1=shared_block(L1); b2=shared_block(L2)

res=dict(
    name="stratamap_breast", platform="Illumina StrataMap", tissue="breast",
    modality="sequencing", kind="stratamap_sbc", unit="1um SBC feature",
    n_units=n_sbc,
    n_gex_genes=n_pc_detected,          # panel-comparable breadth = protein-coding detected
    n_gex_genes_annotation=n_genes,     # full GENCODE annotation (incl. noncoding) - not a fair panel size
    n_pc_annotation=n_pc_annotation, n_pc_detected=n_pc_detected, n_noncoding_detected=n_noncoding_detected,
    area_mm2=area_mm2, area_bbox_mm2=bbox_mm2, n_bin8um=int(n_bin8),
    total_gex_transcripts=total_pc,     # protein-coding basis, comparable to the ~18k-panel platforms
    total_gex_transcripts_all=total_gex, n_genes_detected=n_detected,
    transcripts_per_mm2=total_pc/area_mm2,
    L1_n_genes_present=b1["n"], L1_total_transcripts=b1["total"],
    L1_transcripts_per_mm2=b1["tx_per_mm2"], L1_transcripts_per_mm2_per_gene=b1["tx_per_mm2_per_gene"],
    L2_n_genes_present=b2["n"], L2_total_transcripts=b2["total"],
    L2_transcripts_per_mm2=b2["tx_per_mm2"], L2_transcripts_per_mm2_per_gene=b2["tx_per_mm2_per_gene"],
    # per-cell metrics require cell segmentation aggregation (contour coords) - not computed here
    median_transcripts_per_cell=None, median_genes_per_cell=None,
    note="segmentation-free: raw 1um SBC matrix; area = 8um-occupied bins x 64um^2",
)
json.dump(res, open(f"{OUT}/per_dataset/stratamap_breast.json","w"), indent=2)
# pseudobulk (symbol -> count) for optional concordance
with open(f"{OUT}/pseudobulk/stratamap_breast.csv","w") as fh:
    fh.write("gene,count\n")
    for i in range(1,n_genes+1):
        if pb[i]>0: fh.write(f"{sym[i]},{int(pb[i])}\n")

print("StrataMap (Grade3 breast) segmentation-free metrics:")
for k in ["n_gex_genes","n_units","n_bin8um","area_mm2","area_bbox_mm2","total_gex_transcripts",
          "n_genes_detected","transcripts_per_mm2",
          "L1_n_genes_present","L1_transcripts_per_mm2_per_gene",
          "L2_n_genes_present","L2_transcripts_per_mm2_per_gene"]:
    print(f"  {k:34s} {res[k]}")
