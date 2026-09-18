#!/usr/bin/env python3
"""
CosMx Human Multiomic Breast (Bruker/NanoString WTx, ~18.9k-gene whole transcriptome + 64-plex protein)
recompute, using the SAME metric definitions as 02_metrics.py so it is apples-to-apples with Atera.

Source: exprMat flat file (cell x [genes + Negative* + SystemControl*]). We recompute everything
from raw counts (vendor nCount/nFeature only used as a cross-check).

Metrics computed (mirroring 02_metrics.py imaging block):
  - n cells, n genes, median/mean transcripts per cell, median genes per cell
  - specificity: Negative-probe (nonspecific binding, ~ Atera Negative Control Probe)
                 SystemControl/falsecode (decoding error, ~ Atera Negative Control Codeword)
                 both as normalised rate = per-control-feature-per-cell / per-gene-per-cell (lower=cleaner)
  - imaged area (n FOV placed x FOV footprint) -> transcripts/mm2, cells/mm2  [approximate]
  - shared-gene per-gene sensitivity vs Atera (mean tx/gene/cell on the intersecting gene set)
  - pseudobulk -> Spearman concordance with Atera on shared genes

Output: outputs/per_dataset/cosmx_breast.json + outputs/pseudobulk/cosmx_breast.csv
"""
import json
import numpy as np
import polars as pl

ROOT = "/srv/GT/analysis/pgueguen/spatial_platform_comparison"
OUT  = f"{ROOT}/outputs"
WORK = f"{ROOT}/data/cosmx_breast"
EXPR = f"{WORK}/BreastCancer_exprMat_file.csv.gz"
META = f"{WORK}/BreastCancer_metadata_file.csv.gz"
PX_MM = 0.00012028   # CosMx pixel size, derived from fov_positions (4256 px = 0.51192 mm)

print("[cosmx] reading exprMat header ...", flush=True)
header = pl.read_csv(EXPR, n_rows=0).columns
non_gene = {"fov", "cell_ID", "cell"}
neg_cols  = [c for c in header if c.startswith("Negative")]
sysc_cols = [c for c in header if c.startswith("SystemControl")]
ctrl = set(neg_cols) | set(sysc_cols) | non_gene
gene_cols = [c for c in header if c not in ctrl]
print(f"[cosmx] {len(gene_cols)} genes, {len(neg_cols)} Negative probes, {len(sysc_cols)} SystemControl", flush=True)

print("[cosmx] reading full exprMat (this is the heavy step) ...", flush=True)
df = pl.read_csv(EXPR)
ncells = df.height
print(f"[cosmx] {ncells} cells", flush=True)

# --- gene matrix -> numpy (dense; ~11.5 GB int32, fine on the 1TB node) ---
G = df.select(gene_cols).to_numpy()
tot   = G.sum(1).astype(np.float64)                 # transcripts / cell
ngene = (G > 0).sum(1).astype(np.float64)           # genes / cell
pb    = G.sum(0).astype(np.float64)                 # pseudobulk per gene
total_gex = float(tot.sum())
del G

# --- controls ---
neg_total  = float(df.select(neg_cols).sum_horizontal().sum())   if neg_cols  else 0.0
sysc_total = float(df.select(sysc_cols).sum_horizontal().sum())  if sysc_cols else 0.0

# --- area from FOV footprint ---
# footprint = (max local-x span) x (max local-y span) over cells within FOVs, x px^2
meta = pl.read_csv(META, columns=["fov","CenterX_local_px","CenterY_local_px","Width","Height","Area.um2"],
                   infer_schema_length=5000)
# FOV pixel extent: use the 99.5th percentile of (center + half-size) so a few outliers don't inflate it
fx = (meta["CenterX_local_px"] + meta["Width"]/2).to_numpy()
fy = (meta["CenterY_local_px"] + meta["Height"]/2).to_numpy()
fov_w_px = float(np.percentile(fx, 99.5))
fov_h_px = float(np.percentile(fy, 99.5))
fov_footprint_mm2 = fov_w_px * PX_MM * fov_h_px * PX_MM
n_fov = int(meta["fov"].n_unique())
area_mm2 = n_fov * fov_footprint_mm2
print(f"[cosmx] FOV footprint ~{fov_w_px*PX_MM:.3f} x {fov_h_px*PX_MM:.3f} mm = {fov_footprint_mm2:.4f} mm^2; "
      f"{n_fov} FOV -> imaged area ~{area_mm2:.1f} mm^2", flush=True)

n_genes = len(gene_cols)
per_gene_rate = total_gex / n_genes / ncells
res = dict(
    name="cosmx_breast", platform="CosMx WTx (Bruker)", tissue="breast",
    modality="imaging", kind="cosmx",
    n_units=ncells, unit="cell",
    n_gex_genes=n_genes,
    area_mm2=area_mm2,
    total_gex_transcripts=total_gex,
    median_transcripts_per_cell=float(np.median(tot)),
    mean_transcripts_per_cell=float(tot.mean()),
    median_genes_per_cell=float(np.median(ngene)),
    transcripts_per_mm2=total_gex / area_mm2,
    cells_per_mm2=ncells / area_mm2,
    fov_footprint_mm2=fov_footprint_mm2, n_fov=n_fov,
    # negative probe = nonspecific binding (~ Atera Negative Control Probe)
    negctrl_probe_counts=neg_total, n_negctrl_probes=len(neg_cols),
    negctrl_probe_frac_of_gex=neg_total/total_gex if total_gex else None,
    negctrl_norm_rate=((neg_total/len(neg_cols)/ncells)/per_gene_rate) if neg_cols else None,
    # SystemControl/falsecode = decoding error (~ Atera Negative Control Codeword)
    falsecode_counts=sysc_total, n_falsecodes=len(sysc_cols),
    falsecode_frac_of_gex=sysc_total/total_gex if total_gex else None,
    falsecode_norm_rate=((sysc_total/len(sysc_cols)/ncells)/per_gene_rate) if sysc_cols else None,
    median_cell_area_um2=float(meta["Area.um2"].median()),
)

# --- pseudobulk out (upper-case gene names, matching the rest of the pipeline) ---
gup = [g.upper() for g in gene_cols]
with open(f"{OUT}/pseudobulk/cosmx_breast.csv","w") as fh:
    fh.write("gene,count\n")
    for g,c in zip(gup, pb):
        if c>0: fh.write(f"{g},{int(c)}\n")

# --- shared-gene per-gene sensitivity + concordance vs Atera ---
def load_pb(name):
    d={}
    with open(f"{OUT}/pseudobulk/{name}.csv") as fh:
        next(fh)
        for line in fh:
            g,c=line.rstrip("\n").split(","); d[g]=float(c)
    return d
atera = load_pb("wta_breast")
aj = json.load(open(f"{OUT}/per_dataset/wta_breast.json"))
cos = dict(zip(gup, pb))
shared = sorted(set(atera) & set(cos))
res["shared_with_atera_n_genes"] = len(shared)
# mean tx / gene / cell on the shared set (intensive per-gene sensitivity, no area, no panel-size confound)
cos_sh = float(sum(cos[g] for g in shared))
at_sh  = float(sum(atera[g] for g in shared))
res["shared_atera_mean_tx_per_gene_per_cell_cosmx"] = cos_sh/(len(shared)*ncells)
res["shared_atera_mean_tx_per_gene_per_cell_atera"] = at_sh/(len(shared)*aj["n_units"])
# Spearman on log1p CPM of shared pseudobulk
def cpm_log(pbdict, keys, tot):
    v=np.array([pbdict[g] for g in keys],float); return np.log1p(v/tot*1e6)
from scipy.stats import spearmanr
tc=sum(cos.values()); ta=sum(atera.values())
rho,_=spearmanr(cpm_log(cos,shared,tc), cpm_log(atera,shared,ta))
res["concordance_spearman_vs_atera"] = float(rho)
print(f"[cosmx] shared genes vs Atera: {len(shared)}; "
      f"per-gene tx/cell CosMx={res['shared_atera_mean_tx_per_gene_per_cell_cosmx']:.4f} "
      f"Atera={res['shared_atera_mean_tx_per_gene_per_cell_atera']:.4f}; Spearman={rho:.3f}", flush=True)

json.dump(res, open(f"{OUT}/per_dataset/cosmx_breast.json","w"), indent=2)
print("[cosmx] wrote per_dataset/cosmx_breast.json", flush=True)
for k in ["n_units","n_gex_genes","median_transcripts_per_cell","median_genes_per_cell",
          "transcripts_per_mm2","negctrl_norm_rate","falsecode_norm_rate","area_mm2"]:
    print(f"   {k:42s} {res[k]}")
