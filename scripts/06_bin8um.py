#!/usr/bin/env python3
"""
Unit-matched comparison: grid each imaging platform's RAW transcripts onto an 8um square
grid - identical units to Visium HD 8um bins, NO segmentation for anyone (molecules->bins).
This bypasses the cell-vs-bin / segmentation confound entirely.

Per dataset -> outputs/bin8um/<name>.json with per-occupied-8um-bin medians
(full panel + shared L1/L2) plus per-area-per-gene (region area).
"""
import polars as pl, json, os, sys, argparse

ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"
os.makedirs(f"{OUT}/bin8um", exist_ok=True)
L1=set(open(f"{OUT}/shared_genes_L1.txt").read().split())
L2=set(open(f"{OUT}/shared_genes_L2.txt").read().split())
# HGNC symbol drift (must match 02_metrics.py): map current symbols back to the legacy ones the
# shared sets use, so e.g. Atera's KARS1 matches L1's KARS.
ALIAS={"KARS1":"KARS","LARS1":"LARS","NARS1":"NARS","QARS1":"QARS","WARS1":"WARS","CYRIA":"FAM49A"}
# control / non-gene feature prefixes (uppercased) to drop when no is_gene flag
CTRL = r"^(NEGCONTROL|BLANK|ANTISENSE|UNASSIGNED|DEPRECATED|INTERGENIC|GENOMICCONTROL|GENOMIC_CONTROL)"

# name -> (transcripts.parquet path, region_area_mm2 from metrics for per-area)
DS = {
 "stdxenium_breast": (f"{ROOT}/data/stdxenium_breast/transcripts.parquet", 41.1187),
 "prime5k_breast":   (f"{ROOT}/data/prime5k_breast/transcripts.parquet", 155.1598),
 "wta_breast":       (f"{ROOT}/data/atera_breast_tx/transcripts.parquet", 58.9444),
}

def bin_one(name, path, area_mm2, qv_min=20.0, bin_um=8.0):
    lf = pl.scan_parquet(path)
    cols = lf.collect_schema().names()
    flt = pl.col("qv") >= qv_min
    if "is_gene" in cols:
        flt = flt & pl.col("is_gene")
    g = pl.col("feature_name").cast(pl.Utf8).str.to_uppercase().replace(ALIAS)
    lf = (lf.filter(flt)
            .with_columns([(pl.col("x_location")/bin_um).floor().cast(pl.Int32).alias("bx"),
                           (pl.col("y_location")/bin_um).floor().cast(pl.Int32).alias("by"),
                           g.alias("g")])
            .filter(~pl.col("g").str.contains(CTRL)))
    # counts per (bin, gene)
    bg = lf.group_by(["bx","by","g"]).agg(pl.len().alias("n")).collect(engine="streaming")
    n_genes_panel = bg["g"].n_unique()
    # per-bin full-panel
    perbin = bg.group_by(["bx","by"]).agg(pl.len().alias("ng"), pl.col("n").sum().alias("nt"))
    n_bins = perbin.height
    res = dict(name=name, unit="8um_bin", n_bins=int(n_bins),
               n_gex_genes=int(n_genes_panel), area_mm2=area_mm2,
               median_genes_per_bin=float(perbin["ng"].median()),
               median_tx_per_bin=float(perbin["nt"].median()),
               total_tx=float(bg["n"].sum()))
    # shared-gene blocks: per-bin sum over shared genes, filled 0 across ALL occupied bins
    for tag, S in [("L1", L1), ("L2", L2)]:
        present = [x for x in bg["g"].unique().to_list() if x in S]
        sub = bg.filter(pl.col("g").is_in(list(S))).group_by(["bx","by"]).agg(pl.col("n").sum().alias("st"))
        # left-join to all occupied bins, fill 0
        j = perbin.select(["bx","by"]).join(sub, on=["bx","by"], how="left").with_columns(pl.col("st").fill_null(0))
        tot = float(j["st"].sum())
        res[f"{tag}_n_genes_present"] = len(present)
        res[f"{tag}_median_tx_per_bin"] = float(j["st"].median())
        res[f"{tag}_transcripts_per_mm2_per_gene"] = tot/area_mm2/len(present) if present else None
    json.dump(res, open(f"{OUT}/bin8um/{name}.json","w"), indent=2)
    print(f"[bin8um] {name}: {n_bins:,} bins, med genes/bin={res['median_genes_per_bin']:.0f}, "
          f"med tx/bin={res['median_tx_per_bin']:.0f}, L1 med tx/bin={res['L1_median_tx_per_bin']:.0f}", flush=True)
    return res

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--only",default=None); a=ap.parse_args()
    for name,(path,area) in DS.items():
        if a.only and name not in a.only.split(","): continue
        if not os.path.exists(path): print(f"[skip] {name}: {path} missing"); continue
        bin_one(name, path, area)
