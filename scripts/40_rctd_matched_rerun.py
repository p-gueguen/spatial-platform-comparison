#!/usr/bin/env python3
"""40_rctd_matched_rerun.py - the vendor-vs-Proseg RCTD comparison, redone like-for-like.

Why this exists. The numbers first reported from 36_rctd_gpu_benchmark.py (StrataMap G1 rejects
55.6% -> 21.2% with Proseg; Atera 8.8% -> 7.5%) compared unlike things in three ways:
  1. Region: the vendor rows were 5,000 cells drawn across the whole section, the Proseg rows came
     from one 1.5 x 1.5 mm window.
  2. Thresholds: 36 scales DOUBLET_THRESHOLD and CONFIDENCE_THRESHOLD by the INPUT matrix's feature
     count (n_vars / 5000). StrataMap's vendor matrix carries 61,906 GENCODE features, its Proseg
     matrix 34,888, Atera 18,028 -> scales of 12.4, 7.0 and 3.6. In doublet mode a cell is rejected
     when neither top type appears in every pair scoring within CONFIDENCE_THRESHOLD of the best
     pair (rctd-py _doublet.py, spacexr RCTD_helper.R), so a larger threshold means MORE rejects.
  3. Gene space: features outside the reference still count toward each cell's nUMI.

Here every condition uses the SAME 1.5 mm window, the input is subset to the reference's genes, and
one threshold scale (18028 / 5000, i.e. Atera's original value) is used for all. Two extra runs
repeat the original per-input scaling on StrataMap, to show how much of the old gap was the config.

Package note: 36 passed sigma_override=0.8, which only early rctd-py releases accepted. This script
runs on rctd-py 0.3.8, where sigma is fitted per dataset (standard RCTD). All six conditions share that
setting, so they compare cleanly with each other; the "orig" rows reproduce 36's threshold scaling,
not its fixed sigma.

Usage:
  python 40_rctd_matched_rerun.py --build            # build StrataMap vendor-segmentation window matrix
  python 40_rctd_matched_rerun.py --run <condition>  # one RCTD run (see RUNS)
  python 40_rctd_matched_rerun.py --summarise        # table over new + original results
"""
import argparse, glob, json, os
import numpy as np
import polars as pl

ROOT = "/srv/GT/analysis/pgueguen/spatial_platform_comparison"
OUT = f"{ROOT}/outputs/rctd_matched"
REF = f"{ROOT}/data/wta_breast/reference.h5ad"      # CELLxGENE Census breast cancer, 10x assays
FIXED_SCALE = 18028 / 5000
# the scale 36 actually applied to each original input (n_vars / 5000 of the h5ad it was given):
# StrataMap vendor 5k subsets carried all 61,906 GENCODE features, the Proseg matrix 34,888
ORIG_SCALE = {"sm_vendor": 61906 / 5000, "sm_proseg": 34888 / 5000}

INPUTS = {
    "atera_vendor": f"{ROOT}/outputs/proseg_atera/atera_roi_standard.h5ad",
    "atera_proseg": f"{ROOT}/outputs/proseg_atera/proseg_atera.h5ad",
    "sm_vendor":    f"{OUT}/sm_vendor_roi.h5ad",
    "sm_proseg":    f"{ROOT}/outputs/proseg_stratamap/proseg_stratamap.h5ad",
}
# condition -> (input, config). "fixed": reference genes only + FIXED_SCALE. "orig": as 36 ran it.
RUNS = {
    "atera_vendor_fixed": ("atera_vendor", "fixed"),
    "atera_proseg_fixed": ("atera_proseg", "fixed"),
    "sm_vendor_fixed":    ("sm_vendor", "fixed"),
    "sm_proseg_fixed":    ("sm_proseg", "fixed"),
    "sm_vendor_orig":     ("sm_vendor", "orig"),
    "sm_proseg_orig":     ("sm_proseg", "orig"),
}

LINEAGE = {  # identical to the map in the original *_gpu_rctd_results.csv files
    "B cell": "B_plasma", "CD4-positive helper T cell": "T_NK",
    "CD4-positive, CD25-positive, CCR4-positive, alpha-beta regulatory T cell": "T_NK",
    "CD8-positive, alpha-beta regulatory T cell": "T_NK", "IgG plasma cell": "B_plasma", "T cell": "T_NK",
    "T follicular helper cell": "T_NK", "adipocyte": "Adipocyte", "blood vessel endothelial cell": "Endothelial",
    "blood vessel smooth muscle cell": "Mural", "capillary endothelial cell": "Endothelial",
    "conventional dendritic cell": "Myeloid", "cycling T cell": "T_NK", "cycling macrophage": "Myeloid",
    "cycling stromal cell": "Fibroblast_stromal", "effector memory CD4-positive, alpha-beta T cell": "T_NK",
    "effector memory CD8-positive, alpha-beta T cell": "T_NK", "endothelial cell": "Endothelial",
    "endothelial cell of artery": "Endothelial", "endothelial cell of vascular tree": "Endothelial",
    "epithelial cell": "Epithelial", "exhausted T cell": "T_NK", "fibroblast": "Fibroblast_stromal",
    "keratinocyte": "Epithelial", "macrophage": "Myeloid", "malignant cell": "Malignant", "mast cell": "Mast",
    "mature NK T cell": "T_NK", "memory B cell": "B_plasma", "monocyte": "Myeloid",
    "mononuclear phagocyte": "Myeloid", "myeloid dendritic cell": "Myeloid", "myofibroblast cell": "Fibroblast_stromal",
    "naive T cell": "T_NK", "natural killer cell": "T_NK", "pericyte": "Mural", "plasma cell": "B_plasma",
    "plasmacytoid dendritic cell": "Myeloid", "vascular associated smooth muscle cell": "Mural",
    "vein endothelial cell": "Endothelial",
}
IMMUNE = {"T_NK", "B_plasma", "Myeloid", "Mast"}


def build_sm_vendor():
    """StrataMap G1 cells from Illumina's Expanded-5um contours, restricted to the Proseg window."""
    import anndata as ad, scipy.sparse as sp
    from PIL import Image, ImageDraw
    os.makedirs(OUT, exist_ok=True)
    pc = pl.read_parquet(f"{ROOT}/outputs/proseg_stratamap/cells.parquet")
    wx0, wx1 = float(pc["centroid_x"].min()), float(pc["centroid_x"].max())
    wy0, wy1 = float(pc["centroid_y"].min()), float(pc["centroid_y"].max())
    t = (pl.scan_parquet(f"{ROOT}/data/proseg_bench/stratamap_roi_transcripts.parquet").filter(pl.col("is_gene"))
         .select(pl.col("feature_name").alias("gene"), pl.col("x_location").alias("x"), pl.col("y_location").alias("y")).collect())
    x0, y0 = float(t["x"].min()) - 20, float(t["y"].min()) - 20
    x1, y1 = float(t["x"].max()) + 20, float(t["y"].max()) + 20
    cf = glob.glob(f"{ROOT}/data/stratamap_breast/grade1/*Expanded_5um_cell_contour_coords_local.csv")[0]
    con = pl.read_csv(cf).filter(pl.col("vertex_x").is_between(x0, x1) & pl.col("vertex_y").is_between(y0, y1))
    W, H = int(np.ceil(x1 - x0)) + 1, int(np.ceil(y1 - y0)) + 1
    img = Image.new("I", (W, H), 0); dr = ImageDraw.Draw(img); cents = {}
    for (cid,), g in con.group_by(["cell_id"]):
        vx, vy = g["vertex_x"].to_numpy(), g["vertex_y"].to_numpy()
        if len(vx) < 3: continue
        dr.polygon(list(zip(vx - x0, vy - y0)), fill=int(cid), outline=int(cid)); cents[int(cid)] = (vx.mean(), vy.mean())
    lab = np.asarray(img, dtype=np.int64)
    px = np.clip(np.round(t["x"].to_numpy() - x0).astype(np.int64), 0, W - 1)
    py = np.clip(np.round(t["y"].to_numpy() - y0).astype(np.int64), 0, H - 1)
    inwin = {c for c, (cx, cy) in cents.items() if wx0 <= cx <= wx1 and wy0 <= cy <= wy1}
    t = t.with_columns(pl.Series("cell", lab[py, px])).filter(pl.col("cell").is_in(list(inwin)))
    agg = t.group_by("cell", "gene").len()
    cells = sorted(agg["cell"].unique().to_list()); genes = sorted(agg["gene"].unique().to_list())
    ci = {c: i for i, c in enumerate(cells)}; gi = {g: i for i, g in enumerate(genes)}
    X = sp.csr_matrix((agg["len"].to_numpy().astype(np.float32),
                       ([ci[c] for c in agg["cell"].to_list()], [gi[g] for g in agg["gene"].to_list()])),
                      shape=(len(cells), len(genes)))
    a = ad.AnnData(X=X)
    a.obs_names = [f"cell_{c}" for c in cells]; a.var_names = genes
    a.obs["cx"] = [cents[c][0] for c in cells]; a.obs["cy"] = [cents[c][1] for c in cells]
    a.write_h5ad(INPUTS["sm_vendor"])
    print(f"sm_vendor_roi: {a.n_obs:,} cells x {a.n_vars:,} features, window x[{wx0:.0f},{wx1:.0f}] y[{wy0:.0f},{wy1:.0f}]")


def run(cond):
    import anndata as ad, pandas as pd, torch
    from rctd import Reference, run_rctd, RCTDConfig
    src, cfg = RUNS[cond]
    sp_ = ad.read_h5ad(INPUTS[src]); sp_.var_names_make_unique()
    ref_ad = ad.read_h5ad(REF); ref = Reference(ref_ad, cell_type_col="cell_type")
    if cfg == "fixed":
        keep = sp_.var_names.isin(ref_ad.var_names)
        sp_ = sp_[:, keep].copy(); scale = FIXED_SCALE
    else:
        scale = ORIG_SCALE[src]                                # what 36 applied to the original input
    conf = RCTDConfig(device="cuda" if torch.cuda.is_available() else "cpu",
                      DOUBLET_THRESHOLD=20.0 * scale, CONFIDENCE_THRESHOLD=5.0 * scale,
                      UMI_min=20)                              # sigma fitted per dataset (see header)
    print(f"{cond}: {sp_.n_obs:,} cells x {sp_.n_vars:,} genes, scale {scale:.2f}, device {conf.device}", flush=True)
    res = run_rctd(sp_, ref, mode="doublet", config=conf, batch_size=8000)
    names = list(res.cell_type_names)
    mask = res.pixel_mask if getattr(res, "pixel_mask", None) is not None else np.ones(sp_.n_obs, bool)
    cls = np.array(["reject", "singlet", "doublet_certain", "doublet_uncertain"])
    sc = np.asarray(res.spot_class)
    df = pd.DataFrame({"barcode": np.asarray(sp_.obs_names)[mask],
                       "spot_class": cls[sc] if np.issubdtype(sc.dtype, np.integer) else sc,
                       "first_type": [names[i] for i in res.first_type],
                       "second_type": [names[i] for i in res.second_type]})
    for i, ct in enumerate(names): df[f"w.{ct}"] = np.asarray(res.weights)[:, i]
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/{cond}.csv", index=False)
    json.dump({"condition": cond, "input": INPUTS[src], "config": cfg, "scale": scale,
               "n_input_cells": int(sp_.n_obs), "n_genes": int(sp_.n_vars), "n_fitted": int(mask.sum())},
              open(f"{OUT}/{cond}.meta.json", "w"), indent=1)
    print(f"{cond}: saved {len(df):,} cells", flush=True)


def summarise_csv(path):
    d = pl.read_csv(path, columns=["spot_class", "first_type"])
    n = d.height
    frac = {k: v / n for k, v in d.group_by("spot_class").len().iter_rows()}
    s = d.filter(pl.col("spot_class") == "singlet").with_columns(pl.col("first_type").replace_strict(LINEAGE).alias("lin"))
    lin = {k: v / max(s.height, 1) for k, v in s.group_by("lin").len().iter_rows()}
    return {"n": n, "reject": frac.get("reject", 0.0), "singlet": frac.get("singlet", 0.0),
            "doublet": frac.get("doublet_certain", 0.0) + frac.get("doublet_uncertain", 0.0),
            "immune_of_singlets": sum(lin.get(k, 0.0) for k in IMMUNE), "T_NK_of_singlets": lin.get("T_NK", 0.0),
            "malignant_of_singlets": lin.get("Malignant", 0.0) + lin.get("Epithelial", 0.0),
            "stroma_of_singlets": lin.get("Fibroblast_stromal", 0.0) + lin.get("Mural", 0.0),
            "endothelial_of_singlets": lin.get("Endothelial", 0.0)}


def summarise():
    old = f"{ROOT}/outputs/rctd_benchmark_gpu"
    rows = []
    for tag, p in [("ORIGINAL Atera section 5k", f"{old}/Atera_FFPE_5k_gpu_rctd_results.csv"),
                   ("ORIGINAL Atera Proseg window 2.5k", f"{old}/Atera_Proseg_2.5k_gpu_rctd_results.csv"),
                   ("ORIGINAL StrataMap G1 section 5k", f"{old}/StrataMap_Grade1_5k_gpu_rctd_results.csv"),
                   ("ORIGINAL StrataMap Proseg window 5k", f"{old}/StrataMap_Proseg_5k_gpu_rctd_results.csv")] + \
                  [(c, f"{OUT}/{c}.csv") for c in RUNS]:
        if os.path.exists(p): rows.append({"condition": tag, **summarise_csv(p)})
    df = pl.DataFrame(rows)
    df.write_csv(f"{OUT}/summary.csv")
    with pl.Config(tbl_rows=-1, tbl_width_chars=200, float_precision=3):
        print(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true"); ap.add_argument("--run", choices=list(RUNS))
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.build: build_sm_vendor()
    if a.run: run(a.run)
    if a.summarise: summarise()
