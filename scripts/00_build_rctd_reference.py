"""00_build_rctd_reference.py - the RCTD reference used by 36/40 (and by the SPLIT step, 37).

Build public scRNA-seq references for the Atera benchmark from CZ CELLxGENE Census.

Breast: tissue_general=='breast' & disease=='breast cancer' (10x assays)  -> rich tumor atlas
Cervix : tissue=='uterine cervix' (all 'normal' in census; no cervical cancer exists)
         -> tissue-matched normal cervix reference

Writes reference.h5ad (cells x genes, raw counts, obs['cell_type']) into each staging dir,
restricted to genes overlapping the Atera 18,028-gene panel.

What this reference is, because the report once mislabelled it: poly-A 10x 3' / 5' scRNA-seq
(breast: 10,689 cells, 40 types, 10x 3' v3 5,168 / 3' v2 4,177 / 5' v2 1,344), NOT Janesick et al.'s
Chromium Flex. Its chemistry is closer to StrataMap (poly-A capture) than to Atera (probes), and its
genes are Atera's panel - both worth holding in mind when reading RCTD numbers across platforms.
Needs `cellxgene-census` and network access to the public Census.
"""
import sys
import h5py
import numpy as np
import cellxgene_census

CENSUS_VERSION = "2025-11-08"
ROOT = "/srv/GT/analysis/pgueguen/spatial_platform_comparison"
ATERA = {  # Atera outs from 01_download.sh; reference.h5ad is written next to them
    "breast": f"{ROOT}/data/wta_breast",
    "cervical": f"{ROOT}/data/wta_cervical",
}

# Per-dataset census query config
CONFIG = {
    "breast": dict(
        value_filter="tissue_general == 'breast' and disease == 'breast cancer' and is_primary_data == True",
        only_10x=True,
        cell_min=25,
        n_per_type=300,
    ),
    "cervical": dict(
        value_filter="tissue == 'uterine cervix' and is_primary_data == True",
        only_10x=False,         # microwell-seq; keep it (only cervix data available)
        cell_min=15,
        n_per_type=300,
    ),
}

TENX_PREFIX = "10x"


def atera_gene_symbols(h5_path):
    with h5py.File(h5_path, "r") as f:
        feats = f["matrix"]["features"]
        names = feats["name"][:].astype(str)
        ftypes = feats["feature_type"][:].astype(str)
    return set(names[ftypes == "Gene Expression"])


def build_one(census, tag):
    cfg = CONFIG[tag]
    outdir = ATERA[tag]
    print(f"\n========== {tag} ==========", flush=True)
    print(f"[{tag}] querying obs metadata ...", flush=True)
    obs = cellxgene_census.get_obs(
        census, "homo_sapiens",
        value_filter=cfg["value_filter"],
        column_names=["soma_joinid", "cell_type", "assay"],
    )
    print(f"[{tag}] obs rows={len(obs):,}  types={obs.cell_type.nunique()}", flush=True)
    if cfg["only_10x"]:
        obs = obs[obs.assay.str.startswith(TENX_PREFIX)]
        print(f"[{tag}] after 10x filter: rows={len(obs):,}  types={obs.cell_type.nunique()}", flush=True)

    rng = np.random.default_rng(0)
    keep_ids = []
    kept_types = []
    for ct, grp in obs.groupby("cell_type", observed=True):
        if len(grp) < cfg["cell_min"]:
            continue
        ids = grp["soma_joinid"].to_numpy()
        if len(ids) > cfg["n_per_type"]:
            ids = rng.choice(ids, cfg["n_per_type"], replace=False)
        keep_ids.extend(int(x) for x in ids)
        kept_types.append((ct, len(ids)))
    keep_ids = sorted(keep_ids)
    print(f"[{tag}] sampled {len(keep_ids):,} cells across {len(kept_types)} types:", flush=True)
    for ct, n in sorted(kept_types, key=lambda x: -x[1]):
        print(f"      {n:>4}  {ct}", flush=True)

    print(f"[{tag}] fetching expression for sampled cells ...", flush=True)
    adata = cellxgene_census.get_anndata(
        census, organism="homo_sapiens", X_name="raw",
        obs_coords=keep_ids,
        column_names={"obs": ["cell_type", "assay", "soma_joinid"],
                      "var": ["feature_id", "feature_name"]},
    )
    # gene symbols as var_names
    adata.var["feature_name"] = adata.var["feature_name"].astype(str)
    adata.var_names = adata.var["feature_name"].values
    adata.var_names_make_unique()

    # intersect with Atera panel symbols
    atera_syms = atera_gene_symbols(f"{ATERA[tag]}/cell_feature_matrix.h5")
    overlap = [g for g in adata.var_names if g in atera_syms]
    print(f"[{tag}] genes: ref={adata.n_vars:,}  Atera panel={len(atera_syms):,}  overlap={len(overlap):,}", flush=True)
    adata = adata[:, overlap].copy()

    # clean obs for RCTD
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    adata.obs["nUMI"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs_names = [f"{tag}_{i}" for i in range(adata.n_obs)]

    out = f"{outdir}/reference.h5ad"
    adata.write_h5ad(out)
    print(f"[{tag}] WROTE {out}  shape={adata.shape}  K={adata.obs.cell_type.nunique()}", flush=True)
    print(f"[{tag}] X dtype={adata.X.dtype} max={adata.X.max():.0f} (raw counts check)", flush=True)


def main():
    print(f"Opening census {CENSUS_VERSION} ...", flush=True)
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        for tag in ("breast", "cervical"):
            try:
                build_one(census, tag)
            except Exception as e:
                print(f"[{tag}] ERROR: {e!r}", flush=True)
                import traceback; traceback.print_exc()
    print("\nREF_BUILD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
