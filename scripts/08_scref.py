#!/usr/bin/env python3
"""Dissociated scRNA-seq sensitivity reference (breast cancer, different chemistries) from
CELLxGENE Census. Subsamples cells per assay, computes median genes/cell + UMIs/cell from raw counts."""
import cellxgene_census, numpy as np, json
OUT="/srv/GT/analysis/pgueguen/spatial_platform_comparison/outputs"
rng=np.random.default_rng(0)
ASSAYS={"10x 3' v3":"10x 3' v3 (droplet, poly-A)",
        "10x 5' v2":"10x 5' v2 (droplet, poly-A)",
        "Smart-seq2":"Smart-seq2 (plate, full-length)"}
N=3000
census=cellxgene_census.open_soma(census_version="stable")
res={}
for assay,label in ASSAYS.items():
    obs=census["census_data"]["homo_sapiens"].obs.read(
        value_filter=f'tissue_general == "breast" and is_primary_data == True and assay == "{assay}"',
        column_names=["soma_joinid","disease"]).concat().to_pandas()
    canc=obs[obs.disease.str.contains('carcinoma|cancer|tumor|neoplasm',case=False,na=False)]
    if len(canc)==0: canc=obs
    ids=canc.soma_joinid.to_numpy()
    pick=rng.choice(ids, size=min(N,len(ids)), replace=False).tolist()
    ad=cellxgene_census.get_anndata(census,"homo_sapiens", X_name="raw", obs_coords=pick)
    X=ad.X.tocsr()
    genes=np.asarray((X>0).sum(1)).ravel()
    umis=np.asarray(X.sum(1)).ravel()
    res[assay]=dict(label=label, n_cells_total=int(len(canc)), n_sampled=int(X.shape[0]),
                    median_genes_per_cell=float(np.median(genes)),
                    median_umis_per_cell=float(np.median(umis)))
    print(f"{label:34s} n={X.shape[0]:5d}  med genes/cell={np.median(genes):6.0f}  med UMIs/cell={np.median(umis):7.0f}", flush=True)
census.close()
json.dump(res, open(f"{OUT}/scref.json","w"), indent=2)
print("-> scref.json")
