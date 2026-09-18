#!/usr/bin/env python3
"""
Aggregate per-dataset metrics into tidy tables + cross-platform concordance.
Run after 02_metrics.py has produced all per_dataset/*.json and pseudobulk/*.csv.

Outputs:
  outputs/master_metrics.csv      one row per dataset, all key metrics
  outputs/concordance_L1.csv      Spearman corr matrix on shared L1 genes (all 5 tiers)
  outputs/concordance_L2.csv      Spearman corr matrix on shared L2 genes (high-plex)
  outputs/bundle.json             everything the Artifact needs, in one file
"""
import os, json, glob
import numpy as np

ROOT = "/srv/GT/analysis/pgueguen/spatial_platform_comparison"
OUT  = f"{ROOT}/outputs"

ORDER = ["stdxenium_breast","prime5k_breast","wta_breast",
         "visiumhd65_breast_8um","visiumhd11_breast_8um","visiumhd11_breast_seg",
         "prime5k_cervical","wta_cervical"]

COLS = ["name","platform","tissue","modality","kind","unit","n_units","n_gex_genes",
        "area_mm2","total_gex_transcripts","median_transcripts_per_cell","mean_transcripts_per_cell",
        "median_genes_per_cell","transcripts_per_mm2","cells_per_mm2",
        "negctrl_probe_frac_of_gex","negctrl_codeword_frac_of_gex","negctrl_norm_rate",
        "L1_n_genes_present","L1_median_transcripts_per_cell","L1_median_genes_per_cell",
        "L1_mean_tx_per_gene_per_cell","L1_transcripts_per_mm2","L1_transcripts_per_mm2_per_gene",
        "L1_frac_genes_detected_per_cell",
        "L2_n_genes_present","L2_median_transcripts_per_cell","L2_mean_tx_per_gene_per_cell",
        "L2_transcripts_per_mm2","L2_transcripts_per_mm2_per_gene"]

def load_all():
    out = {}
    for p in glob.glob(f"{OUT}/per_dataset/*.json"):
        d = json.load(open(p)); out[d["name"]] = d
    return out

def master_table(D):
    names = [n for n in ORDER if n in D] + [n for n in D if n not in ORDER]
    with open(f"{OUT}/master_metrics.csv","w") as fh:
        fh.write(",".join(COLS)+"\n")
        for n in names:
            d = D[n]
            row = []
            for c in COLS:
                v = d.get(c)
                row.append("" if v is None else (f"{v:.6g}" if isinstance(v,float) else str(v)))
            fh.write(",".join(row)+"\n")
    print(f"[master] {len(names)} datasets -> master_metrics.csv")
    return names

def concordance(shared_file, tag, members):
    genes = sorted(set(open(shared_file).read().split()))
    gidx = {g:i for i,g in enumerate(genes)}
    mats, labels = [], []
    for n in members:
        f = f"{OUT}/pseudobulk/{n}.csv"
        if not os.path.exists(f): continue
        v = np.zeros(len(genes))
        with open(f) as fh:
            next(fh)
            for line in fh:
                g,c = line.rstrip().split(",")
                if g in gidx: v[gidx[g]] = float(c)
        if v.sum()==0: continue
        v = v / v.sum() * 1e6          # CPM over shared genes
        mats.append(np.log1p(v)); labels.append(n)
    if len(mats) < 2:
        print(f"[concordance {tag}] <2 datasets, skip"); return
    M = np.vstack(mats)                # datasets x genes
    # Spearman = Pearson on ranks
    R = np.vstack([rankdata(r) for r in M])
    C = np.corrcoef(R)
    with open(f"{OUT}/concordance_{tag}.csv","w") as fh:
        fh.write(","+",".join(labels)+"\n")
        for i,l in enumerate(labels):
            fh.write(l+","+",".join(f"{C[i,j]:.4f}" for j in range(len(labels)))+"\n")
    print(f"[concordance {tag}] {len(labels)} datasets, {len(genes)} genes -> concordance_{tag}.csv")

def rankdata(a):
    order = a.argsort(); ranks = np.empty_like(order, float); ranks[order] = np.arange(len(a))
    return ranks

def main():
    D = load_all()
    names = master_table(D)
    # concordance among breast tiers (cell/bin level pseudobulk)
    if os.path.exists(f"{OUT}/shared_genes_L1.txt"):
        # L1 concordance = flagship whole-tx core only; Prime 5K (187/304) would zero-fill the
        # missing genes and spuriously deflate its correlation, so it is scored in L2 instead.
        concordance(f"{OUT}/shared_genes_L1.txt","L1",
                    ["stdxenium_breast","wta_breast",
                     "visiumhd65_breast_8um","visiumhd11_breast_8um","stratamap_breast"])
        concordance(f"{OUT}/shared_genes_L2.txt","L2",
                    ["prime5k_breast","wta_breast","visiumhd65_breast_8um","visiumhd11_breast_8um","stratamap_breast"])
    bundle = dict(datasets={n:D[n] for n in names},
                  shared_L1=sorted(set(open(f"{OUT}/shared_genes_L1.txt").read().split())) if os.path.exists(f"{OUT}/shared_genes_L1.txt") else [],
                  shared_L2_n=len(open(f"{OUT}/shared_genes_L2.txt").read().split()) if os.path.exists(f"{OUT}/shared_genes_L2.txt") else 0)
    json.dump(bundle, open(f"{OUT}/bundle.json","w"), indent=1)
    print(f"[bundle] {len(names)} datasets -> bundle.json")

if __name__ == "__main__":
    main()
