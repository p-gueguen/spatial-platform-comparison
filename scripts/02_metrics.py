#!/usr/bin/env python3
"""
Uniform metric recompute for the spatial platform comparison.
Every comparable number is computed from the count matrix with ONE definition,
so platforms are apples-to-apples (vendor metrics_summary values are NOT).

Phase A: collect Gene Expression gene lists for every available dataset -> shared sets.
Phase B: per dataset, load matrix once, compute:
  - full-panel: n cells, area, median transcripts/cell, median genes/cell, transcripts/mm2
  - specificity: negative-control probe & codeword fraction, genomic-control fraction (imaging)
  - shared-gene (L1 = all 5; L2 = high-plex 4): median tx/cell, median genes/cell,
    mean transcripts per gene per cell  (the true per-gene sensitivity)
  - pseudobulk (per gene total) for cross-platform concordance
  - marker-based major cell type -> per-cell-type shared-gene sensitivity

Outputs (under outputs/):
  shared_genes_L1.txt, shared_genes_L2.txt
  per_dataset/<name>.json          (all scalar metrics)
  pseudobulk/<name>.csv            (gene, count)  GEX only
  celltype/<name>.csv              (celltype, n_cells, median_shared_tx_per_cell, ...)
"""
import os, json, argparse
import numpy as np
import scipy.sparse as sp
import h5py

ROOT = "/srv/GT/analysis/pgueguen/spatial_platform_comparison"
DATA = f"{ROOT}/data"
OUT  = f"{ROOT}/outputs"
for d in ["per_dataset", "pseudobulk", "celltype", "gene_lists"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

# HGNC symbol drift: newer references (Atera WTA, GENCODE) carry current symbols while the std
# Xenium panel and the Visium HD probe set still use legacy ones. Map current -> legacy so a
# symbol intersection does not drop genes that ARE measured under a renamed symbol
# (aminoacyl-tRNA synthetase family +1 renames, and FAM49A -> CYRIA).
ALIAS = {"KARS1":"KARS","LARS1":"LARS","NARS1":"NARS","QARS1":"QARS","WARS1":"WARS","CYRIA":"FAM49A"}
def norm(g): return ALIAS.get(g, g)

# ----------------------------------------------------------------------------- registry
# kind: xenium | hd_bin (8um bins) | hd_seg (segmented cells)
def reg():
    R = [
        dict(name="stdxenium_breast", platform="Xenium 313-plex", tissue="breast",
             modality="imaging", kind="xenium",
             h5=f"{DATA}/stdxenium_breast/cell_feature_matrix.h5",
             metrics=f"{DATA}/stdxenium_breast/metrics_summary.csv"),
        dict(name="prime5k_breast", platform="Xenium Prime 5K", tissue="breast",
             modality="imaging", kind="xenium",
             h5=f"{DATA}/prime5k_breast/cell_feature_matrix.h5",
             metrics=f"{DATA}/prime5k_breast/metrics_summary.csv"),
        dict(name="wta_breast", platform="Xenium WTA (Atera)", tissue="breast",
             modality="imaging", kind="xenium",
             h5=f"{DATA}/wta_breast/cell_feature_matrix.h5",
             metrics=f"{DATA}/wta_breast/metrics_summary.csv"),
        dict(name="prime5k_cervical", platform="Xenium Prime 5K", tissue="cervical",
             modality="imaging", kind="xenium",
             h5=f"{DATA}/prime5k_cervical/cell_feature_matrix.h5",
             metrics=f"{DATA}/prime5k_cervical/metrics_summary.csv"),
        dict(name="wta_cervical", platform="Xenium WTA (Atera)", tissue="cervical",
             modality="imaging", kind="xenium",
             h5=f"{DATA}/wta_cervical/cell_feature_matrix.h5",
             metrics=f"{DATA}/wta_cervical/metrics_summary.csv"),
        dict(name="visiumhd65_breast_8um", platform="Visium HD 6.5mm (8um bin)", tissue="breast",
             modality="sequencing", kind="hd_bin", bin_um=8.0,
             h5=f"{DATA}/visiumhd_65_breast/binned_outputs/square_008um/filtered_feature_bc_matrix.h5",
             metrics=f"{DATA}/visiumhd_65_breast/metrics_summary.csv"),
        dict(name="visiumhd11_breast_8um", platform="Visium HD 11mm (8um bin)", tissue="breast",
             modality="sequencing", kind="hd_bin", bin_um=8.0,
             h5=f"{DATA}/visiumhd_11_breast/binned_outputs/square_008um/filtered_feature_bc_matrix.h5",
             metrics=f"{DATA}/visiumhd_11_breast/metrics_summary.csv"),
        dict(name="visiumhd11_breast_seg", platform="Visium HD 11mm (segmented cell)", tissue="breast",
             modality="sequencing", kind="hd_seg",
             h5=f"{DATA}/visiumhd_11_breast/segmented_outputs/filtered_feature_cell_matrix.h5",
             metrics=f"{DATA}/visiumhd_11_breast/metrics_summary.csv"),
        # Fresh-Frozen 6.5mm breast, probe-based (Human Tx Probe Set v2), SR 4.0.1, native cell segmentation.
        # Same probe chemistry as the FFPE platforms; the confound vs the rest of the benchmark is FF vs FFPE prep.
        dict(name="visiumhd65ff_breast_seg", platform="Visium HD 6.5mm FF probe (segmented cell)", tissue="breast",
             modality="sequencing", kind="hd_seg",
             h5=f"{DATA}/visiumhd65ff_breast_seg/segmented_outputs/filtered_feature_cell_matrix.h5",
             metrics=f"{DATA}/visiumhd65ff_breast_seg/metrics_summary.csv"),
    ]
    return R

# ----------------------------------------------------------------------------- markers
MARKERS = {
    "Epithelial/Tumor": ["EPCAM","KRT8","KRT18","KRT19","KRT7","ELF3","CDH1"],
    "T cell":           ["CD3D","CD3E","CD3G","CD2","TRAC","CD8A","CD4","IL7R"],
    "B/Plasma":         ["MS4A1","CD79A","CD79B","IGHG1","MZB1","JCHAIN"],
    "Myeloid":          ["CD68","LYZ","ITGAX","CD14","C1QA","C1QC","CD163"],
    "Fibroblast":       ["COL1A1","COL1A2","DCN","LUM","PDGFRA","PDGFRB"],
    "Endothelial":      ["PECAM1","VWF","CLDN5","CDH5","EGFL7"],
    "Mast":             ["TPSAB1","CPA3","MS4A2"],
}

# ----------------------------------------------------------------------------- io
def load_10x_h5(path):
    """Return (csc features x cells, names[str], feature_type[str])."""
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        data = g["data"][:]; indices = g["indices"][:]; indptr = g["indptr"][:]
        shape = tuple(int(x) for x in g["shape"][:])
        names = g["features/name"][:].astype(str)
        if "features/feature_type" in g:
            ftype = g["features/feature_type"][:].astype(str)
        else:
            ftype = np.array(["Gene Expression"] * shape[0])
    M = sp.csc_matrix((data, indices, indptr), shape=shape)  # features x cells (CSC by cell)
    return M, names, ftype

def read_metrics_csv(path):
    if not path or not os.path.exists(path):
        return {}
    import csv
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return rows[0] if rows else {}

def gex_genes(ds):
    """Read ONLY the feature names/types (no matrix data) -> uppercase GEX gene list."""
    with h5py.File(ds["h5"], "r") as f:
        names = f["matrix/features/name"][:].astype(str)
        if "matrix/features/feature_type" in f and ds["kind"] == "xenium":
            ftype = f["matrix/features/feature_type"][:].astype(str)
            mask = ftype == "Gene Expression"
        else:
            mask = np.ones(len(names), bool)  # HD probe set: all GEX
    return [norm(n.upper()) for n in names[mask]]

# ----------------------------------------------------------------------------- per dataset
def area_mm2(ds, n_cols):
    if ds["kind"] == "xenium":
        m = read_metrics_csv(ds["metrics"])
        ra = m.get("region_area")
        if ra:
            return float(ra) / 1e6  # um2 -> mm2
        return np.nan
    if ds["kind"] == "hd_bin":
        # tissue area = n bins under tissue * bin_area
        return n_cols * (ds["bin_um"] ** 2) / 1e6
    if ds["kind"] == "hd_seg":
        m = read_metrics_csv(ds["metrics"])
        # use the 11mm region/tissue area from metrics if present, else NaN (report per-cell only)
        for k in ("total_cell_area","region_area"):
            if m.get(k):
                return float(m[k]) / 1e6
        return np.nan

def percell_celltype(Mgex, gene_upper, totals):
    """Marker-argmax major type. Mgex: genes x cells csr-ish; totals: per-cell GEX sum."""
    idx = {g:i for i,g in enumerate(gene_upper)}
    scores = []
    types = []
    safe_tot = np.maximum(totals, 1.0)
    for ct, mk in MARKERS.items():
        rows = [idx[g] for g in mk if g in idx]
        if not rows:
            continue
        sub = Mgex[rows, :]                      # markers x cells (sparse)
        s = np.asarray(sub.sum(0)).ravel().astype(np.float64)
        s = np.log1p(s / safe_tot * 1e4)         # CP10k log marker load, length-normalised
        scores.append(s); types.append(ct)
    if not scores:
        return np.array(["Unassigned"] * len(totals))
    S = np.vstack(scores)                        # ntypes x cells
    # z-score per type across cells so types are comparable, then argmax
    Z = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
    lab = np.array(types)[Z.argmax(0)]
    lab[totals < 5] = "Unassigned"               # too few transcripts to type
    return lab

def process(ds, L1, L2):
    print(f"[proc] {ds['name']} ...", flush=True)
    M, names, ftype = load_10x_h5(ds["h5"])
    up = np.array([norm(n.upper()) for n in names])
    if ds["kind"] == "xenium":
        gex = ftype == "Gene Expression"
        negp = ftype == "Negative Control Probe"
        negc = ftype == "Negative Control Codeword"
        genc = ftype == "Genomic Control"
        blank = ftype == "Blank Codeword"
    else:
        gex = np.ones(len(names), bool)
        negp = negc = genc = blank = np.zeros(len(names), bool)

    Mg = M[gex, :].tocsr()                       # genes x cells
    gup = up[gex]
    ncells = Mg.shape[1]
    tot = np.asarray(Mg.sum(0)).ravel().astype(np.float64)     # transcripts/cell
    ngene = np.asarray((Mg > 0).sum(0)).ravel().astype(np.float64)  # genes/cell
    total_gex = float(tot.sum())
    A = area_mm2(ds, ncells)

    res = dict(
        name=ds["name"], platform=ds["platform"], tissue=ds["tissue"],
        modality=ds["modality"], kind=ds["kind"],
        n_units=int(ncells), unit=("cell" if ds["kind"] != "hd_bin" else "8um_bin"),
        n_gex_genes=int(gex.sum()),
        area_mm2=float(A) if A==A else None,
        total_gex_transcripts=total_gex,
        median_transcripts_per_cell=float(np.median(tot)),
        mean_transcripts_per_cell=float(tot.mean()),
        median_genes_per_cell=float(np.median(ngene)),
        transcripts_per_mm2=(total_gex/A) if A==A else None,
        cells_per_mm2=(ncells/A) if A==A else None,
    )

    # specificity (imaging only)
    if ds["kind"] == "xenium":
        sp_probe = float(M[negp,:].sum()); sp_cw = float(M[negc,:].sum())
        sp_gen = float(M[genc,:].sum()); sp_blank = float(M[blank,:].sum())
        res.update(
            negctrl_probe_counts=sp_probe, negctrl_codeword_counts=sp_cw,
            genomic_control_counts=sp_gen, blank_codeword_counts=sp_blank,
            negctrl_probe_frac_of_gex=sp_probe/total_gex if total_gex else None,
            negctrl_codeword_frac_of_gex=sp_cw/total_gex if total_gex else None,
            n_negctrl_probes=int(negp.sum()), n_negctrl_codewords=int(negc.sum()),
        )
        # 10x-style normalised rate: counts per neg-probe per cell vs counts per gene per cell
        if negp.sum() and gex.sum():
            per_negprobe = sp_probe/negp.sum()/ncells
            per_gene = total_gex/gex.sum()/ncells
            res["negctrl_norm_rate"] = per_negprobe/per_gene if per_gene else None

    # shared-gene blocks
    def shared_block(shared, tag):
        rows = [i for i,g in enumerate(gup) if g in shared]
        if not rows:
            res[f"{tag}_n_genes_present"] = 0
            return None
        sub = Mg[rows, :]
        st = np.asarray(sub.sum(0)).ravel().astype(np.float64)
        sg = np.asarray((sub > 0).sum(0)).ravel().astype(np.float64)
        res[f"{tag}_n_genes_present"] = len(rows)
        res[f"{tag}_median_transcripts_per_cell"] = float(np.median(st))
        res[f"{tag}_mean_transcripts_per_cell"]   = float(st.mean())
        res[f"{tag}_median_genes_per_cell"]       = float(np.median(sg))
        res[f"{tag}_mean_tx_per_gene_per_cell"]   = float(st.sum()/(len(rows)*ncells))
        res[f"{tag}_frac_genes_detected_per_cell"]= float((sg/len(rows)).mean())
        res[f"{tag}_total_transcripts"]           = float(st.sum())
        # area-normalised shared-gene sensitivity (no cell/bin-size confound), per gene
        if A == A and len(rows):
            res[f"{tag}_transcripts_per_mm2"]          = float(st.sum()/A)
            res[f"{tag}_transcripts_per_mm2_per_gene"] = float(st.sum()/A/len(rows))
        return st  # per-cell shared totals (for celltype)

    st_L1 = shared_block(set(L1), "L1")
    shared_block(set(L2), "L2")

    # pseudobulk (GEX) for concordance
    pb = np.asarray(Mg.sum(1)).ravel().astype(np.float64)
    with open(f"{OUT}/pseudobulk/{ds['name']}.csv","w") as fh:
        fh.write("gene,count\n")
        for g,c in zip(gup, pb):
            if c>0: fh.write(f"{g},{int(c)}\n")

    # per-cell-type shared-L1 sensitivity
    try:
        lab = percell_celltype(Mg, list(gup), tot)
        with open(f"{OUT}/celltype/{ds['name']}.csv","w") as fh:
            fh.write("celltype,n_cells,median_tx_per_cell,median_genes_per_cell,median_L1_tx_per_cell\n")
            for ct in list(MARKERS)+["Unassigned"]:
                sel = lab==ct
                if sel.sum()==0: continue
                mL1 = float(np.median(st_L1[sel])) if st_L1 is not None else float("nan")
                fh.write(f"{ct},{int(sel.sum())},{float(np.median(tot[sel])):.3f},"
                         f"{float(np.median(ngene[sel])):.3f},{mL1:.3f}\n")
    except Exception as e:
        print(f"  [celltype warn] {e}", flush=True)

    # spec-sheet values for cross-check
    res["vendor_metrics"] = read_metrics_csv(ds["metrics"])
    with open(f"{OUT}/per_dataset/{ds['name']}.json","w") as fh:
        json.dump(res, fh, indent=2)
    print(f"  done: {ncells} units, area={res['area_mm2']}, med tx/cell={res['median_transcripts_per_cell']}", flush=True)
    return res

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of dataset names")
    ap.add_argument("--phase", choices=["genes","metrics","both"], default="both")
    args = ap.parse_args()

    R = reg()
    avail = [d for d in R if os.path.exists(d["h5"])]
    miss = [d["name"] for d in R if not os.path.exists(d["h5"])]
    if miss: print("[skip - file not present yet]:", miss, flush=True)
    if args.only:
        keep = set(args.only.split(","))
        avail = [d for d in avail if d["name"] in keep]

    # Phase A: gene lists + shared sets
    glists = {}
    for d in avail:
        gl = gex_genes(d)
        glists[d["name"]] = set(gl)
        with open(f"{OUT}/gene_lists/{d['name']}.txt","w") as fh:
            fh.write("\n".join(sorted(gl)))
        print(f"[genes] {d['name']}: {len(gl)} GEX genes", flush=True)

    # L1 = FLAGSHIP CORE: the 313-plex std-panel genes measurable on the whole-transcriptome
    # comparators (Atera WTA + both Visium HD). Prime 5K is a *targeted* 5,101-gene panel that
    # carries only 189/313 of the flagship panel, so it is NOT part of the apples-to-apples core
    # (it stays in the L2 high-plex track). With alias resolution this yields ~304 genes.
    # CosMx WTx (311/313) and StrataMap (313/313) contain all of it and are scored downstream.
    core_wtx  = [n for n in ["stdxenium_breast","wta_breast",
                             "visiumhd65_breast_8um","visiumhd11_breast_8um"] if n in glists]
    highplex  = [n for n in ["prime5k_breast","wta_breast",
                             "visiumhd65_breast_8um","visiumhd11_breast_8um"] if n in glists]
    if core_wtx:
        L1 = set.intersection(*[glists[n] for n in core_wtx])
    else:
        L1 = set.intersection(*glists.values()) if glists else set()
    if highplex:
        L2 = set.intersection(*[glists[n] for n in highplex])
    else:
        L2 = set(L1)
    # only finalise/overwrite shared sets when the full whole-tx core is available
    have_all_breast = len(core_wtx) == 4
    if have_all_breast or not os.path.exists(f"{OUT}/shared_genes_L1.txt"):
        open(f"{OUT}/shared_genes_L1.txt","w").write("\n".join(sorted(L1)))
        open(f"{OUT}/shared_genes_L2.txt","w").write("\n".join(sorted(L2)))
        print(f"[shared] L1={len(L1)} genes (from {core_wtx}); L2={len(L2)} (from {highplex})", flush=True)
    else:
        L1 = set(open(f"{OUT}/shared_genes_L1.txt").read().split())
        L2 = set(open(f"{OUT}/shared_genes_L2.txt").read().split())
        print(f"[shared] reusing cached L1={len(L1)}, L2={len(L2)}", flush=True)

    if args.phase == "genes":
        return
    for d in avail:
        process(d, L1, L2)

if __name__ == "__main__":
    main()
