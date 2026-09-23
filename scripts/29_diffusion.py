#!/usr/bin/env python3
"""
Transcript diffusion / lateral spillover, ONE definition for every platform.

The question this answers: when a gene is expressed only inside a compact domain (epithelial
tumour nests, immune aggregates), how far outside that domain does its signal appear, and is the
leak ISOTROPIC (ordinary diffusion / optical or segmentation blur) or DIRECTIONAL (the systematic
transcript drift that Visium v1 showed, where lateral fluid movement carries transcripts
preferentially along one axis)?

Everything is measured on the SAME 8 um square lattice the report already uses for the
unit-matched sensitivity line (06_bin8um.py), so there is no segmentation anywhere and no
cell-vs-bin confound. Imaging platforms are gridded from raw transcript coordinates, Visium HD
uses its native 8 um bins, StrataMap is gridded from the 1 um SBC matrix.

Estimator, per platform and per source cell type:
  1. Split that type's markers into two disjoint halves, A1 and A2. A1 DEFINES the source domain,
     A2 is the field we MEASURE. The halves are disjoint so the domain definition cannot
     manufacture the signal it is then used to measure.
  2. Both halves are expressed as a fraction of the bin's total counts, which removes the ~100x
     sensitivity spread between platforms from the comparison.
  3. Source domain = bins above the (1 - src_frac) quantile of a 3x3-smoothed A1 fraction,
     morphologically opened and closed so the domain is a region and not speckle.
  4. For every non-source tissue bin: r = Euclidean distance to the nearest source bin (um), and
     its A2 fraction. Profile y(r), in-domain level y_S (mask eroded 1 bin), ambient floor
     c_inf = mean beyond 120 um.
  5. lambda = decay length of p(r) = (y(r) - c_inf) / (y_S - c_inf), least squares on log p.
  6. Anisotropy: the same fit inside each of 8 angular sectors, taken relative to each bin's
     nearest source bin -> lambda_max / lambda_min and the axis of lambda_max.
  7. Drift vector D: excess-weighted mean unit displacement away from the source, over bins within
     40 um. |D| = 0 is isotropic leak, |D| -> 1 is a fully one-sided leak. Significance from 200
     permutations of the weights WITHIN each distance ring, which preserves the radial profile and
     destroys only the angular structure.

Nulls carried alongside every number:
  lambda_null  - the same fit after rolling the source mask half a field away, i.e. the value the
                 estimator returns on a domain that is the right size and shape but in the wrong
                 place. lambda ~ lambda_null means no measurable spillover.
  drift_p      - permutation p for |D|.

--selftest runs the positive controls on synthetic fields: a known isotropic blur must raise
lambda monotonically, a known anisotropic blur must be recovered on the right axis, a known
directional shift must be recovered as a drift vector pointing the right way, and an unperturbed
field must come back with no significant drift. Run it before trusting any number below.

Outputs outputs/diffusion/<name>.json   (plus outputs/diffusion/_selftest.json for --selftest)
"""
import os, sys, json, gzip, argparse, subprocess
import numpy as np

ROOT = "/srv/GT/analysis/pgueguen/spatial_platform_comparison"
OUT  = f"{ROOT}/outputs"
os.makedirs(f"{OUT}/diffusion", exist_ok=True)

BIN_UM   = 8.0
SRC_FRAC = 0.10      # source domain = top 10% of occupied bins by smoothed A1 fraction
R_MAX    = 10        # rings used for the lambda fit: 1..10 bins = 8..80 um
R_FAR    = 15        # ambient floor measured beyond 15 bins = 120 um
R_DRIFT  = 5         # drift vector uses bins within 5 bins = 40 um
N_PERM   = 200

# HGNC symbol drift (must match 02_metrics.py) so newer references match the legacy-symbol L1 core.
ALIAS = {"KARS1":"KARS","LARS1":"LARS","NARS1":"NARS","QARS1":"QARS","WARS1":"WARS","CYRIA":"FAM49A"}

# Marker halves. Every gene here is in shared_genes_L1.txt, so the SAME genes are measurable on
# all six breast platforms - the comparison is not confounded by panel membership.
# A1 defines the domain, A2 is measured. Disjoint by construction, and deliberately 6-8 genes per
# half: a 2-3 gene half is a negligible fraction of an 18k-gene whole-transcriptome library, and on
# Prime 5K that gave an in-domain / ambient contrast of only 1.9 and no usable fit at all.
# Three compartments, because the epithelium/stroma boundary is the sharpest in this tissue and the
# immune domains are the smallest: a platform-level effect has to show up in more than one of them.
SOURCES = {
    "epithelial": dict(A1=["KRT8","ELF3","S100A14","CLDN4","DSP","TACSTD2","KRT7","AGR3"],
                       A2=["EPCAM","CDH1","KRT23","JUP","LYPD3","MLPH","CEACAM6","TPD52"]),
    "immune":     dict(A1=["PTPRC","CD3E","CD68","C1QA","TYROBP","FCER1G","AIF1","LYZ"],
                       A2=["CD3D","TRAC","CD14","CD163","CD247","IL2RG","CYTIP","LY86"]),
    "stromal":    dict(A1=["LUM","POSTN","PDGFRB","FBLN1","PCOLCE","CCDC80"],
                       A2=["DPT","SFRP4","MMP2","PDGFRA","CXCL12","PTGDS"]),
}


# --------------------------------------------------------------------------- estimator
def _fit_lambda(r, p):
    """Least-squares decay length (in bins) of p(r) = exp(-r/lam). Returns (lam, n_used, r2)."""
    ok = np.isfinite(p) & (p > 0.02) & (p < 1.5) & (r >= 1) & (r <= R_MAX)
    if ok.sum() < 4:
        return None, int(ok.sum()), None
    x = r[ok].astype(float); y = np.log(p[ok])
    sl, ic = np.polyfit(x, y, 1)
    if sl >= 0:                     # no decay at all
        return None, int(ok.sum()), None
    yh = sl * x + ic
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - yh) ** 2).sum()) / ss if ss > 0 else None
    return float(-1.0 / sl), int(ok.sum()), r2


def _profile(f2, occ, mask, dist, sel=None):
    """Ring means of f2 over non-source tissue bins, plus in-domain level and ambient floor."""
    from scipy.ndimage import binary_erosion
    core = binary_erosion(mask, np.ones((3, 3), bool)) & occ
    y_s  = float(f2[core].mean()) if core.any() else float(f2[mask & occ].mean())
    tgt  = occ & ~mask
    if sel is not None:
        tgt = tgt & sel
    rb   = np.rint(dist).astype(int)
    far  = tgt & (rb > R_FAR)
    c_inf = float(f2[far].mean()) if far.sum() > 50 else 0.0
    rs, ys, ns = [], [], []
    for r in range(1, R_MAX + 1):
        m = tgt & (rb == r)
        if m.sum() < 30:
            continue
        rs.append(r); ys.append(float(f2[m].mean())); ns.append(int(m.sum()))
    return np.array(rs), np.array(ys), np.array(ns), y_s, c_inf


def spillover(A1, A2, TOT, src_frac=SRC_FRAC, seed=0):
    """Core estimator. A1/A2/TOT are 2D count arrays on a common 8 um lattice."""
    from scipy.ndimage import (uniform_filter, binary_opening, binary_closing,
                               distance_transform_edt)
    rng = np.random.default_rng(seed)
    occ = TOT > 0
    if occ.sum() < 5000:
        return dict(error=f"only {int(occ.sum())} occupied bins")

    den = np.maximum(TOT.astype(np.float64), 1.0)
    f1  = np.where(occ, A1 / den, 0.0)
    f2  = np.where(occ, A2 / den, 0.0)

    # --- source domain from A1 only
    f1s  = uniform_filter(f1, 3)
    thr  = float(np.quantile(f1s[occ], 1.0 - src_frac))
    mask = occ & (f1s >= thr)
    st   = np.ones((3, 3), bool)
    mask = binary_closing(mask, st)
    mask = binary_opening(mask, st) & occ
    if mask.sum() < 200:
        return dict(error=f"source domain too small ({int(mask.sum())} bins)")

    # --- distance and nearest-source direction
    dist, idx = distance_transform_edt(~mask, return_indices=True)
    rs, ys, ns, y_s, c_inf = _profile(f2, occ, mask, dist)
    if len(rs) < 4 or y_s <= c_inf:
        return dict(error="no usable radial profile (in-domain level not above ambient)")
    p = (ys - c_inf) / (y_s - c_inf)
    lam, n_fit, r2 = _fit_lambda(rs, p)

    # --- lambda null: same domain shape, wrong place
    sh = (mask.shape[0] // 2, mask.shape[1] // 2)
    mnull = np.roll(mask, sh, axis=(0, 1)) & occ
    lam_null = None
    if mnull.sum() >= 200:
        dn = distance_transform_edt(~mnull)
        rn, yn, _, ysn, cn = _profile(f2, occ, mnull, dn)
        if len(rn) >= 4 and ysn > cn:
            lam_null, _, _ = _fit_lambda(rn, (yn - cn) / (ysn - cn))

    # --- anisotropy: lambda per 45 deg sector, sector taken from the nearest source bin
    rr, cc = np.indices(mask.shape)
    dy = (rr - idx[0]).astype(np.float64)
    dx = (cc - idx[1]).astype(np.float64)
    ang = np.arctan2(dy, dx)                      # -pi..pi, +x = 0, +y = pi/2 (row index down)
    sec = np.floor((ang + np.pi) / (np.pi / 4)).astype(int) % 8
    lam_sec = {}
    for s in range(8):
        r_s, y_sec, n_s, ys_s, c_s = _profile(f2, occ, mask, dist, sel=(sec == s))
        if len(r_s) >= 4 and ys_s > c_s:
            l_s, _, _ = _fit_lambda(r_s, (y_sec - c_s) / (ys_s - c_s))
        else:
            l_s = None
        lam_sec[s] = l_s
    good = {s: v for s, v in lam_sec.items() if v is not None and v > 0}
    aniso = (max(good.values()) / min(good.values())) if len(good) >= 6 else None
    sec_max = max(good, key=lambda s: good[s]) if len(good) >= 6 else None

    # --- drift vector: excess-weighted mean unit displacement within R_DRIFT.
    # The weighted mean is CENTRED on the UNWEIGHTED mean of the same ring, because the target
    # bins are not angularly symmetric (blobby domains, tissue edges, image borders): with equal
    # weights the raw vector sum is already non-zero, and reading that as drift is exactly the
    # mistake the selftest's isotropic negative control catches. Centring per ring also removes
    # the radial profile, so the permutation null below has expectation zero by construction.
    rb  = np.rint(dist).astype(int)
    tgt = occ & ~mask & (rb >= 1) & (rb <= R_DRIFT)
    w   = np.clip(f2[tgt] - c_inf, 0.0, None)
    L   = np.hypot(dy[tgt], dx[tgt]); L[L == 0] = 1.0
    ux, uy = dx[tgt] / L, dy[tgt] / L
    ring = rb[tgt]
    if w.sum() <= 0:
        return dict(error="no positive excess within the drift radius")
    rings = [(r, ring == r) for r in range(1, R_DRIFT + 1)]
    rings = [(r, m) for r, m in rings if m.sum() >= 30]
    base  = {r: (float(ux[m].mean()), float(uy[m].mean())) for r, m in rings}

    def dvec(weights):
        nx = ny = tw = 0.0
        for r, m in rings:
            wr = weights[m]; sw = float(wr.sum())
            if sw <= 0:
                continue
            bx_, by_ = base[r]
            nx += sw * (float((wr * ux[m]).sum()) / sw - bx_)
            ny += sw * (float((wr * uy[m]).sum()) / sw - by_)
            tw += sw
        return (nx / tw, ny / tw) if tw > 0 else (0.0, 0.0)

    Dx, Dy = dvec(w)
    Dmag = float(np.hypot(Dx, Dy))
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        wp = w.copy()
        for _, m in rings:
            wp[m] = rng.permutation(wp[m])
        nx, ny = dvec(wp)
        null[i] = np.hypot(nx, ny)
    drift_p = float((null >= Dmag).sum() + 1) / (N_PERM + 1)

    # Threshold-free companion to lambda: what SHARE of the compartment's own signal is sitting
    # outside its domain? No exponential fit, no floor subtraction, so it does not inherit
    # lambda's sensitivity to the shape of the tail - it is the number a user actually feels
    # ("how much of my epithelial signal is in the stroma?").
    a2_tot = float(A2[occ].sum())
    rb0 = np.rint(dist).astype(int)
    off = {}
    for lim in (2, 5):
        m = occ & ~mask & (rb0 > lim)
        off[f"offdomain_signal_frac_gt{int(lim * BIN_UM)}um"] = (
            float(A2[m].sum() / a2_tot) if a2_tot > 0 else None)

    return dict(
        n_occupied_bins=int(occ.sum()), n_source_bins=int(mask.sum()),
        src_frac=src_frac, bin_um=BIN_UM, **off,
        in_domain_frac=y_s, ambient_frac=c_inf,
        contrast=float(y_s / c_inf) if c_inf > 0 else None,
        lambda_um=(lam * BIN_UM) if lam else None, lambda_fit_n=n_fit, lambda_fit_r2=r2,
        lambda_null_um=(lam_null * BIN_UM) if lam_null else None,
        profile_r_um=[float(x * BIN_UM) for x in rs],
        profile_p=[float(x) for x in p], profile_n=[int(x) for x in ns],
        aniso_ratio=aniso,
        aniso_axis_deg=(float((sec_max * 45.0) - 180.0 + 22.5) if sec_max is not None else None),
        lambda_sector_um={str(s): (v * BIN_UM if v else None) for s, v in lam_sec.items()},
        drift_x=Dx, drift_y=Dy, drift_mag=Dmag,
        drift_deg=float(np.degrees(np.arctan2(Dy, Dx))),
        drift_null_mean=float(null.mean()), drift_null_p95=float(np.quantile(null, 0.95)),
        drift_p=drift_p,
    )


# --------------------------------------------------------------------------- field builders
def _flags(genes_upper, want):
    """want: dict tag -> list of symbols. Returns tag -> set(upper symbols)."""
    return {t: set(g.upper() for g in gl) for t, gl in want.items()}


def fields_xenium(path, want, qv_min=20.0):
    """Grid raw transcripts to 8 um. Returns dict tag -> 2D array, plus TOT and off-cell stats."""
    import polars as pl
    sets = _flags(None, want)
    allg = set().union(*sets.values())
    lf = pl.scan_parquet(path)
    cols = lf.collect_schema().names()
    dt = dict(zip(lf.collect_schema().names(), lf.collect_schema().dtypes()))

    g = pl.col("feature_name")
    if dt["feature_name"] == pl.Binary:
        g = g.cast(pl.Utf8)
    g = g.str.to_uppercase().replace(ALIAS)

    flt = pl.col("qv") >= qv_min
    if "is_gene" in cols:
        flt = flt & pl.col("is_gene")
    else:
        # 2022 outs have no is_gene: drop control codewords by name
        flt = flt & (~g.str.contains(r"^(NEGCONTROL|BLANK|ANTISENSE|UNASSIGNED|DEPRECATED|INTERGENIC)"))

    base = (lf.filter(flt)
              .with_columns([(pl.col("x_location") / BIN_UM).floor().cast(pl.Int32).alias("bx"),
                             (pl.col("y_location") / BIN_UM).floor().cast(pl.Int32).alias("by"),
                             g.alias("g")]))
    aggs = [pl.len().alias("tot")]
    for tag, S in sets.items():
        aggs.append(pl.col("g").is_in(list(S)).sum().alias(tag))
    df = base.group_by(["bx", "by"]).agg(aggs).collect(engine="streaming")

    # off-cell fraction: same filter, cell_id sentinel differs by pipeline version
    cid = pl.col("cell_id")
    if dt["cell_id"] in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
        offc = cid.cast(pl.Int64) <= 0
    else:
        offc = cid.cast(pl.Utf8).is_in(["UNASSIGNED", "-1", "0", ""])
    oc = (lf.filter(flt).select([offc.sum().alias("off"), pl.len().alias("n")])
            .collect(engine="streaming"))
    off_frac = float(oc["off"][0]) / float(oc["n"][0]) if oc["n"][0] else None

    return _to_arrays(df, list(sets.keys())), dict(off_cell_frac=off_frac,
                                                   n_transcripts=int(oc["n"][0]))


def _to_arrays(df, tags):
    bx = df["bx"].to_numpy(); by = df["by"].to_numpy()
    bx = bx - bx.min(); by = by - by.min()
    H, W = int(by.max()) + 1, int(bx.max()) + 1
    out = {}
    for name in ["tot"] + tags:
        a = np.zeros((H, W), np.float64)
        a[by, bx] = df[name].to_numpy()
        out[name] = a
    return out


def fields_hd(base, want):
    """Visium HD native 8 um bins -> arrays on the array_row/array_col lattice."""
    import polars as pl, h5py, scipy.sparse as sp
    sets = _flags(None, want)
    h5 = f"{base}/filtered_feature_bc_matrix.h5"
    with h5py.File(h5, "r") as f:
        gm = f["matrix"]
        names = np.array([ALIAS.get(x, x) for x in
                          np.char.upper(gm["features/name"][:].astype(str))])
        # 10x h5 is CSC: indptr runs over BARCODES, indices over features (same as 02_metrics.py)
        M = sp.csc_matrix((gm["data"][:], gm["indices"][:], gm["indptr"][:]),
                          shape=tuple(int(x) for x in gm["shape"][:]))   # features x bins
        bcs = gm["barcodes"][:].astype(str)
    tot = np.asarray(M.sum(0)).ravel()
    sums = {}
    for tag, S in sets.items():
        rows = np.where(np.isin(names, list(S)))[0]
        sums[tag] = np.asarray(M[rows, :].sum(0)).ravel() if len(rows) else np.zeros(M.shape[1])

    tp = pl.read_parquet(f"{base}/spatial/tissue_positions.parquet")
    pos = dict(zip(tp["barcode"].to_list(), zip(tp["array_row"].to_list(), tp["array_col"].to_list())))
    rc = np.array([pos.get(b, (-1, -1)) for b in bcs])
    keep = rc[:, 0] >= 0
    r0, c0 = rc[keep, 0], rc[keep, 1]
    r0 = r0 - r0.min(); c0 = c0 - c0.min()
    H, W = int(r0.max()) + 1, int(c0.max()) + 1
    out = {}
    for name, v in [("tot", tot)] + list(sums.items()):
        a = np.zeros((H, W), np.float64); a[r0, c0] = v[keep]; out[name] = a
    return out, dict(off_cell_frac=None, n_transcripts=float(tot.sum()))


def fields_stratamap(want):
    """StrataMap: 1 um SBC matrix -> 8 um bins. One streaming pass, marker rows only for the sets."""
    import polars as pl
    sets = _flags(None, want)
    W = f"{ROOT}/data/stratamap_breast/grade3"; RM = f"{W}/Raw_Matrix_Files"
    sym = [""]
    with gzip.open(f"{RM}/features.tsv.gz", "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            s = (p[1] if len(p) > 1 else p[0]).upper()
            sym.append(ALIAS.get(s, s))
    ng = len(sym) - 1
    flag = {t: np.zeros(ng + 1, bool) for t in sets}
    for i in range(1, ng + 1):
        for t, S in sets.items():
            if sym[i] in S:
                flag[t][i] = True
    for t in sets:
        print(f"    [{t}] {int(flag[t].sum())} matrix rows", flush=True)

    bc = pl.read_csv(f"{RM}/barcodes.tsv.gz", has_header=False, new_columns=["sbc"])
    pr = bc["sbc"].str.split(":")
    by = np.floor(pr.list.get(1).cast(pl.Float64).to_numpy() / 1000.0 / BIN_UM).astype(np.int64)
    bx = np.floor(pr.list.get(2).cast(pl.Float64).to_numpy() / 1000.0 / BIN_UM).astype(np.int64)
    by -= by.min(); bx -= bx.min()
    H, Wd = int(by.max()) + 1, int(bx.max()) + 1
    lin = by * Wd + bx
    del by, bx

    acc = {k: np.zeros(H * Wd, np.float64) for k in ["tot"] + list(sets.keys())}
    proc = subprocess.Popen(["bash", "-c", f"pigz -dc {RM}/matrix.mtx.gz"], stdout=subprocess.PIPE)
    reader = pl.read_csv_batched(proc.stdout, separator=" ", has_header=False, skip_rows=3,
                                 new_columns=["g", "s", "c"], batch_size=80_000_000)
    nb = 0
    while True:
        b = reader.next_batches(1)
        if not b:
            break
        df = b[0]; nb += 1
        g = df["g"].to_numpy(); s = df["s"].to_numpy(); c = df["c"].to_numpy().astype(np.float64)
        bn = lin[s - 1]
        acc["tot"] += np.bincount(bn, weights=c, minlength=H * Wd)
        for t in sets:
            m = flag[t][g]
            if m.any():
                acc[t] += np.bincount(bn[m], weights=c[m], minlength=H * Wd)
        print(f"    batch {nb}", flush=True)
    return ({k: v.reshape(H, Wd) for k, v in acc.items()},
            dict(off_cell_frac=1.0 - 0.7507569083435881,   # 1 - percell_sbc_assigned_frac (17_)
                 off_cell_note="fraction of SBC signal outside any segmented cell, from 17_stratamap_percell_fast",
                 n_transcripts=float(acc["tot"].sum())))


# --------------------------------------------------------------------------- registry
D = f"{ROOT}/data"
DS = {
    "stdxenium_breast":      ("xenium", f"{D}/stdxenium_breast/transcripts.parquet"),
    "prime5k_breast":        ("xenium", f"{D}/prime5k_breast/transcripts.parquet"),
    "wta_breast":            ("xenium", f"{D}/atera_breast_tx/transcripts.parquet"),
    "visiumhd65_breast_8um": ("hd",     f"{D}/visiumhd_65_breast/binned_outputs/square_008um"),
    "visiumhd11_breast_8um": ("hd",     f"{D}/visiumhd_11_breast/binned_outputs/square_008um"),
    "stratamap_breast":      ("strata", None),
}


def run_one(name, recache=False):
    kind, path = DS[name]
    want = {}
    for src, hv in SOURCES.items():
        want[f"{src}_A1"] = hv["A1"]
        want[f"{src}_A2"] = hv["A2"]
    # The field pass is the only expensive step (a full streaming read of a 1.6-10 GB transcript
    # table). Cache it so the estimator can be re-run and re-checked for free.
    # Arrays only, no pickle: the small side-car metadata goes to its own JSON file.
    cache, cmeta = f"{OUT}/diffusion/_fields_{name}.npz", f"{OUT}/diffusion/_fields_{name}.json"
    if os.path.exists(cache) and os.path.exists(cmeta) and not recache:
        z = np.load(cache)
        arr = {k: z[k] for k in z.files}
        extra = json.load(open(cmeta))
        print(f"[diffusion] {name} ({kind}) fields from cache", flush=True)
    else:
        print(f"[diffusion] {name} ({kind}) building 8 um fields ...", flush=True)
        if kind == "xenium":
            arr, extra = fields_xenium(path, want)
        elif kind == "hd":
            arr, extra = fields_hd(path, want)
        else:
            arr, extra = fields_stratamap(want)
        np.savez_compressed(cache, **arr)
        json.dump(extra, open(cmeta, "w"), indent=2)

    cal = _load_calibration()
    if cal is None:
        print("  [warn] no drift calibration on disk; run --selftest first so drift can be "
              "reported in um instead of an uninterpretable |D|", flush=True)
    res = dict(name=name, kind=kind, bin_um=BIN_UM, **extra, sources={})
    for src in SOURCES:
        a1, a2 = arr[f"{src}_A1"], arr[f"{src}_A2"]
        r = spillover(a1, a2, arr["tot"])
        r["A1_genes"] = list(SOURCES[src]["A1"]); r["A2_genes"] = list(SOURCES[src]["A2"])
        r["A1_total"] = float(a1.sum()); r["A2_total"] = float(a2.sum())
        if cal and "error" not in r:
            r["drift_equiv_um"] = _drift_equiv_um(r["drift_mag"], cal[0], cal[1])
            r["drift_resolved"] = bool(r["drift_equiv_um"] and r["drift_equiv_um"] >= BIN_UM)
        # Swap control: if lambda is a property of the platform's spatial blur it must survive
        # exchanging which marker half defines the domain and which is measured. If it does not,
        # the number is a property of the two gene lists, not of the platform.
        rsw = spillover(a2, a1, arr["tot"])
        r["swap_lambda_um"] = rsw.get("lambda_um")
        r["swap_aniso_ratio"] = rsw.get("aniso_ratio")
        if cal and "error" not in rsw:
            r["swap_drift_equiv_um"] = _drift_equiv_um(rsw["drift_mag"], cal[0], cal[1])
        if r.get("lambda_um") and r.get("swap_lambda_um"):
            r["swap_lambda_ratio"] = float(r["swap_lambda_um"] / r["lambda_um"])
        res["sources"][src] = r
        if "error" in r:
            print(f"    {src}: SKIP {r['error']}", flush=True)
        else:
            print(f"    {src}: lambda={r['lambda_um']} um (displaced-domain null "
                  f"{r['lambda_null_um']}, A1/A2 swapped {r.get('swap_lambda_um')}), "
                  f"aniso={r['aniso_ratio']}, "
                  f"drift |D|={r['drift_mag']:.4f} = {r.get('drift_equiv_um')} um equiv "
                  f"at {r['drift_deg']:.0f} deg, "
                  f"ambient={r['ambient_frac']:.4g} vs in-domain {r['in_domain_frac']:.4g}",
                  flush=True)
    json.dump(res, open(f"{OUT}/diffusion/{name}.json", "w"), indent=2)
    return res


# --------------------------------------------------------------------------- drift calibration
def _drift_equiv_um(dmag, offs, mags):
    """Map an observed |D| onto the one-sided offset (um) that reproduces it, from the synthetic
    calibration curve built by --selftest. Monotone interpolation, clipped at the ends."""
    if dmag is None or not mags or any(m is None for m in mags):
        return None
    return float(np.interp(dmag, mags, offs))


def _load_calibration():
    p = f"{OUT}/diffusion/_drift_calibration.json"
    if not os.path.exists(p):
        return None
    c = json.load(open(p))
    return c["offset_um"], c["drift_mag"]


# --------------------------------------------------------------------------- positive controls
def _synthetic(H=320, W=320, seed=1):
    """Blobby source domain + a matched total-counts field. A2_true lives only inside the domain."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((H, W))
    src = np.zeros((H, W), bool)
    for _ in range(28):
        cy, cx, rad = rng.integers(20, H - 20), rng.integers(20, W - 20), rng.integers(6, 16)
        src |= (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
    tot = np.full((H, W), 200.0)
    A1 = np.where(src, 40.0, 0.4)      # domain-defining half, small ambient
    A2 = np.where(src, 30.0, 0.0)      # measured half, initially ZERO outside
    return A1, A2, tot, src


def selftest():
    from scipy.ndimage import gaussian_filter, shift as ndshift
    A1, A2, tot, src = _synthetic()
    log = {}
    ok = True

    def rep(tag, r):
        log[tag] = {k: r.get(k) for k in
                    ("lambda_um", "lambda_null_um", "aniso_ratio", "aniso_axis_deg",
                     "drift_mag", "drift_deg", "drift_p", "ambient_frac", "in_domain_frac")}
        print(f"  {tag:26s} lambda={r.get('lambda_um')} aniso={r.get('aniso_ratio')} "
              f"axis={r.get('aniso_axis_deg')} |D|={r.get('drift_mag')} "
              f"deg={r.get('drift_deg')} p={r.get('drift_p')}", flush=True)
        return r

    print("[selftest] 1. resolution floor: no spillover and a sub-bin blur must NOT be given a "
          "lambda; a blur wider than the bin must be, and must grow with sigma", flush=True)
    lams = {}
    for s in (0.0, 1.0, 2.0, 4.0):
        A2b = gaussian_filter(A2, s) if s > 0 else A2.copy()
        A2b = A2b + 0.05                       # flat ambient so c_inf is estimable
        r = rep(f"iso_sigma_{s:g}bins", spillover(A1, A2b, tot))
        lams[s] = r.get("lambda_um")
    if lams[0.0] is not None:
        print(f"  FAIL: a field with ZERO spillover was given lambda={lams[0.0]}"); ok = False
    elif lams[2.0] is None or lams[4.0] is None:
        print(f"  FAIL: resolvable blurs gave no lambda: {lams}"); ok = False
    elif not lams[2.0] < lams[4.0]:
        print(f"  FAIL: lambda not monotone in sigma: {lams}"); ok = False
    else:
        print(f"  PASS: sigma 0 um -> unresolved, 8 um -> {lams[1.0]}, "
              f"16 um -> {lams[2.0]:.1f} um, 32 um -> {lams[4.0]:.1f} um. "
              f"An 8 um lattice cannot resolve a decay shorter than one bin; that is a floor, "
              f"not a zero.")

    print("[selftest] 2. anisotropy must be ~1 when isotropic and must GROW with the injected "
          "axis ratio, on the right axis", flush=True)
    ars = {}
    for sy, sx, tag in ((3.0, 3.0, "iso"), (1.5, 4.0, "mild_x"), (0.5, 5.0, "strong_x")):
        r = rep(f"aniso_{tag}", spillover(A1, gaussian_filter(A2, (sy, sx)) + 0.05, tot))
        ars[tag] = (r.get("aniso_ratio"), r.get("aniso_axis_deg"))
    if any(v[0] is None for v in ars.values()):
        print(f"  FAIL: anisotropy not returned for every case: {ars}"); ok = False
    elif not (ars["iso"][0] < 1.3 <= min(ars["mild_x"][0], ars["strong_x"][0])):
        print(f"  FAIL: isotropic and anisotropic cases not separated: {ars}"); ok = False
    elif not all(abs(ars[t][1]) < 45 or abs(abs(ars[t][1]) - 180) < 45 for t in ("mild_x", "strong_x")):
        print(f"  FAIL: recovered axis is not the injected x axis: {ars}"); ok = False
    else:
        print(f"  PASS: ratio {ars['iso'][0]:.2f} isotropic vs {ars['mild_x'][0]:.2f} (2.7:1 "
              f"injected) and {ars['strong_x'][0]:.2f} (10:1 injected), axis "
              f"{ars['strong_x'][1]:.0f} deg. The ratio SATURATES once the short axis falls "
              f"below one bin, so it separates isotropic from directional but does not "
              f"quantify how directional.")

    print("[selftest] 3. CALIBRATION: |D| against a known one-sided offset. |D| on its own is "
          "not interpretable and its permutation p saturates at 10^4-10^5 bins, so the reported "
          "number is the offset in um that reproduces the observed |D|.", flush=True)
    cal = []
    for sx in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
        A2s = ndshift(gaussian_filter(A2, 1.5), (0.0, sx), order=1) + 0.05
        r = rep(f"shift_{sx*BIN_UM:g}um", spillover(A1, A2s, tot))
        cal.append((sx * BIN_UM, r.get("drift_mag"), r.get("drift_deg")))
    log["calibration_um_to_Dmag"] = [[u, d] for u, d, _ in cal]
    mags = [d for _, d, _ in cal]
    degs = [g for _, _, g in cal]
    if any(m is None for m in mags):
        print("  FAIL: calibration incomplete"); ok = False
    elif not all(mags[i] < mags[i + 1] for i in range(len(mags) - 1)):
        print(f"  FAIL: |D| not monotone in the injected offset: {list(zip([c[0] for c in cal], mags))}")
        ok = False
    elif not all(abs(g) < 45 for g in degs[1:]):
        print(f"  FAIL: direction wrong somewhere: {degs}"); ok = False
    else:
        print("  PASS: " + ", ".join(f"{u:g}um->{m:.3f}" for u, m, _ in cal) +
              f"; all directions within 45 deg of +x")
        json.dump({"offset_um": [c[0] for c in cal], "drift_mag": mags},
                  open(f"{OUT}/diffusion/_drift_calibration.json", "w"), indent=2)

    print("[selftest] 4. negative control: a purely isotropic blur must calibrate BELOW one bin",
          flush=True)
    r = rep("negative_isotropic", spillover(A1, gaussian_filter(A2, 2.0) + 0.05, tot))
    eq = _drift_equiv_um(r.get("drift_mag"), [c[0] for c in cal], mags) if all(
        m is not None for m in mags) else None
    if eq is None:
        print("  FAIL: could not calibrate the negative control"); ok = False
    elif eq >= BIN_UM:
        print(f"  FAIL: isotropic blur read back as a {eq:.1f} um offset, i.e. above the "
              f"{BIN_UM:g} um detection floor"); ok = False
    else:
        print(f"  PASS: |D|={r['drift_mag']:.4f} -> equivalent offset {eq:.1f} um, below the "
              f"{BIN_UM:g} um floor. (Its permutation p is {r['drift_p']:.3g}, which is why the "
              f"p-value is not the criterion.)")

    print("[selftest] 5. the displaced-domain null must not reproduce a real decay", flush=True)
    r = log.get("iso_sigma_4bins", {})
    lr, ln = r.get("lambda_um"), r.get("lambda_null_um")
    if lr is None:
        print("  FAIL: reference case has no lambda to compare against"); ok = False
    elif ln is not None and ln >= 0.5 * lr:
        print(f"  FAIL: rolling the domain to the wrong place still gave lambda={ln} "
              f"vs real {lr}: the estimator is fitting the field, not the domain"); ok = False
    else:
        print(f"  PASS: real {lr:.1f} um vs displaced-domain null {ln}")

    log["_all_passed"] = ok
    json.dump(log, open(f"{OUT}/diffusion/_selftest.json", "w"), indent=2)
    print(f"\n[selftest] {'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated dataset names")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--recache", action="store_true", help="rebuild the 8 um fields from source")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    for n in DS:
        if a.only and n not in a.only.split(","):
            continue
        run_one(n, recache=a.recache)
