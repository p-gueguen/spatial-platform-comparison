#!/usr/bin/env python3
"""
REAL-DATA positive control for the diffusion section, using Visium v1.

29_diffusion.py and 31_grid_registration.py are calibrated against offsets and blurs injected
into synthetic fields. That proves the estimators are not blind to an artifact of a known size,
but it does not prove they are sensitive to the artifact as it actually occurs in tissue. For
that you need a platform that demonstrably HAS the problem. Visium v1 is that platform: lateral
diffusion during permeabilisation was its defining weakness, and the section's own capture area
gives a direct handle on it.

The handle: a Visium capture area has spots BEYOND the tissue. Space Ranger flags them
(`in_tissue = 0`). Any signal recovered from a spot that contains no tissue got there by moving.
So for every off-tissue spot we take its distance to the nearest in-tissue spot and its counts
relative to the in-tissue level, and fit

    y(r) = c_inf + a * exp(-r / lambda_out)

The two terms separate the two competing explanations, which is the whole point of fitting a
floor rather than a plain exponential:

  c_inf        ambient / free-floating RNA in the reaction well. NON-spatial: it does not care
               how far a spot is from the tissue, so it is flat in r.
  a, lambda_out  lateral transport out of the tissue. Spatial: it must decay with distance.

Reporting only "% of counts off-tissue" would conflate the two, which is why that number alone is
not the finding here. Note that lambda itself is DESCRIPTIVE: a and c are not separately
identifiable once lambda is large, so neither is used as a test statistic. What is tested is
(i) a model-free Spearman correlation between distance and counts, and (ii) the one-sidedness |D|,
each against a 200-shuffle permutation null; and what is COMPARED between platforms is the
effect size, because with 1.5 M off-tissue bins on the 11 mm section every p-value is small.

Visium HD is run through the identical code as the modern comparator: same vendor lineage, same
in_tissue flag, same estimator, same rings in um. If v1 shows a big resolvable lambda_out and HD
does not, the machinery is proven sensitive to real diffusion, and the negative drift result for
the six modern platforms means something. If they look the same, the machinery is blind and the
report has to say so.

Controls carried:
  permutation null  200 shuffles of the counts among off-tissue spots, which preserves the count
                    distribution and destroys the distance and direction relationships, so both
                    the Spearman correlation and |D| must collapse. Two earlier versions of this
                    check were wrong and are recorded here so they are not reintroduced: a single
                    shuffle (cannot separate noise from noise - it "resolved" a shuffled section),
                    and a "beats a flat model" adjusted-R2 test that was arithmetically always
                    zero for the flat model, i.e. always true.
  two sections      v1 block A sections 1 and 2 are independent sections of the same block; a real
                    chemistry effect has to appear in both.
  footprint caveat  a v1 spot is 55 um across on a 100 um pitch, so a spot 50 um outside the
                    detected boundary can physically overlap tissue the mask missed, inflating
                    v1's first-ring level. That effect is ISOTROPIC and therefore cannot explain
                    the directional asymmetry |D| - which is why |D| is the footprint-immune half
                    of this control and the half the verdict leans on.

Outputs outputs/diffusion/_v1_control.json
"""
import os, json, csv, argparse
import numpy as np

ROOT = "/srv/GT/analysis/pgueguen/spatial_platform_comparison"
OUT  = f"{ROOT}/outputs"; D = f"{ROOT}/data"

RING_UM  = 100.0    # one Visium v1 spot pitch; used for BOTH platforms so lambda is comparable
R_MAX_UM = 1000.0
MIN_PER_RING = 20
V1_SPOT_UM = 55.0   # Visium v1 spot diameter, used to convert spot_diameter_fullres -> um/pixel

DATASETS = {
    "visiumv1_breast_s1": dict(kind="v1",  path=f"{D}/visiumv1_breast_s1"),
    "visiumv1_breast_s2": dict(kind="v1",  path=f"{D}/visiumv1_breast_s2"),
    "visiumhd65_breast_8um": dict(kind="hd",
        path=f"{D}/visiumhd_65_breast/binned_outputs/square_008um"),
    "visiumhd11_breast_8um": dict(kind="hd",
        path=f"{D}/visiumhd_11_breast/binned_outputs/square_008um"),
}


# --------------------------------------------------------------------------- loaders
def _h5_totals(path):
    """Per-barcode total counts from a 10x CSC h5, without densifying anything."""
    import h5py, scipy.sparse as sp
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        M = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                          shape=tuple(int(x) for x in g["shape"][:]))
        bcs = g["barcodes"][:].astype(str)
    return bcs, np.asarray(M.sum(0)).ravel()


def load_v1(base):
    """Visium v1: hex lattice, um/px derived from spot_diameter_fullres (v1 has no
    microns_per_pixel field in scalefactors_json)."""
    sf = json.load(open(f"{base}/spatial/scalefactors_json.json"))
    um_per_px = V1_SPOT_UM / float(sf["spot_diameter_fullres"])
    pos = {}
    with open(f"{base}/spatial/tissue_positions_list.csv") as fh:
        for r in csv.reader(fh):
            if len(r) < 6 or not r[1].lstrip("-").isdigit():
                continue                       # v1 files carry no header, but be tolerant
            pos[r[0]] = (int(r[1]), float(r[4]), float(r[5]))
    bcs, tot = _h5_totals(f"{base}/raw_feature_bc_matrix.h5")
    keep = np.array([b in pos for b in bcs])
    it = np.array([pos[b][0] for b in bcs[keep]], bool)
    y  = np.array([pos[b][1] for b in bcs[keep]]) * um_per_px
    x  = np.array([pos[b][2] for b in bcs[keep]]) * um_per_px
    return x, y, tot[keep].astype(float), it, dict(um_per_px=um_per_px,
                                                   unit="55 um spot, 100 um pitch")


def load_hd(base):
    """Visium HD 8 um bins. Needs the RAW matrix: the filtered one drops off-tissue bins, which
    are precisely the population this measurement is about."""
    import polars as pl
    raw = f"{base}/raw_feature_bc_matrix.h5"
    if not os.path.exists(raw):
        return None
    tp = pl.read_parquet(f"{base}/spatial/tissue_positions.parquet")
    sf = json.load(open(f"{base}/spatial/scalefactors_json.json"))
    umpp = float(sf["microns_per_pixel"])
    pos = {b: (i, pr, pc) for b, i, pr, pc in zip(
        tp["barcode"].to_list(), tp["in_tissue"].to_list(),
        tp["pxl_row_in_fullres"].to_list(), tp["pxl_col_in_fullres"].to_list())}
    bcs, tot = _h5_totals(raw)
    keep = np.array([b in pos for b in bcs])
    it = np.array([pos[b][0] for b in bcs[keep]], bool)
    y  = np.array([pos[b][1] for b in bcs[keep]]) * umpp
    x  = np.array([pos[b][2] for b in bcs[keep]]) * umpp
    return x, y, tot[keep].astype(float), it, dict(um_per_px=umpp, unit="8 um bin")


# --------------------------------------------------------------------------- estimator
def _fit_exp_floor(r, y):
    """y(r) = c + a*exp(-r/lam). Returns (c, a, lam, adjR2) or all-None.

    NOTE: there is deliberately no "flat model adjusted R2" here. An earlier version compared
    against `adj(ss_tot, 1)`, which is identically 0 for every input - so "the exponential beats
    a flat line" reduced to "adjR2 > 0.05", which fitting noise passes. Whether the leak really
    depends on distance is decided by the model-free Spearman test below, not by R2 and not by
    this fit: `a` and `c` trade off against each other at large lambda, so shuffled data can fit
    a LARGER amplitude than the real profile (measured: null p99 476% vs observed 54%).
    """
    from scipy.optimize import curve_fit
    if len(r) < 4:
        return (None,) * 4
    f = lambda rr, c, a, lam: c + a * np.exp(-rr / lam)
    p0 = [float(y[-1]), float(max(y[0] - y[-1], 1e-9)), 150.0]
    try:
        p, _ = curve_fit(f, r, y, p0=p0, maxfev=20000,
                         bounds=([0, 0, 10.0], [np.inf, np.inf, 5000.0]))
    except Exception:
        return (None,) * 4
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return (None,) * 4
    n = len(r)
    r2 = 1.0 - (float(((y - f(r, *p)) ** 2).sum()) / max(n - 3, 1)) / (ss_tot / max(n - 1, 1))
    return float(p[0]), float(p[1]), float(p[2]), float(r2)


def offtissue(x, y, counts, in_tissue, seed=0, ring=RING_UM, rmax=R_MAX_UM, n_perm=200):
    """Signal in spots/bins that contain NO tissue, as a function of distance from the tissue."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    if in_tissue.sum() < 100 or (~in_tissue).sum() < 50:
        return dict(error=f"{int(in_tissue.sum())} in-tissue / "
                          f"{int((~in_tissue).sum())} off-tissue: not enough of one class")
    tin = np.c_[x[in_tissue], y[in_tissue]]
    tout = np.c_[x[~in_tissue], y[~in_tissue]]
    cin, cout = counts[in_tissue], counts[~in_tissue]
    ref = float(np.median(cin))
    if ref <= 0:
        return dict(error="median in-tissue count is zero")

    tree = cKDTree(tin)
    dist, idx = tree.query(tout, k=1)

    edges = np.arange(0, rmax + ring, ring)
    rs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() < MIN_PER_RING:
            continue
        rs.append(float((lo + hi) / 2)); ys.append(float(cout[m].mean() / ref))
        ns.append(int(m.sum()))
    rs, ys = np.array(rs), np.array(ys)
    c, a, lam, r2 = _fit_exp_floor(rs, ys)

    # ---- geometry shared by the drift statistic and its null
    near = np.asarray(tin)[idx]
    dvx, dvy = tout[:, 0] - near[:, 0], tout[:, 1] - near[:, 1]
    L = np.hypot(dvx, dvy); L[L == 0] = 1.0
    ux, uy = dvx / L, dvy / L
    ring_id = np.floor(dist / ring).astype(int)
    rings = [(rr, ring_id == rr) for rr in np.unique(ring_id)]
    rings = [(rr, m) for rr, m in rings if m.sum() >= MIN_PER_RING]
    base = {rr: (float(ux[m].mean()), float(uy[m].mean())) for rr, m in rings}
    floor = c if c is not None else 0.0

    def drift_of(cv):
        w = np.clip(cv / ref - floor, 0.0, None)
        nx = ny = tw = 0.0
        for rr, m in rings:
            sw = float(w[m].sum())
            if sw <= 0:
                continue
            bx, by = base[rr]
            nx += sw * (float((w[m] * ux[m]).sum()) / sw - bx)
            ny += sw * (float((w[m] * uy[m]).sum()) / sw - by)
            tw += sw
        return (nx / tw, ny / tw) if tw > 0 else (0.0, 0.0)

    def profile_of(cv):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (dist >= lo) & (dist < hi)
            if m.sum() < MIN_PER_RING:
                continue
            out.append(float(cv[m].mean() / ref))
        return np.array(out)

    Dx, Dy = drift_of(cout)
    Dmag = float(np.hypot(Dx, Dy))

    # ---- permutation nulls, N shuffles not one. Shuffling counts among off-tissue spots keeps
    # the count distribution and destroys the distance and direction relationships, so both the
    # decay AMPLITUDE and the drift magnitude must collapse. One shuffle cannot distinguish
    # noise from noise, which is what an earlier version of this script tried to do.
    # The test statistic for "does the leak decay with distance" is a model-free Spearman
    # correlation between an off-tissue spot's distance and its counts, NOT the fitted amplitude.
    # The amplitude `a` trades off against the floor `c` whenever lambda is large, so it is not
    # separately identifiable: shuffled profiles fit LARGER amplitudes than the real data
    # (measured null p99 of 476% against an observed 54%), which made it useless as a statistic.
    from scipy.stats import spearmanr
    sp_r = float(spearmanr(dist, cout).statistic)

    d_null, sp_null = [], []
    for _ in range(n_perm):
        cs = rng.permutation(cout)
        sp_null.append(float(spearmanr(dist, cs).statistic))
        nx, ny = drift_of(cs)
        d_null.append(float(np.hypot(nx, ny)))
    d_null = np.array(d_null); sp_null = np.array(sp_null)
    # one-sided: a decay means counts FALL with distance, i.e. a negative correlation
    sp_p = float((sp_null <= sp_r).sum() + 1) / (len(sp_null) + 1)
    drift_p = float((d_null >= Dmag).sum() + 1) / (len(d_null) + 1) if len(d_null) else None

    # Model-free effect sizes. These, not the p-values, are what the v1-vs-HD comparison rests on:
    # with 1.5 M off-tissue bins on the 11 mm section every p-value is small regardless.
    near_far = float(ys[0] / ys[-1]) if (len(ys) > 1 and ys[-1] > 0) else None
    resolved = bool(sp_r < 0 and sp_p < 0.01)

    return dict(
        n_in_tissue=int(in_tissue.sum()), n_off_tissue=int((~in_tissue).sum()),
        median_in_tissue_counts=ref,
        offtissue_count_frac=float(cout.sum() / counts.sum()),
        offtissue_level_at_first_ring=float(ys[0]) if len(ys) else None,
        ring_um=ring, profile_r_um=[float(v) for v in rs], profile_y=[float(v) for v in ys],
        profile_n=ns,
        ambient_floor=c, amplitude=a, lambda_out_um=lam, fit_adjR2=r2,
        resolved=resolved,
        lambda_below_ring=bool(lam is not None and lam < ring),
        lambda_note=("descriptive only: the amplitude and the floor are not separately "
                     "identifiable at large lambda, so lambda is not the test statistic"),
        n_perm=int(n_perm),
        near_far_ratio=near_far,
        spearman_dist_vs_counts=sp_r, spearman_p=sp_p,
        spearman_null_p01=float(np.quantile(sp_null, 0.01)) if len(sp_null) else None,
        drift_x=Dx, drift_y=Dy, drift_mag=Dmag,
        drift_deg=float(np.degrees(np.arctan2(Dy, Dx))),
        drift_p=drift_p, drift_null_p99=(float(np.quantile(d_null, 0.99))
                                         if len(d_null) else None),
    )


def main(only=None):
    res = {}
    for name, spec in DATASETS.items():
        if only and name not in only.split(","):
            continue
        print(f"[v1ctl] {name} ...", flush=True)
        loaded = load_v1(spec["path"]) if spec["kind"] == "v1" else load_hd(spec["path"])
        if loaded is None:
            print("    SKIP: raw matrix not on disk", flush=True)
            res[name] = dict(error="raw_feature_bc_matrix.h5 not on disk")
            continue
        x, y, counts, it, meta = loaded
        r = offtissue(x, y, counts, it)
        r.update(kind=spec["kind"], **meta)
        res[name] = r
        if "error" in r:
            print(f"    {r['error']}", flush=True)
            continue
        print(f"    off-tissue {r['n_off_tissue']:,} spots, {r['offtissue_count_frac']*100:.2f}% "
              f"of counts; first ring at {r['offtissue_level_at_first_ring']*100:.1f}% of the "
              f"in-tissue level", flush=True)
        print(f"    decays with distance: Spearman r = {r['spearman_dist_vs_counts']:+.3f} "
              f"(p = {r['spearman_p']:.3g} vs {r['n_perm']} shuffles, null 1st pct "
              f"{r['spearman_null_p01']:+.3f}) -> "
              f"{'YES' if r['resolved'] else 'NO'}", flush=True)
        print(f"    near/far ratio {r['near_far_ratio']:.1f}x, ambient floor "
              f"{r['ambient_floor']*100:.2f}% of in-tissue (non-spatial), "
              f"lambda_out {r['lambda_out_um']:.0f} um (descriptive), fit adjR2 "
              f"{r['fit_adjR2']:.3f}", flush=True)
        print(f"    one-sidedness |D| = {r['drift_mag']:.3f} at {r['drift_deg']:.0f} deg "
              f"(p = {r['drift_p']:.3g}, null p99 {r['drift_null_p99']:.3f})", flush=True)

    json.dump(res, open(f"{OUT}/diffusion/_v1_control.json", "w"), indent=2)
    print(f"\nwrote {OUT}/diffusion/_v1_control.json")

    # verdict: does the control behave as a control must?
    v1 = [v for k, v in res.items() if v.get("kind") == "v1" and "error" not in v]
    hd = [v for k, v in res.items() if v.get("kind") == "hd" and "error" not in v]
    if v1 and hd:
        print("\n=== CONTROL VERDICT ===")
        ok_v1 = all(v["resolved"] for v in v1)
        print(f"  leak decays with distance on v1: "
              f"{sum(v['resolved'] for v in v1)}/{len(v1)} sections "
              f"(Spearman p = {[('%.3g' % v['spearman_p']) for v in v1]}) "
              f"{'PASS' if ok_v1 else 'FAIL'}")
        print(f"  signal in the ring just outside the tissue, as % of the in-tissue level:")
        print(f"      v1 {['%.1f%%' % (v['offtissue_level_at_first_ring']*100) for v in v1]}"
              f"  vs  HD {['%.1f%%' % (v['offtissue_level_at_first_ring']*100) for v in hd]}")
        print(f"  lambda_out: v1 {['%.0f um' % v['lambda_out_um'] for v in v1]}  vs  "
              f"HD {['%.0f um' % v['lambda_out_um'] for v in hd]}")
        print(f"  ambient floor (non-spatial): "
              f"v1 {['%.2f%%' % (v['ambient_floor']*100) for v in v1]}  vs  "
              f"HD {['%.2f%%' % (v['ambient_floor']*100) for v in hd]}")
        print(f"  one-sidedness |D|: v1 {['%.3f' % v['drift_mag'] for v in v1]} "
              f"(p {[('%.3g' % v['drift_p']) for v in v1]})  vs  "
              f"HD {['%.3f' % v['drift_mag'] for v in hd]} "
              f"(p {[('%.3g' % v['drift_p']) for v in hd]})")
        dv = np.mean([v["drift_mag"] for v in v1]); dh = np.mean([v["drift_mag"] for v in hd])
        fv = np.mean([v["offtissue_level_at_first_ring"] for v in v1])
        fh = np.mean([v["offtissue_level_at_first_ring"] for v in hd])
        print(f"\n  v1 leaks {fv/fh:.1f}x more into the adjacent off-tissue ring than HD, and its "
              f"leak is {dv/dh:.1f}x more one-sided.")
        if ok_v1 and dv > dh:
            print("  => PASS. The estimators detect real lateral diffusion, and its directional "
                  "component, on a platform known to have it - in both sections independently. "
                  "The negative drift result for the six modern platforms is therefore a real "
                  "negative, not a blind test.")
        else:
            print("  => FAIL. The machinery did not separate a platform known to diffuse from one "
                  "that does not. Every negative result in the diffusion section must be "
                  "caveated as possibly reflecting an insensitive test.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    raise SystemExit(main(ap.parse_args().only))
