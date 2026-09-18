#!/usr/bin/env python3
"""
Is the capture grid registered to the tissue? The GLOBAL version of the drift question.

29_diffusion.py asks whether one gene set leaks preferentially in one direction RELATIVE TO
another gene set in the same section. That test is blind by construction to a rigid offset of the
whole capture grid against the tissue, because such an offset moves every gene together. But a
rigid offset is exactly what Visium v1's systematic transcript drift was: signal recovered a fixed
distance away from the cells that produced it. So it has to be measured separately.

Only the SEQUENCING platforms can fail this way. On Xenium / Atera the transcript coordinates and
the morphology image come out of the same optical frame in the same instrument pass, so there is no
grid to mis-register against the tissue; the transcript's x,y IS an image coordinate. On Visium HD
the barcode lattice is a physical feature of the slide and the tissue image is registered to it in
software, so the two can disagree.

Method: put the per-bin total-counts field and the tissue image on the SAME 8 um bin lattice, then
find the (dy, dx) shift maximising their masked Pearson correlation, refined to sub-bin precision by
a parabolic fit around the integer peak. Reported in um.

Positive control (--selftest): shift the image field by a known offset and confirm the estimator
returns it. Without that, a "0 um offset" result is worthless.

Outputs outputs/diffusion/_registration.json
"""
import os, json, argparse
import numpy as np

ROOT = "/srv/GT/analysis/pgueguen/spatial_platform_comparison"
OUT  = f"{ROOT}/outputs"; D = f"{ROOT}/data"
BIN_UM = 8.0
SEARCH = 6           # +/- bins searched = +/- 48 um

HD = {
    "visiumhd65_breast_8um": f"{D}/visiumhd_65_breast/binned_outputs/square_008um",
    "visiumhd11_breast_8um": f"{D}/visiumhd_11_breast/binned_outputs/square_008um",
}


def load_fields(base):
    """Return (counts, tissue, valid) as 2D arrays on the array_row/array_col bin lattice."""
    import polars as pl, h5py, scipy.sparse as sp
    from PIL import Image
    with h5py.File(f"{base}/filtered_feature_bc_matrix.h5", "r") as f:
        gm = f["matrix"]
        M = sp.csc_matrix((gm["data"][:], gm["indices"][:], gm["indptr"][:]),
                          shape=tuple(int(x) for x in gm["shape"][:]))
        bcs = gm["barcodes"][:].astype(str)
    tot = np.asarray(M.sum(0)).ravel()

    tp = pl.read_parquet(f"{base}/spatial/tissue_positions.parquet")
    sf = json.load(open(f"{base}/spatial/scalefactors_json.json"))
    hs = float(sf["tissue_hires_scalef"])

    img = np.asarray(Image.open(f"{base}/spatial/tissue_hires_image.png").convert("L"),
                     dtype=np.float64)
    # H&E / eosin on a pale background: tissue is DARK, so invert to get a density.
    tis_img = 255.0 - img

    idx = {b: i for i, b in enumerate(tp["barcode"].to_list())}
    rows = np.array(tp["array_row"].to_list()); cols = np.array(tp["array_col"].to_list())
    prow = np.array(tp["pxl_row_in_fullres"].to_list()); pcol = np.array(tp["pxl_col_in_fullres"].to_list())

    r0, c0 = rows.min(), cols.min()
    H, W = int(rows.max() - r0) + 1, int(cols.max() - c0) + 1
    counts = np.zeros((H, W)); tissue = np.zeros((H, W)); valid = np.zeros((H, W), bool)

    # tissue image sampled at every bin's own pixel position -> same lattice as the counts
    iy = np.rint(prow * hs).astype(int); ix = np.rint(pcol * hs).astype(int)
    inb = (iy >= 0) & (iy < tis_img.shape[0]) & (ix >= 0) & (ix < tis_img.shape[1])
    rr = (rows - r0).astype(int); cc = (cols - c0).astype(int)
    tissue[rr[inb], cc[inb]] = tis_img[iy[inb], ix[inb]]
    valid[rr[inb], cc[inb]] = True

    keep = np.array([idx.get(b, -1) for b in bcs])
    ok = keep >= 0
    counts[rr[keep[ok]], cc[keep[ok]]] = tot[ok]
    return counts, tissue, valid


def _masked_r(a, b, m):
    if m.sum() < 500:
        return np.nan
    x = a[m]; y = b[m]
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


def offset(counts, tissue, valid, search=SEARCH):
    """Shift the TISSUE field over the counts field; return the best (dy, dx) in um and the surface."""
    surf = np.full((2 * search + 1, 2 * search + 1), np.nan)
    for i, dy in enumerate(range(-search, search + 1)):
        for j, dx in enumerate(range(-search, search + 1)):
            t = np.roll(np.roll(tissue, dy, axis=0), dx, axis=1)
            v = np.roll(np.roll(valid, dy, axis=0), dx, axis=1) & valid & (counts > 0)
            surf[i, j] = _masked_r(counts, t, v)
    # Polarity is a property of the image, not of the registration: on some releases the tissue
    # is dark on pale background, on others the reverse, so counts can anti-correlate with the
    # inverted grey. Take the sign from the zero-shift correlation and maximise in that direction.
    sign = 1.0 if (np.isfinite(surf[search, search]) and surf[search, search] >= 0) else -1.0
    work = sign * surf
    i0, j0 = np.unravel_index(np.nanargmax(work), work.shape)
    # parabolic sub-bin refinement along each axis around the integer peak
    def refine(v_m, v_0, v_p):
        den = v_m - 2 * v_0 + v_p
        return 0.0 if not np.isfinite(den) or den == 0 else float(0.5 * (v_m - v_p) / den)
    interior = (0 < i0 < work.shape[0] - 1) and (0 < j0 < work.shape[1] - 1)
    dsy = refine(work[i0 - 1, j0], work[i0, j0], work[i0 + 1, j0]) if interior else 0.0
    dsx = refine(work[i0, j0 - 1], work[i0, j0], work[i0, j0 + 1]) if interior else 0.0
    dy = (i0 - search + dsy) * BIN_UM
    dx = (j0 - search + dsx) * BIN_UM
    # Validity: a peak sitting ON the search boundary means the estimator never locked on, and a
    # weak peak means there is nothing to lock on to. Either way the offset is not a measurement.
    rpk = float(work[i0, j0])
    valid = bool(interior and rpk >= 0.15)
    why = None if valid else (
        "the correlation peak sits on the edge of the search window, so the estimator never "
        "locked on" if not interior else
        f"the peak correlation is only {rpk:.3f}: the counts field and the tissue image do not "
        f"agree well enough for an offset to mean anything")
    return dict(offset_y_um=dy, offset_x_um=dx,
                offset_mag_um=float(np.hypot(dy, dx)),
                offset_deg=float(np.degrees(np.arctan2(dy, dx))),
                polarity=float(sign),
                r_at_peak=rpk, r_at_zero=float(sign * surf[search, search]),
                r_gain=float(rpk - sign * surf[search, search]),
                integer_peak_bins=[int(i0 - search), int(j0 - search)],
                peak_interior=bool(interior), valid=valid, invalid_because=why,
                surface=[[None if not np.isfinite(v) else round(float(v), 5) for v in row]
                         for row in surf])


def main(selftest=False):
    res = {}
    for name, base in HD.items():
        img = f"{base}/spatial/tissue_hires_image.png"
        if not (os.path.exists(img) and os.path.getsize(img) > 0):
            print(f"[reg] {name}: SKIP, no readable {img}", flush=True)
            res[name] = dict(error="tissue_hires_image.png not present on disk")
            continue
        print(f"[reg] {name} ...", flush=True)
        counts, tissue, valid = load_fields(base)
        r = offset(counts, tissue, valid)
        r["n_bins"] = int((valid & (counts > 0)).sum())
        print(f"    offset = {r['offset_mag_um']:.1f} um "
              f"(dy={r['offset_y_um']:.1f}, dx={r['offset_x_um']:.1f}), "
              f"r {r['r_at_zero']:.3f} -> {r['r_at_peak']:.3f}, "
              f"polarity {r['polarity']:+.0f}, "
              f"{'VALID' if r['valid'] else 'INVALID: ' + str(r['invalid_because'])}", flush=True)

        if selftest:
            # positive control: displace the tissue field by a known amount and re-measure
            r["controls"] = {}
            for dy, dx in ((2, 0), (0, 3), (-4, 2)):
                ts = np.roll(np.roll(tissue, dy, axis=0), dx, axis=1)
                vs = np.roll(np.roll(valid, dy, axis=0), dx, axis=1)
                c = offset(counts, ts, vs)
                # the estimator shifts the tissue to match the counts, so it must UNDO the
                # injected displacement: recovered = measured - injected
                got = (c["offset_y_um"] - r["offset_y_um"], c["offset_x_um"] - r["offset_x_um"])
                want = (-dy * BIN_UM, -dx * BIN_UM)
                err = float(np.hypot(got[0] - want[0], got[1] - want[1]))
                ok = err <= BIN_UM
                r["controls"][f"inject_{dy}_{dx}bins"] = dict(
                    injected_um=list(want), recovered_um=[round(g, 2) for g in got],
                    error_um=round(err, 2), passed=bool(ok))
                print(f"    control inject ({dy},{dx}) bins: want {want}, got "
                      f"({got[0]:.1f},{got[1]:.1f}), err {err:.1f} um -> "
                      f"{'PASS' if ok else 'FAIL'}", flush=True)
        res[name] = r

    res["_note"] = ("Imaging platforms (Xenium 313-plex, Prime 5K, Atera) are absent by design: "
                    "their transcript coordinates and morphology image come from the same optical "
                    "frame, so there is no capture grid that could be mis-registered against the "
                    "tissue. This test applies only where the barcode lattice and the tissue image "
                    "are registered to each other in software.")
    json.dump(res, open(f"{OUT}/diffusion/_registration.json", "w"), indent=2)
    allc = [c["passed"] for v in res.values() if isinstance(v, dict)
            for c in v.get("controls", {}).values()]
    if selftest:
        print(f"\n[selftest] {sum(allc)}/{len(allc)} injected-offset controls recovered within "
              f"{BIN_UM:g} um")
        return 0 if allc and all(allc) else 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="also inject known offsets and check they are recovered")
    a = ap.parse_args()
    raise SystemExit(main(selftest=a.selftest))
