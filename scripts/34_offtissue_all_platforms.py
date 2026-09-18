#!/usr/bin/env python3
"""
Put ALL SIX platforms on the Visium v1 off-tissue axis.

32_visium_v1_control.py could only compare Visium v1 with Visium HD, because it keyed off the
vendor `in_tissue` flag and only barcoded capture platforms have one. The imaging platforms
(Xenium 313-plex, Prime 5K, Atera) and StrataMap have no capture grid and no such flag - but they
do have a tissue boundary, and the 8 um count fields cached by 29_diffusion.py are enough to find
it. So the mask is derived FROM THE DATA, with one definition for every platform, and the leak
outside it is measured exactly as for v1.

Two things make this honest rather than just possible:

1. INTERIOR HOLES MUST NOT COUNT AS "OUTSIDE THE TISSUE". A lumen or a fat globule inside the
   section has tissue on every side, so signal reaching it arrives from all directions - nothing
   like a spot on the free margin beyond the section edge, which is what v1's off-tissue spots
   are. The mask therefore has its holes FILLED, which counts a lumen as tissue and keeps it out
   of the margin measurement entirely. Consequence worth stating because it makes an obvious
   check come back empty: after hole-filling the complement has no interior component left (the
   interior pass below reports 0 bins on every platform, by construction and not by accident),
   so the exterior margin is the only off-tissue population there is.

2. A MASK BUILT FROM THE COUNTS IS CIRCULAR. Define tissue as "where counts are high" and then
   measure "counts outside tissue", and the threshold silently sets the answer: a permissive
   threshold swallows the leak into the mask and shrinks it. Two guards. First the whole thing is
   run at three thresholds and the spread of BOTH the leak level and its directionality is
   reported, so threshold sensitivity is visible rather than hidden. It matters: on StrataMap the
   leak level moves 19x across the three masks and is therefore not usable, while its
   directionality is stable - so the two statistics have to be judged separately, not together. Second, and more important, the two Visium HD sections have BOTH a vendor
   `in_tissue` flag and a data-derived mask, so they are measured both ways: that is the bridge
   that says whether the data-derived number can be read on the same axis as the v1 number, and
   it is reported whether it agrees or not.

Outputs outputs/diffusion/_offtissue_all.json
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_m32 = import_module("32_visium_v1_control")
offtissue = _m32.offtissue

ROOT = "/srv/GT/analysis/pgueguen/spatial_platform_comparison"
OUT  = f"{ROOT}/outputs"; DIF = f"{OUT}/diffusion"
BIN_UM = 8.0
RING_UM = 100.0          # same rings as the v1 control, so the numbers sit on one axis
THRESH_Q = [0.10, 0.20, 0.35]   # tissue-mask quantile of occupied-bin counts
PLATFORMS = ["stdxenium_breast", "prime5k_breast", "wta_breast",
             "visiumhd65_breast_8um", "visiumhd11_breast_8um", "stratamap_breast"]


def tissue_mask(tot, q):
    """One definition for every platform: bins above the q-quantile of occupied-bin counts,
    opened to drop speckle, closed to bridge gaps, holes filled."""
    from scipy.ndimage import binary_closing, binary_opening, binary_fill_holes
    occ = tot > 0
    if not occ.any():
        return None
    thr = float(np.quantile(tot[occ], q))
    m = tot > max(thr, 1.0)
    m = binary_opening(m, np.ones((3, 3), bool))
    m = binary_closing(m, np.ones((5, 5), bool))
    return binary_fill_holes(m)


def split_complement(mask):
    """(exterior, interior) boolean arrays. Exterior = the complement component touching the
    field border; everything else in the complement is an interior hole."""
    from scipy.ndimage import label
    comp = ~mask
    lab, n = label(comp)
    border = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
    border.discard(0)
    ext = np.isin(lab, list(border)) & comp
    return ext, comp & ~ext


def as_points(tot, mask, keep):
    """Raster -> the (x_um, y_um, counts, in_tissue) point form offtissue() expects, keeping the
    tissue bins plus only the requested slice of the complement."""
    sel = mask | keep
    yy, xx = np.nonzero(sel)
    return (xx * BIN_UM, yy * BIN_UM, tot[yy, xx].astype(float), mask[yy, xx])


HD_PATHS = {
    "visiumhd65_breast_8um": f"{ROOT}/data/visiumhd_65_breast/binned_outputs/square_008um",
    "visiumhd11_breast_8um": f"{ROOT}/data/visiumhd_11_breast/binned_outputs/square_008um",
}
_hd_cache = {}


def total_field(name):
    """Total counts on the 8 um lattice, covering the WHOLE captured/imaged area.

    Visium HD must come from the RAW matrix, not from the field cached by 29_diffusion.py: that
    one is built from filtered_feature_bc_matrix.h5, which contains only bins the vendor already
    called in-tissue. Measured on the cached field, Visium HD's "exterior" turns out to be mostly
    low-count area INSIDE the vendor's tissue boundary (71% of exterior bins non-zero, mean 14
    counts against an in-tissue median of 81) - a different geography from Visium v1's off-tissue
    spots, and not something to put on the same axis. The imaging platforms and StrataMap need no
    such fix: their fields are gridded from raw transcript coordinates over the full imaged
    region, blank slide included.
    """
    if name in HD_PATHS:
        if name not in _hd_cache:
            loaded = _m32.load_hd(HD_PATHS[name])
            if loaded is None:
                return None, None
            x, y, c, it, _ = loaded
            bx = np.rint(x / BIN_UM).astype(int); by = np.rint(y / BIN_UM).astype(int)
            bx -= bx.min(); by -= by.min()
            tot = np.zeros((by.max() + 1, bx.max() + 1))
            ven = np.zeros_like(tot, bool)
            np.add.at(tot, (by, bx), c)
            ven[by[it], bx[it]] = True
            _hd_cache[name] = (tot, ven)
        return _hd_cache[name]
    f = f"{DIF}/_fields_{name}.npz"
    if not os.path.exists(f):
        return None, None
    return np.load(f)["tot"], None


def run(name, q, which="exterior", n_perm=100):
    tot, _ven = total_field(name)
    if tot is None:
        return dict(error=f"no field available for {name}; run 29_diffusion.py first")
    m = tissue_mask(tot, q)
    if m is None or m.sum() < 500:
        return dict(error="tissue mask empty or tiny")
    ext, inner = split_complement(m)
    keep = ext if which == "exterior" else inner
    if keep.sum() < 200:
        return dict(error=f"only {int(keep.sum())} {which} bins")
    x, y, c, it = as_points(tot, m, keep)
    r = offtissue(x, y, c, it, ring=RING_UM, n_perm=n_perm)
    r.update(mask_quantile=q, complement=which, bin_um=BIN_UM,
             tissue_bins=int(m.sum()), exterior_bins=int(ext.sum()),
             interior_bins=int(inner.sum()),
             interior_share_of_complement=float(inner.sum() / max((~m).sum(), 1)))
    return r


def vendor_bridge(name, q):
    """For a Visium HD section: measure the SAME quantity with the vendor in_tissue flag and with
    the data-derived mask, and report how far apart they land. This is what licenses reading the
    imaging platforms' numbers on the same axis as Visium v1's."""
    v1c = json.load(open(f"{DIF}/_v1_control.json")) if os.path.exists(f"{DIF}/_v1_control.json") else {}
    vend = v1c.get(name, {})
    derived = run(name, q, n_perm=20)
    if "error" in derived or not vend or "error" in vend:
        return None
    a = vend.get("offtissue_level_at_first_ring"); b = derived.get("offtissue_level_at_first_ring")
    return dict(vendor_first_ring=a, derived_first_ring=b,
                ratio=(b / a if (a and b) else None),
                vendor_drift=vend.get("drift_mag"), derived_drift=derived.get("drift_mag"),
                vendor_unit=vend.get("unit"), derived_unit=f"{BIN_UM:g} um bin, mask q={q}")


def main(q_main=0.20):
    res = {"_config": dict(ring_um=RING_UM, bin_um=BIN_UM, mask_quantiles=THRESH_Q,
                           mask_quantile_main=q_main)}
    print(f"=== exterior margin, mask quantile {q_main} (rings of {RING_UM:g} um) ===", flush=True)
    for p in PLATFORMS:
        r = run(p, q_main, "exterior")
        res[p] = r
        if "error" in r:
            print(f"  {p:24s} SKIP: {r['error']}", flush=True)
            continue
        print(f"  {p:24s} tissue {r['tissue_bins']/1e3:6.0f}k bins | exterior "
              f"{r['exterior_bins']/1e3:6.0f}k | first ring "
              f"{r['offtissue_level_at_first_ring']*100:5.1f}% of in-tissue | "
              f"rho {r['spearman_dist_vs_counts']:+.3f} | |D| {r['drift_mag']:.3f} "
              f"(p {r['drift_p']:.3g}, null p99 {r['drift_null_p99']:.3f})", flush=True)

    print(f"\n=== interior holes, same mask (a DIFFERENT geometry, never pooled above) ===",
          flush=True)
    res["_interior"] = {}
    for p in PLATFORMS:
        r = run(p, q_main, "interior", n_perm=20)
        res["_interior"][p] = r
        if "error" in r:
            print(f"  {p:24s} SKIP: {r['error']}", flush=True)
        else:
            print(f"  {p:24s} first ring {r['offtissue_level_at_first_ring']*100:5.1f}% | "
                  f"interior is {r['interior_share_of_complement']*100:.0f}% of the complement",
                  flush=True)

    print(f"\n=== threshold sensitivity: first-ring % at mask quantiles {THRESH_Q} ===", flush=True)
    res["_threshold_sensitivity"] = {}
    for p in PLATFORMS:
        vals, dvals = [], []
        for q in THRESH_Q:
            r = run(p, q, "exterior", n_perm=5)
            vals.append(None if "error" in r else r["offtissue_level_at_first_ring"])
            dvals.append(None if "error" in r else r["drift_mag"])
        res["_threshold_sensitivity"][p] = dict(
            level=dict(zip([str(q) for q in THRESH_Q], vals)),
            drift=dict(zip([str(q) for q in THRESH_Q], dvals)))
        g = [v for v in vals if v is not None]
        gd = [v for v in dvals if v is not None]
        print(f"  {p:24s} level " + " ".join("--" if v is None else f"{v*100:5.1f}%" for v in vals)
              + (f" ({max(g)/min(g):5.2f}x)" if len(g) > 1 and min(g) > 0 else "")
              + "  |D| " + " ".join("--" if v is None else f"{v:.3f}" for v in dvals)
              + (f" ({max(gd)/min(gd):5.2f}x)" if len(gd) > 1 and min(gd) > 0 else ""), flush=True)

    print(f"\n=== bridge: vendor in_tissue flag vs data-derived mask, same section ===", flush=True)
    res["_vendor_bridge"] = {}
    for p in ("visiumhd65_breast_8um", "visiumhd11_breast_8um"):
        b = vendor_bridge(p, q_main)
        res["_vendor_bridge"][p] = b
        if b:
            print(f"  {p:24s} vendor {b['vendor_first_ring']*100:5.1f}%  vs  derived "
                  f"{b['derived_first_ring']*100:5.1f}%  ({b['ratio']:.2f}x) | "
                  f"|D| vendor {b['vendor_drift']:.3f} vs derived {b['derived_drift']:.3f}",
                  flush=True)
        else:
            print(f"  {p:24s} bridge unavailable", flush=True)

    json.dump(res, open(f"{DIF}/_offtissue_all.json", "w"), indent=2)
    print(f"\nwrote {DIF}/_offtissue_all.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.20)
    raise SystemExit(main(ap.parse_args().q))
