# Spatial Platform Comparison: Benchmark & Deconvolution Analysis

A reproducible cross-platform benchmark of spatial transcriptomics technologies on human breast cancer:
- **10x Genomics Xenium** (313-plex probe · FFPE)
- **10x Genomics Xenium Prime 5K** (~5,101-plex probe · FFPE)
- **10x Genomics Atera** (18,028-gene whole-transcriptome probe, public preview data · FFPE)
- **10x Genomics Visium HD** (8 µm bins and segmented cells, 6.5 mm & 11 mm · FFPE and fresh-frozen)
- **Illumina StrataMap** (poly(A) whole-transcriptome sequencing, 1 µm spatial-barcode grid · fresh-frozen, IDC grades 1, 2 and 3)
- **Bruker/NanoString CosMx SMI** (18,942-gene whole-transcriptome + 64-plex protein · FFPE)

Authored by **Paul Gueguen** (Functional Genomics Center Zurich, ETH Zurich / University of Zurich).

Every comparable number is **recomputed from the count matrices** with one definition per metric.
Vendor `metrics_summary.csv` values are not used for any cross-platform number, because they are not
defined the same way across chemistries.

---

## Key findings

### 1. Breadth vs. depth
Across platforms, panel breadth trades against per-gene sensitivity (breast, 304 shared genes, L1):
- The targeted 313-plex Xenium panel captures ~2,500 transcripts/mm²/gene.
- High-plex panels capture 128 (Prime 5K) to 880 (Atera) transcripts/mm²/gene.
- StrataMap (fresh-frozen) reaches 3,909. Read that with the section-thickness caveat below: its demo
  section is 10 µm, the FFPE sections are most likely 5 µm, and no per-area number here is normalised
  for thickness.

### 2. RCTD on vendor vs. Proseg segmentation (Atera vs. StrataMap)
RCTD (`rctd-py` 0.3.8, doublet mode) on one 1.5 x 1.5 mm window per platform, each segmented two
ways: the vendor's cells and Proseg's. Every run uses the reference genes only and the same thresholds,
so within a platform only the segmentation changes (`40_rctd_matched_rerun.py`).

- **Proseg barely moves either platform.** Rejects: Atera 14.0% → 13.1%,
  StrataMap grade 1 18.5% → 19.1%. Immune share of singlets: Atera
  9.8% → 8.4%, StrataMap 0.9% → 1.4%.
- **The immune gap is what separates the platforms**: 9.8% vs 0.9% of singlets
  (T/NK 6.2% vs 0.0%), under a reference whose poly-A chemistry is closer to
  StrataMap's. It does not separate chemistry from specimen (fresh-frozen DCIS/IDC grade 1 vs FFPE grade 3).
- In the grade 1 window, cells with *CD3D* ≥ 2: 28 vendor, 19 Proseg (of ~21.4k); Atera 267 / 315 (of 12.4k).

| Platform | Segmentation | Cells | Rejects | Singlets | Immune (of singlets) | T / NK | Malignant / epithelial | Stroma |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atera | vendor (10x) | 12,398 | 14.0% | 29.1% | 9.8% | 6.2% | 65.8% | 21.9% |
| Atera | Proseg | 12,388 | 13.1% | 27.0% | 8.4% | 3.7% | 64.4% | 24.0% |
| StrataMap G1 | vendor (Illumina) | 21,419 | 18.5% | 69.3% | 0.9% | 0.0% | 71.1% | 25.7% |
| StrataMap G1 | Proseg | 20,503 | 19.1% | 63.7% | 1.4% | 0.0% | 72.1% | 23.5% |
| StrataMap G1 | vendor, first config* | 21,425 | 45.9% | 53.8% | 0.7% | 0.0% | 86.2% | 12.5% |
| StrataMap G1 | Proseg, first config* | 20,725 | 29.6% | 65.7% | 1.2% | 0.0% | 78.5% | 18.4% |

\* The first version of this analysis multiplied RCTD's thresholds by each input's feature count / 5,000
(12.4 for StrataMap vendor, 7.0 for StrataMap Proseg, 3.6 for Atera). Same cells, same window: that
config alone takes StrataMap vendor rejects from 18.5% to 45.9%. See the corrections log.

### 3. Lateral diffusion
Diffusion tests are calibrated before use (injected blur and offsets, `--selftest`) and checked
against a real-data positive control (Visium v1 off-tissue spots). Visium v1 leaks 42.2% / 38.8% of
its in-tissue level into the first off-tissue ring (two sections), against 15.2% / 11.4% for Visium HD,
and its leak is ~7x more one-sided. StrataMap's directional statistic is threshold-unstable
(|D| 0.065-0.257 across tissue-mask thresholds), so no directional ranking against v1 is made for it.

---

## Metric definitions

**Sensitivity, full panel.** Median transcripts per cell, median genes per cell, transcripts per mm².
Total yield only, not per-gene comparable across panels of different size.

**Sensitivity, shared gene.** Every platform restricted to a common gene set, then median
transcripts/cell and mean transcripts per gene per cell. **L1**: 304 genes present in all five breast
tiers. **L2**: 4,843 genes present in Prime 5K, Atera WTA and both Visium HD. Gene sets in
`outputs/shared_genes_L1.txt` and `shared_genes_L2.txt`. Symbol intersection uses a small
current-to-legacy alias map (`ALIAS` in `02_metrics.py`: `KARS1`, `LARS1`, `NARS1`, `QARS1`, `WARS1`,
`CYRIA`). Both sets are built from 10x-designed probe panels, so the "shared" axis is not
chemistry-neutral.

**Specificity.** Imaging platforms: negative-control probe and codeword rates from the h5 feature
rows. Visium HD has no negative-control probes; cross-platform pseudobulk concordance on shared genes
is the accuracy proxy that works everywhere.

**Diffusion.** Two tests on the 8 µm lattice (no segmentation): a marker-half domain test giving a
decay length `λ` and a directional statistic `|D|` (`29_diffusion.py`), and capture-grid vs.
tissue-image registration (`31_grid_registration.py`). `λ` is withheld unless the fit converged, the
in/out contrast is ≥1.5x, and the displaced-domain null is under half the fitted value. `λ` mixes real
biology with technical spillover and is an upper bound on the technical part.

**Per-cell unit.** Xenium tiers use native segmented cells. Visium HD 11 mm uses SpaceRanger
`segmented_outputs`; 6.5 mm has no segmentation and uses 8 µm bins. StrataMap uses Illumina's
Expanded-5 µm contours, whose cell IDs match its nuclei 1:1. An 8 µm lattice is also computed for every
platform so one line of the report has no segmentation in it (`06_bin8um.py`).

**Per-area.** Real geometry: Xenium cell coordinates and region area; Visium HD `n_tissue_bins x
bin_area`; StrataMap occupied 8 µm bins.

---

## Caveats

- **Section thickness is not normalised.** StrataMap's demo is a 10 µm fresh-frozen section; the
  Xenium/Atera FFPE sections are most likely 5 µm (not confirmed from the dataset pages). A 10 µm
  section carries about twice the tissue per mm², which inflates StrataMap's per-area numbers.
- **Per-cell numbers track cell area.** StrataMap's median cell area is 40-49 µm² (grades 1-3), Atera's
  64.9 µm²; on Atera, transcripts per cell correlate with cell area at Spearman 0.81. Per-cell
  comparisons between these platforms are partly comparisons of segmented area.
- **Different specimens and fixation.** The breast sections come from different blocks, not serial
  sections; StrataMap is fresh-frozen, every other breast column FFPE. Differences include biology and
  prep, not chemistry alone.
- **Atera is pre-commercial preview data** from a development pipeline.
- **RCTD depends on the reference.** Ours is a CELLxGENE Census breast-cancer reference of 10x 3′/5′
  poly-A data restricted to the Atera panel genes (`00_build_rctd_reference.py`); its chemistry is
  closer to StrataMap's than to Atera's. A Chromium Flex (probe) reference would lean the other way.
- **n = 1 per platform** for most comparisons, and one 1.5 x 1.5 mm window per platform for the
  segmentation work.
- Visium HD 6.5 mm has no cell segmentation; the 11 mm section is a TMA. Cervical covers two imaging
  tiers only. Prices are 10x list prices, not FGCZ quotes. Singular G4X and Element Teton appear as a
  vendor-specification tier only, walled off from every recomputed number.

## Corrections log

- **2026-09-23.** The first RCTD comparison (`36_rctd_gpu_benchmark.py`) compared a section-wide vendor
  subset with a Proseg window, and scaled RCTD's thresholds with each input's feature count, which
  gave StrataMap a 3.4x larger `CONFIDENCE_THRESHOLD` than Atera and therefore more rejects. Replaced by
  `40_rctd_matched_rerun.py` (same window, reference genes only, one threshold). The numbers in the
  first version of this README (55.6% → 21.2% rejects) are superseded.
- **2026-09-23.** An earlier README stated that Proseg assigns 55.8% of StrataMap grade 1 transcripts to
  ambient background. No output supports that figure: Proseg flags 5.5% as background, and the vendor
  contours leave 16.8% of the window's transcripts outside every cell.
- **2026-09-23.** The RCTD reference was mislabelled "Janesick et al. single-nucleus"; it is the Census
  poly-A reference above.

## Provenance gaps

- The section-wide 5,000-cell RCTD inputs (`rctd_inputs/*_5k.h5ad`) and the original `*_gpu_rctd_results.csv`
  files were produced by an earlier script version (fixed sigma 0.8) that is not in this repository.
  They are no longer used for any reported number.

---

## Repository structure

```
scripts/
  00_build_rctd_reference.py       CELLxGENE Census RCTD reference (breast, cervix)
  01_download.sh                   fetch the public 10x datasets (Atera: one manual download)
  02_metrics.py .. 05_build_artifact.py   metrics, aggregation, figures, HTML report
  06/07                            8 µm lattice, no segmentation
  08/09                            dissociated scRNA-seq sensitivity reference
  10/11                            CosMx comparison
  12..23                           StrataMap streaming passes, per-cell aggregation, QC
  24..28                           gene-abundance distribution, bulk / TCGA comparisons, reference reconcile
  29..35                           diffusion, grid registration, Visium v1 positive control, off-tissue axis
  36_rctd_gpu_benchmark.py         SUPERSEDED first RCTD run (kept as a record)
  37_split_purify.R                SPLIT purification on RCTD weights
  38_proseg_diffusion.sh           Proseg resegmentation
  39_immune_rescue_figure.py       figure 28, from 40's summary
  40_rctd_matched_rerun.py         RCTD, vendor vs Proseg, same window, one config
outputs/                           derived tables and per-dataset metric JSONs behind every figure
```

Paths are absolute at the top of each script (`ROOT = ...`); change `ROOT` and the `data/` layout
created by `01_download.sh` is the only assumption.

## Reproducing

- Python 3.12 with `numpy`, `scipy`, `polars`, `pandas`, `anndata`, `h5py`, `matplotlib`,
  `scikit-image`, `pillow`; `cellxgene-census` for `00` and `08`; `rctd-py` ≥ 0.3.8 with `torch` for
  `40` (a GPU is strongly recommended: sigma estimation alone took ~13 min per 12k-cell run on an L40S).
- R 4.5+ with `Matrix`, `SPLIT` (≥ 0.2.0), `qs2`, `anndataR` for `37`. Proseg for `38`.

```bash
python scripts/00_build_rctd_reference.py
python scripts/40_rctd_matched_rerun.py --build
python scripts/40_rctd_matched_rerun.py --run sm_vendor_fixed   # one per condition, see RUNS
python scripts/40_rctd_matched_rerun.py --summarise
python scripts/39_immune_rescue_figure.py
python scripts/05_build_artifact.py                              # outputs/spatial_platform_comparison.html
```

## Data sources

- **10x Genomics**: Atera preview (FFPE breast and cervical cancer), Xenium FFPE Human Breast
  (Janesick et al. 2023, replicate 1), Xenium Prime 5K breast and cervical, Visium HD breast (6.5 mm and
  11 mm TMA), Visium v1 breast block A sections 1 and 2 (diffusion control only).
- **Illumina StrataMap**: NovaSeq X fresh-frozen human breast cancer demo (DCIS/IDC grades 1, 2, 3),
  BaseSpace Sequence Hub Data Central (account required).
- **Bruker/NanoString**: CosMx Human Multiomic Breast Cancer (FFPE).
- **CZ CELLxGENE Census** 2025-11-08: RCTD reference.

No raw data are included. Derived tables in `outputs/` are computed from the datasets above and
remain subject to their providers' terms; the MIT licence covers the code.

## Contact

Paul Gueguen, Functional Genomics Center Zurich (ETH Zurich / University of Zurich), `pgueguen@ethz.ch`
