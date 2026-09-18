# Spatial Platform Comparison: Benchmark & Deconvolution Analysis

A systematic, reproducible cross-platform benchmark comparing spatial transcriptomics technologies on human breast cancer:
- **10x Genomics Xenium** (313-plex probe · FFPE)
- **10x Genomics Xenium Prime 5K** (~5,101-plex probe · FFPE)
- **10x Genomics Atera** (18,028-gene whole-transcriptome probe · FFPE)
- **10x Genomics Visium HD** (8 µm bins and segmented cells, 6.5 mm & 11 mm · FFPE and fresh-frozen)
- **Illumina StrataMap** (poly(A) whole-transcriptome sequencing, 1 µm spatial-barcode grid · fresh-frozen, IDC Grades 1, 2, and 3)
- **Bruker/NanoString CosMx SMI** (18,942-gene whole-transcriptome + 64-plex protein · FFPE)

Authored by **Paul Gueguen** (Functional Genomics Center Zurich, ETH Zurich / University of Zurich).

---

## Key Findings & Benchmark Summary

### 1. Breadth vs. Depth Trade-Offs
Across platforms, panel breadth directly trades against per-gene sensitivity:
- Targeted panels (Xenium 313-plex) capture ~2,500 transcripts/mm²/gene on shared genes.
- High-plex panels (Prime 5K, Atera) capture 200–880 transcripts/mm²/gene.
- Sequencing-based assays (Visium HD, StrataMap) capture broad gene spaces, but per-gene sensitivity at single-cell resolution requires careful accounting for segmentation and sequencing depth.

### 2. Cell-Type Separability & The Ambient Soup (Atera vs. StrataMap)
In response to independent evaluations (Gottardo & Bilous, CHUV 2026) reporting that Illumina StrataMap fails to resolve immune cells (<0.5% T cells) while Atera cleanly identifies ~10–15% immune populations:

- **Replication of Standard Segmentation**: Running GPU-accelerated RCTD (`rctd-py`) against the Janesick snRNA-seq breast reference confirmed that in standard vendor segmentation, StrataMap Grade 1 and Grade 2 suffer massive mixture noise (**55.6% to 62.5% RCTD rejects**), with T/NK singlets landing at **<0.1%** (1–2 cells). Atera FFPE cleanly recovers **15.97% immune singlets** (10.49% T/NK) with only an 8.8% reject rate.
- **Physical Mechanism**: Proseg diffusion modeling reveals that **55.8% of transcripts in StrataMap Grade 1 belong to unassigned ambient background soup**. In standard segmentation, small lymphocytes are coated with high-abundance epithelial transcripts (*EPCAM*, *KRT8*, *MUC1*), forcing deconvolution models to reject the spot or classify it as malignant.
- **Computational Rescue**: Running probabilistic transcript de-diffusion (**Proseg**) before deconvolution slashed StrataMap rejects from **55.6% down to 21.2%**, boosted usable singlets to **76.4%**, unmasked stroma (7.5% -> 18.9%) and endothelium (2.8% -> 5.6%), and recovered **119 *CD3D*+ T-cell candidates** (63% achieving zero *EPCAM*). Subsequent **SPLIT** transcript purification eliminated 99.7% of remaining *EPCAM* spillover.

| Dataset / Pipeline | Segmentation | RCTD Rejects | Usable Singlets | Total Immune | T / NK Singlets | Myeloid | Malignant | Stroma |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Atera FFPE 5k** | Standard (10x) | **8.8%** | 33.6% | **15.97%** | **10.49%** | 4.65% | 41.3% | 31.3% |
| **StrataMap Grade 1 5k** | Standard (Illumina) | **55.6%** | 44.4% | **1.13%** | **0.09%** *(2 cells)* | 0.90% | 87.8% | 7.48% |
| **StrataMap Grade 2 5k** | Standard (Illumina) | **62.5%** | 37.4% | **1.50%** | **0.05%** *(1 cell)* | 1.39% | 96.3% | 1.82% |
| **StrataMap Grade 3 5k** | Standard (Illumina) | **44.0%** | 55.9% | **6.23%** | **0.14%** *(4 cells)* | 5.73% | 89.9% | 3.26% |
| **StrataMap Grade 1 (Proseg)** | **Proseg De-diffused** | **21.2%** | **76.4%** | **2.23%** | **0.05%** | **1.96%** | 72.3% | **18.86%** |
| **StrataMap Grade 1 (Proseg + SPLIT)** | **De-diffused + Purified** | — | — | **2.31%** | *(50 T spots purified, 99.7% EPCAM stripped)* | 2.10% | — | — |

---

## Repository Structure

```
├── scripts/
│   ├── 01_download.sh                   # Fetch all public 10x breast & cervical datasets
│   ├── 02_metrics.py                    # Recompute core per-cell and per-area metrics
│   ├── 03_aggregate.py                  # Cross-platform summary tables
│   ├── 04_figures.py                    # Generate comparison figures (01-09)
│   ├── 05_build_artifact.py             # Build self-contained HTML report with all figures
│   ├── 06_bin8um.py & 07_bin8um_figure  # Segmentation-free 8 µm lattice comparison
│   ├── 10_cosmx.py & 11_cosmx_figure    # Bruker CosMx SMI multiomic comparison
│   ├── 12_stratamap_passes.sh           # Stream Illumina StrataMap 44.8M SBC matrix
│   ├── 16_stratamap_percell_typed.py    # StrataMap contour rasterisation and typing
│   ├── 29_diffusion.py & 30_figure      # Lateral diffusion and decay length estimation
│   ├── 31_grid_registration.py         # Capture grid vs tissue registration
│   ├── 32_visium_v1_control.py          # Real-data positive diffusion control on Visium v1
│   ├── 34_offtissue_all_platforms.py    # Off-tissue lateral leak across all 6 platforms
│   ├── 36_rctd_gpu_benchmark.py         # GPU-accelerated RCTD cell deconvolution (rctd-py)
│   ├── 37_split_purify.R                # SPLIT transcript purification on deconvolution weights
│   ├── 38_proseg_diffusion.sh           # Proseg probabilistic transcript de-diffusion
│   └── 39_immune_rescue_figure.py       # Publication figure for RCTD and immune rescue (Fig 28)
├── outputs/
│   ├── bundle.json                      # Comprehensive metrics bundle for all platforms
│   ├── master_metrics.csv               # Master metric comparison table
│   ├── concordance_L1.csv               # Cross-platform pseudobulk concordance (304 shared genes)
│   ├── concordance_L2.csv               # High-plex cross-platform concordance (4,843 genes)
│   ├── shared_genes_L1.txt              # Gene symbols in core L1 comparison
│   ├── shared_genes_L2.txt              # Gene symbols in high-plex L2 comparison
│   ├── gpu_rctd_benchmark_summary.json  # Full RCTD deconvolution benchmark metrics
│   └── figs/                            # Publication-quality figures (PNG, 200-300 DPI)
├── README.md                            # Documentation and methodology
└── LICENSE                              # MIT License
```

---

## Reproducing the Analysis

### 1. Requirements

- Python 3.10+ with `anndata`, `scanpy`, `numpy`, `scipy`, `pandas`, `matplotlib`, `pyarrow`
- GPU deconvolution: `torch` with CUDA, `rctd-py` (`pip install rctd-py`)
- R 4.5+ with `Matrix`, `SPLIT` (>= 0.2.0), `qs2`, `anndataR`
- Proseg (for transcript de-diffusion modeling)

### 2. Running GPU Deconvolution Benchmark

```bash
python scripts/36_rctd_gpu_benchmark.py \
  --spatial data/stratamap_breast_grade1_5k.h5ad \
  --reference data/janesick_breast_reference.h5ad \
  --cell-type-col cell_type \
  --output outputs/StrataMap_Grade1_5k_rctd_results.csv \
  --device cuda
```

### 3. Running SPLIT Purification

```bash
Rscript scripts/37_split_purify.R \
  data/stratamap_roi_proseg_5k.h5ad \
  outputs/StrataMap_Proseg_5k_rctd_results.csv \
  data/ref_profiles_10k.csv \
  outputs/StrataMap_Proseg_5k_split_purified.qs2
```

### 4. Building the Full HTML Report

```bash
python scripts/05_build_artifact.py
# Produces outputs/spatial_platform_comparison.html
```

---

## Data Sources

All analyzed data are publicly available:
- **10x Genomics**: Atera Breast Cancer Preview, Xenium FFPE Human Breast (Janesick et al.), Xenium Prime 5K Human Breast, Visium HD Human Breast (6.5 mm & 11 mm TMA).
- **Illumina StrataMap**: NovaSeq X Fresh-Frozen Human Breast Cancer (IDC Grades 1, 2, and 3), DRAGEN Spatial Transcriptome workflow (BaseSpace Sequence Hub Data Central).
- **Bruker/NanoString**: CosMx Human Multiomic Breast Cancer (FFPE).

---

## Citation & Contact

Paul Gueguen  
Functional Genomics Center Zurich (ETH Zurich / University of Zurich)  
Contact: `pgueguen@ethz.ch`
