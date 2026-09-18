#!/usr/bin/env python3
"""
36_rctd_gpu_benchmark.py
GPU-accelerated RCTD cell type deconvolution for whole-transcriptome spatial platforms (rctd-py).
Benchmarking 10x Atera vs Illumina StrataMap (Grades 1, 2, 3) and Proseg-dediffused data.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import torch
from rctd import Reference, run_rctd, RCTDConfig

def run_benchmark(spatial_path, reference_path, cell_type_col="cell_type",
                  output_csv="rctd_results.csv", device="cuda", batch_size=8000):
    print(f"Loading spatial data: {spatial_path}")
    spatial = ad.read_h5ad(spatial_path)
    
    print(f"Loading scRNA-seq reference: {reference_path}")
    ref_adata = ad.read_h5ad(reference_path)
    ref = Reference(ref_adata, cell_type_col=cell_type_col)
    
    n_vars = spatial.shape[1]
    # Scale doublet thresholds proportionally to panel size (scaled from 5k-gene baseline)
    scale = max(1.0, n_vars / 5000.0)
    doublet_thresh = 20.0 * scale
    conf_thresh = 5.0 * scale
    
    config = RCTDConfig(
        device=device,
        DOUBLET_THRESHOLD=doublet_thresh,
        CONFIDENCE_THRESHOLD=conf_thresh,
        sigma_override=0.8,
        UMI_min=20
    )
    
    print(f"Running GPU RCTD: {spatial.shape[0]} cells x {n_vars} genes, {len(ref.cell_types)} reference types")
    t0 = time.time()
    res = run_rctd(spatial, ref, mode="doublet", config=config, batch_size=batch_size)
    elapsed = time.time() - t0
    print(f"Deconvolution completed in {elapsed:.1f}s")
    
    df_out = pd.DataFrame(index=spatial.obs_names)
    df_out["barcode"] = spatial.obs_names
    df_out["spot_class"] = res.spot_class
    df_out["first_type"] = res.first_type
    df_out["second_type"] = res.second_type
    
    ct_names = ref.cell_types
    for i, ct in enumerate(ct_names):
        df_out[f"w.{ct}"] = res.weights[:, i]
        
    df_out.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    return df_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPU RCTD benchmark on spatial transcriptomics")
    parser.add_argument("--spatial", required=True, help="Input spatial AnnData (.h5ad)")
    parser.add_argument("--reference", required=True, help="Reference scRNA-seq AnnData (.h5ad)")
    parser.add_argument("--cell-type-col", default="cell_type", help="Column in reference obs with cell types")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=8000, help="Batch size for GPU computation")
    args = parser.parse_args()
    
    run_benchmark(args.spatial, args.reference, args.cell_type_col, args.output, args.device, args.batch_size)
