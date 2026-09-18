#!/usr/bin/env Rscript
# 37_split_purify.R
# Ambient transcript purification using SPLIT and rctd-py soft deconvolution weights.
suppressPackageStartupMessages({
  library(Matrix)
  library(SPLIT)
  library(qs2)
  library(anndataR)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript 37_split_purify.R <counts.h5ad> <rctd_results.csv> <ref_profiles.csv> <out.qs2>")
}

counts_path <- args[1]
rctd_csv_path <- args[2]
ref_csv_path <- args[3]
out_path <- args[4]

cat("[SPLIT] Reading counts from:", counts_path, "\n")
cat("[SPLIT] Reading RCTD results from:", rctd_csv_path, "\n")
cat("[SPLIT] Reading reference profiles from:", ref_csv_path, "\n")

ann <- anndataR::read_h5ad(counts_path)
counts <- t(as(ann$X, "CsparseMatrix"))
colnames(counts) <- ann$obs_names
rownames(counts) <- ann$var_names

df <- read.csv(rctd_csv_path, check.names = FALSE)
wcols <- grep("^w\\.", colnames(df), value = TRUE)
stopifnot(length(wcols) > 0)

# Filter for cells present in both
common_cells <- intersect(colnames(counts), df$barcode)
cat(sprintf("[SPLIT] Matched %d cells between counts and RCTD predictions\n", length(common_cells)))
stopifnot(length(common_cells) > 0)

counts <- counts[, common_cells, drop = FALSE]
df <- df[match(common_cells, df$barcode), ]

W <- as.matrix(df[, wcols, drop = FALSE])
rownames(W) <- df$barcode
colnames(W) <- sub("^w\\.", "", wcols)
W <- W / pmax(rowSums(W), 1e-12)

primary <- df$first_type
names(primary) <- df$barcode

# Load reference expression profiles (types x genes)
ref_df <- read.csv(ref_csv_path, row.names = 1, check.names = FALSE)
ref_mat <- as.matrix(ref_df)

# Align types
stopifnot(all(colnames(W) %in% rownames(ref_mat)))
ref_mat <- ref_mat[colnames(W), , drop = FALSE]

# Intersect genes
common_genes <- intersect(rownames(counts), colnames(ref_mat))
cat(sprintf("[SPLIT] Intersected %d common genes between panel and reference\n", length(common_genes)))

ref_mat <- ref_mat[, common_genes, drop = FALSE]
counts_sub <- counts[common_genes, , drop = FALSE]

cat(sprintf("[SPLIT] Running rctd_free_purify (%d cells, %d genes, %d types)...\n", 
            ncol(counts_sub), nrow(counts_sub), ncol(W)))
t0 <- Sys.time()
res <- SPLIT::rctd_free_purify(
  counts = counts_sub,
  deconvolution_weights = W,
  primary_cell_type = primary,
  reference = ref_mat,
  DO_run_in_chunks = TRUE,
  chunk_size = 5000
)
elapsed <- round(difftime(Sys.time(), t0, units = "secs"), 1)
cat(sprintf("[SPLIT] Completed in %s seconds\n", elapsed))

pc <- res$purified_counts
cm <- res$cell_meta

purified_frac <- mean(cm$purification_status == "purified")
cat(sprintf("[SPLIT] Fraction of cells purified: %.1f%%\n", 100 * purified_frac))

# Check marker expression before and after in T cells
t_cell_names <- grep("T cell|T-cell|CD4|CD8|T_NK", unique(primary), value = TRUE)
t_idx <- which(primary %in% t_cell_names)
cat(sprintf("[SPLIT] Identified %d primary T-cell spots\n", length(t_idx)))

if (length(t_idx) > 0 && "EPCAM" %in% common_genes) {
  raw_epcam <- mean(counts_sub["EPCAM", t_idx])
  pur_epcam <- mean(pc["EPCAM", t_idx])
  cat(sprintf("[SPLIT] Mean EPCAM in T cells: Raw = %.3f -> Purified = %.3f (%.1f%% reduction)\n",
              raw_epcam, pur_epcam, 100 * (1 - pur_epcam / pmax(raw_epcam, 1e-9))))
}

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
out_list <- list(
  purified_counts = pc,
  cell_meta = cm,
  rctd_df = df
)
qs2::qs_save(out_list, out_path, nthreads = 4)
cat("[SPLIT] Saved results to:", out_path, "\n")
