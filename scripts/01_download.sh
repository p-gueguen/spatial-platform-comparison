#!/usr/bin/env bash
# Download the missing 10x public datasets for the spatial platform comparison.
# Xenium: individual files (small). VisiumHD: tarballs (matrix only available bundled).
# On-disk already (symlinked, not downloaded): std Xenium breast (Janesick), Atera/WTA breast + cervical.
set -uo pipefail   # NOTE: no -e; we want to attempt every file and report at the end

ROOT=/srv/GT/analysis/pgueguen/spatial_platform_comparison
DATA=$ROOT/data
LOG=$ROOT/logs/download.log
mkdir -p "$DATA" "$ROOT/logs"
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
REF='https://www.10xgenomics.com/'

say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
get () {  # get <url> <dest>
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then say "SKIP (exists): $dest"; return 0; fi
  say "GET $url"
  wget -q --show-progress --continue --tries=5 --timeout=60 \
       --user-agent="$UA" --header="Referer: $REF" \
       -O "$dest.part" "$url" && mv "$dest.part" "$dest" \
    && say "OK   $(du -h "$dest" | cut -f1)  $dest" \
    || { say "FAIL $url"; return 1; }
}

# ---------- Xenium Prime 5K (breast + cervical): individual files ----------
for tis in Breast Cervical; do
  base="https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_${tis}_Cancer_FFPE/Xenium_Prime_${tis}_Cancer_FFPE"
  out="$DATA/prime5k_${tis,,}"; mkdir -p "$out"
  get "${base}_cell_feature_matrix.h5" "$out/cell_feature_matrix.h5"
  get "${base}_cells.parquet"          "$out/cells.parquet"
  get "${base}_metrics_summary.csv"    "$out/metrics_summary.csv"
  get "${base}_gene_panel.json"        "$out/gene_panel.json"
done

# ---------- VisiumHD 6.5mm breast (SR 3.1.2; binned only, no segmentation) ----------
out="$DATA/visiumhd_65_breast"; mkdir -p "$out"
b65="https://cf.10xgenomics.com/samples/spatial-exp/3.1.2/Visium_HD_Human_Breast_Cancer_FFPE/Visium_HD_Human_Breast_Cancer_FFPE"
get "${b65}_metrics_summary.csv"   "$out/metrics_summary.csv"
get "${b65}_binned_outputs.tar.gz" "$out/binned_outputs.tar.gz"

# ---------- VisiumHD 11mm breast TMA (SR 4.1.0; binned + segmented) ----------
out="$DATA/visiumhd_11_breast"; mkdir -p "$out"
b11="https://cf.10xgenomics.com/samples/spatial-exp/4.1.0/Visium_HD_11mm_Human_Breast_Cancer_TMA/Visium_HD_11mm_Human_Breast_Cancer_TMA"
get "${b11}_metrics_summary.csv"      "$out/metrics_summary.csv"
get "${b11}_binned_outputs.tar.gz"    "$out/binned_outputs.tar.gz"
get "${b11}_segmented_outputs.tar.gz" "$out/segmented_outputs.tar.gz"

# ---------- Selective extraction: only what we need (8um bins; segmented cell matrix) ----------
say "Extracting VisiumHD 8um bins + segmented cell matrix (selective) ..."
cd "$DATA/visiumhd_65_breast" && [[ -f binned_outputs.tar.gz ]] && \
  tar -xzf binned_outputs.tar.gz --wildcards '*square_008um*' 2>>"$LOG" && say "extracted 6.5mm 8um"
cd "$DATA/visiumhd_11_breast" && [[ -f binned_outputs.tar.gz ]] && \
  tar -xzf binned_outputs.tar.gz --wildcards '*square_008um*' 2>>"$LOG" && say "extracted 11mm 8um"
cd "$DATA/visiumhd_11_breast" && [[ -f segmented_outputs.tar.gz ]] && \
  tar -xzf segmented_outputs.tar.gz 2>>"$LOG" && say "extracted 11mm segmented"

# ---------- Symlink on-disk datasets into one tree ----------
say "Symlinking on-disk datasets ..."
ln -sfn /srv/GT/analysis/rdegottardi/data/Janeson/xenium/outs "$DATA/stdxenium_breast"
ln -sfn /srv/GT/analysis/pgueguen/rctd-py/atera/breast        "$DATA/wta_breast"
ln -sfn /srv/GT/analysis/pgueguen/rctd-py/atera/cervical      "$DATA/wta_cervical"

say "==== DOWNLOAD STAGE COMPLETE ===="
du -sh "$DATA"/* 2>/dev/null | tee -a "$LOG"
