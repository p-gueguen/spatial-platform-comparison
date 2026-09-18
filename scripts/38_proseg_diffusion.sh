#!/bin/bash
# 38_proseg_diffusion.sh
# Probabilistic transcript de-diffusion modeling using Proseg.
# Reassigns ambient diffused transcripts to background and recovers authentic cell boundaries.
set -euo pipefail

TRANSCRIPTS=${1:-"transcripts.parquet"}
CELLS=${2:-"cells.parquet"}
OUTPUT_DIR=${3:-"proseg_output"}

echo "=== Running Proseg Probabilistic De-diffusion ==="
echo "Input transcripts: $TRANSCRIPTS"
echo "Initial segmentation: $CELLS"
echo "Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Run proseg with diffusion model
proseg \
  --transcripts "$TRANSCRIPTS" \
  --cells "$CELLS" \
  --output-dir "$OUTPUT_DIR" \
  --diffusion-sigma 2.5 \
  --iterations 50 \
  --threads 16

echo "Proseg diffusion modeling completed."
