#!/bin/bash
# Two streaming passes over the StrataMap Grade3 raw SBC matrix (segmentation-free).
# Pass A: gene pseudobulk (row -> total counts) from matrix.mtx.gz
# Pass B: 8um-occupied area + coord extent from barcodes.tsv.gz (SBC:Y:X in nm)
set -uo pipefail
W=/srv/GT/analysis/pgueguen/spatial_platform_comparison/data/stratamap_breast/grade3/Raw_Matrix_Files
O=/srv/GT/analysis/pgueguen/spatial_platform_comparison/data/stratamap_breast
DEC="gzip -dc"; command -v pigz >/dev/null && DEC="pigz -dc"
echo "[$(date +%T)] decompressor: $DEC"

echo "[$(date +%T)] Pass A: gene pseudobulk ..."
$DEC "$W/matrix.mtx.gz" | awk 'NR>3{a[$1]+=$3; t+=$3; nz++} END{
  print "#TOTAL_TX", t; print "#NNZ", nz;
  for(g in a) print g, a[g]
}' > "$O/grade3_pseudobulk_rows.txt"
echo "[$(date +%T)] Pass A done: $(wc -l < "$O/grade3_pseudobulk_rows.txt") gene rows"

echo "[$(date +%T)] Pass B: 8um-occupied bins + extent ..."
$DEC "$W/barcodes.tsv.gz" | awk -F: '
  { y=$2+0; x=$3+0;
    by=int(y/8000); bx=int(x/8000); k=by"_"bx;
    if(!(k in s)){s[k]=1; nb++}
    if(NR==1||y<ymin)ymin=y; if(y>ymax)ymax=y;
    if(NR==1||x<xmin)xmin=x; if(x>xmax)xmax=x;
    n++ }
  END{ print "#N_SBC", n; print "#N_BIN8UM", nb;
       print "#YMIN_NM", ymin; print "#YMAX_NM", ymax;
       print "#XMIN_NM", xmin; print "#XMAX_NM", xmax }
' > "$O/grade3_area.txt"
echo "[$(date +%T)] Pass B done:"; cat "$O/grade3_area.txt"
echo "[$(date +%T)] ALL DONE"
