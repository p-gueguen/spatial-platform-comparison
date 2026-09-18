#!/bin/bash
# Sequential regen after the L1 -> 304 flagship-core change. One job, no double-background.
set -uo pipefail
export PATH=/usr/local/ngseq/miniforge3/envs/ps_pgueguen/bin:$PATH
cd /srv/GT/analysis/pgueguen/spatial_platform_comparison
L=logs/rerun_driver.log
echo "[driver] start $(date +%H:%M:%S)" > $L

# wait for the already-running 17 (stratamap per-cell) to finish writing the json
p17=$(pgrep -f 17_stratamap_percell_fast.py | head -1)
if [ -n "${p17:-}" ]; then
  echo "[driver] waiting on 17 pid=$p17" >> $L
  while kill -0 "$p17" 2>/dev/null; do sleep 5; done
fi
echo "[driver] 17 done $(date +%H:%M:%S)" >> $L

python scripts/23_stratamap_percell_pc.py >> $L 2>&1 && echo "[driver] 23 ok" >> $L
python scripts/19_stratamap_bin8um.py     >> $L 2>&1 && echo "[driver] 19 ok" >> $L
python scripts/03_aggregate.py            >> $L 2>&1 && echo "[driver] 03 ok" >> $L
python scripts/04_figures.py              >> $L 2>&1 && echo "[driver] 04 ok" >> $L
python scripts/07_bin8um_figure.py        >> $L 2>&1 && echo "[driver] 07 ok" >> $L
python scripts/18_percell_sharedgene_figure.py >> $L 2>&1 && echo "[driver] 18 ok" >> $L
echo "[driver] ALL DONE $(date +%H:%M:%S)" >> $L
