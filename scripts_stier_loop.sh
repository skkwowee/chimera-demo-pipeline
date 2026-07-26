#!/bin/bash
# b-lite archival scrape (corpus-strategy #8, amended: S-TIER ONLY, stars>=4).
# Stage-1 only: HLTV -> demos -> HF. Nothing baked, nothing merged.
# Paced: 8 matches/pass, 90min sleep => well under rate limits.
# Stops after 40 passes (~300 matches) or when STOP_SCRAPE file exists.
BIN=~/projects/chimera-demo-pipeline/.venv/bin/chimera-demo
LOG=~/projects/chimera-demo-pipeline/logs/stier_scrape.log
for pass in $(seq 1 40); do
  [ -f ~/projects/chimera-demo-pipeline/STOP_SCRAPE ] && echo "$(date) STOP file — exiting" >> "$LOG" && exit 0
  echo "$(date) === pass $pass/40 ===" >> "$LOG"
  $BIN run --stars 4 -n 8 >> "$LOG" 2>&1
  sleep 5400
done
echo "$(date) 40 passes done — exiting" >> "$LOG"
