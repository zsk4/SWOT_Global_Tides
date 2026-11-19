#!/bin/bash

# Sorts SWOT data into folders based on pass number
folder=~/scratch/data/SWOT_Global_Tides

for f in "$folder"/SWOT_L3_LR_SSH_Expert_*.nc; do
    [ -f "$f" ] || continue
    pass=$(basename "$f" | cut -d'_' -f7)
    mkdir -p "$folder/SWOT_Pass${pass}"
    mv -n "$f" "$folder/SWOT_Pass${pass}/"
done