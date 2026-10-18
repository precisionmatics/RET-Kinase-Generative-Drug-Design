#!/usr/bin/env python3
"""
Re-parse all FINAL_RESULTS_MMPBSA.dat files and fix result.json.
Run after 06_mmpbsa_mmgbsa.py completes to correct the parser bug
(MMPBSA output uses spaces not '=' between DELTA TOTAL and value).
"""
import re
import json
from pathlib import Path

LIG_DIR = Path("/home/stalin/Desktop/RET_Generative/mmpbsa/ligands")


def parse(dat_file):
    text = Path(dat_file).read_text()
    pb_val = gb_val = None
    sections = re.split(r'(POISSON BOLTZMANN|GENERALIZED BORN)', text, flags=re.IGNORECASE)
    mode = None
    for seg in sections:
        upper = seg.upper().strip()
        if "POISSON BOLTZMANN" in upper:
            mode = "pb"
        elif "GENERALIZED BORN" in upper:
            mode = "gb"
        else:
            m = re.search(r'DELTA TOTAL\s+([-+]?\d+\.?\d*)', seg)
            if m:
                val = float(m.group(1))
                if mode == "pb" and pb_val is None:
                    pb_val = val
                elif mode == "gb" and gb_val is None:
                    gb_val = val
    return pb_val, gb_val


fixed = failed = 0
for dat in sorted(LIG_DIR.glob("*/FINAL_RESULTS_MMPBSA.dat")):
    cand = dat.parent.name
    cache = dat.parent / "result.json"
    pb, gb = parse(dat)
    if pb is None and gb is None:
        print(f"  WARN {cand}: still null after re-parse — check {dat}")
        failed += 1
        continue
    data = {"CandID": cand, "DELTA_TOTAL_pb": pb, "DELTA_TOTAL_gb": gb}
    cache.write_text(json.dumps(data, indent=2))
    print(f"  FIXED {cand}: PB={pb:.2f}  GB={gb:.2f}")
    fixed += 1

print(f"\nFixed: {fixed}  Still-null: {failed}")
