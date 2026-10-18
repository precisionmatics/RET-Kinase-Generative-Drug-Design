"""
R2 Comment #1 — How were multiple IC50 measurements handled during deduplication?
Analyses the n_measurements column and variability across measurements.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE = "/home/stalin/Desktop/Generative/RET_Generative"

train = pd.read_csv(f"{BASE}/data/processed/RET_training.csv")
curated = pd.read_csv(f"{BASE}/data/processed/RET_curated.csv")

print("Training set columns:", train.columns.tolist())
print("Curated set columns:", curated.columns.tolist())

for label, df in [('Curated (674)', curated), ('Training (546)', train)]:
    if 'n_measurements' in df.columns:
        print(f"\n=== {label} ===")
        print(f"n_measurements distribution:")
        print(df['n_measurements'].value_counts().sort_index().to_string())
        multi = df[df['n_measurements'] > 1]
        print(f"\nCompounds with >1 measurement: {len(multi)} ({100*len(multi)/len(df):.1f}%)")
        print(f"Max measurements per compound: {df['n_measurements'].max()}")
        print(f"Mean measurements per compound: {df['n_measurements'].mean():.2f}")
    else:
        print(f"\n{label}: no n_measurements column")

# Count from raw data - get multi-measurement stats
if 'n_measurements' in curated.columns:
    multi = curated[curated['n_measurements'] > 1]
    print(f"\nSummary for manuscript:")
    print(f"  Total curated: {len(curated)}")
    print(f"  Single measurement: {(curated['n_measurements'] == 1).sum()}")
    print(f"  Multiple measurements: {(curated['n_measurements'] > 1).sum()}")
    print(f"  Geometric mean used for pIC50 aggregation")
