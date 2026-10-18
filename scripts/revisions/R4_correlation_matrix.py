"""
R2 Comment #8 — Correlation between the 6 scoring methods.
Produces: (a) 6x6 Spearman correlation heatmap, (b) updated consensus heatmap with reference compounds.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

BASE = "/home/stalin/Desktop/Generative/RET_Generative"
os.makedirs(f"{BASE}/figures/revisions", exist_ok=True)

# Load all scoring data
dock  = pd.read_csv(f"{BASE}/data/processed/RET_docking_results.csv")    # CandID, DockingScore
boltz = pd.read_csv(f"{BASE}/data/processed/RET_boltz2_affinity.csv")    # CandID, DockingScore, pIC50_boltz
mmpb  = pd.read_csv(f"{BASE}/data/processed/RET_mmpbsa_results.csv")     # CandID, DELTA_TOTAL_pb, DELTA_TOTAL_gb
gnina = pd.read_csv(f"{BASE}/data/processed/RET_gnina_results.csv")      # CandID, cnn_score, cnn_affinity

print("dock cols:", dock.columns.tolist())
print("boltz cols:", boltz.columns.tolist())
print("mmpb cols:", mmpb.columns.tolist())
print("gnina cols:", gnina.columns.tolist())

# Use top 20 (those with all 6 scores)
top20_ids = mmpb['CandID'].tolist()[:20]

# Build merged dataframe for top 20
merged = mmpb[['CandID', 'DELTA_TOTAL_gb', 'DELTA_TOTAL_pb']].copy()
merged = merged.rename(columns={'DELTA_TOTAL_gb': 'MM-GBSA', 'DELTA_TOTAL_pb': 'MM-PBSA'})

# Add docking score
dock_sub = dock[dock['CandID'].isin(top20_ids)][['CandID', 'DockingScore']] if 'CandID' in dock.columns else None
if dock_sub is None and 'Ligand' in dock.columns:
    dock['CandID'] = dock['Ligand']
    dock_sub = dock[dock['CandID'].isin(top20_ids)][['CandID', 'DockingScore']]
if dock_sub is not None:
    merged = merged.merge(dock_sub, on='CandID', how='left')
    merged = merged.rename(columns={'DockingScore': 'AutoDock-GPU'})

# Add Boltz-2
boltz_sub = boltz[boltz['CandID'].isin(top20_ids)][['CandID', 'pIC50_boltz']]
merged = merged.merge(boltz_sub, on='CandID', how='left')
merged = merged.rename(columns={'pIC50_boltz': 'Boltz-2 pIC50'})

# Add GNINA
gnina_sub = gnina[gnina['CandID'].isin(top20_ids)][['CandID', 'cnn_score', 'cnn_affinity']]
merged = merged.merge(gnina_sub, on='CandID', how='left')
merged = merged.rename(columns={'cnn_score': 'GNINA CNNscore', 'cnn_affinity': 'GNINA CNNaffinity'})

print(f"\nMerged shape: {merged.shape}")
print(merged.head())

# Compute 6x6 Spearman correlation
score_cols = ['AutoDock-GPU', 'Boltz-2 pIC50', 'MM-GBSA', 'MM-PBSA', 'GNINA CNNscore', 'GNINA CNNaffinity']
# Filter to available cols
score_cols = [c for c in score_cols if c in merged.columns]
print(f"\nScore columns: {score_cols}")

score_data = merged[score_cols].dropna()
n = len(score_data)
print(f"Compounds with all scores: {n}")

corr_matrix = pd.DataFrame(np.ones((len(score_cols), len(score_cols))),
                            index=score_cols, columns=score_cols)
pval_matrix = pd.DataFrame(np.ones((len(score_cols), len(score_cols))),
                            index=score_cols, columns=score_cols)
for i, c1 in enumerate(score_cols):
    for j, c2 in enumerate(score_cols):
        if i != j:
            rho, pval = spearmanr(score_data[c1], score_data[c2])
            corr_matrix.loc[c1, c2] = rho
            pval_matrix.loc[c1, c2] = pval

print("\nSpearman correlation matrix:")
print(corr_matrix.round(3))

# Plot correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
cmap = plt.cm.RdYlGn
im = ax.imshow(corr_matrix.values.astype(float), cmap=cmap, vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Spearman ρ')

n_cols = len(score_cols)
ax.set_xticks(range(n_cols))
ax.set_yticks(range(n_cols))
short_labels = ['AutoDock\nGPU', 'Boltz-2\npIC50', 'MM-GBSA', 'MM-PBSA', 'GNINA\nCNNscore', 'GNINA\nCNNaffin.']
short_labels = short_labels[:n_cols]
ax.set_xticklabels(short_labels[:n_cols], fontsize=9)
ax.set_yticklabels(short_labels[:n_cols], fontsize=9)

for i in range(n_cols):
    for j in range(n_cols):
        val = corr_matrix.values[i, j].astype(float)
        pval = pval_matrix.values[i, j]
        stars = ''
        if i != j:
            if pval < 0.001:
                stars = '***'
            elif pval < 0.01:
                stars = '**'
            elif pval < 0.05:
                stars = '*'
        txt = f"{val:.2f}{stars}"
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, txt, ha='center', va='center', fontsize=8.5, fontweight='bold', color=color)

ax.set_title(f'Spearman Correlation Between Six Scoring Methods\n(n={n} compounds, *p<0.05, **p<0.01, ***p<0.001)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
out_path = f"{BASE}/figures/revisions/scoring_correlation_matrix.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nCorrelation matrix saved: {out_path}")

# Save correlation matrix as CSV
corr_matrix.to_csv(f"{BASE}/data/processed/RET_scoring_correlation.csv")
print("CSV saved.")
