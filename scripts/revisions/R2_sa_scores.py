"""
R2 Comment #5 — Compute SA (synthetic accessibility) scores for all 100 candidates.
Uses RDKit's SA_Score from sascorer.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

# Try loading sascorer
try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    sys.path.insert(0, os.path.expanduser("~/miniforge3/lib/python3.13/site-packages/rdkit/Contrib/SA_Score"))
    import sascorer

BASE = "/home/stalin/Desktop/Generative/RET_Generative"
cands = pd.read_csv(f"{BASE}/data/processed/RET_candidates.csv")

sa_scores = []
for _, row in cands.iterrows():
    m = Chem.MolFromSmiles(row['SMILES'])
    if m:
        sa = sascorer.calculateScore(m)
    else:
        sa = np.nan
    sa_scores.append(round(sa, 3))

cands['SA_Score'] = sa_scores
cands.to_csv(f"{BASE}/data/processed/RET_candidates_with_sa.csv", index=False)

print("SA Score statistics for all 100 candidates:")
print(cands['SA_Score'].describe().round(3))

top5_ids = [91, 98, 92, 74, 99]
top5_names = [f"RET_{i:03d}" for i in top5_ids]
print("\nTop 5 leads SA scores:")
for i, (idx, name) in enumerate(zip([x-1 for x in top5_ids], top5_names)):
    # Find by rank
    match = cands[cands.index == (top5_ids[i]-1)]
    # Try by rank column if exists
    if 'Rank' in cands.columns:
        match = cands[cands['Rank'] == top5_ids[i]]
    if not match.empty:
        sa = match.iloc[0]['SA_Score']
        print(f"  {name}: SA={sa:.3f} ({'Easy' if sa<3 else 'Moderate' if sa<4 else 'Difficult'})")

# Figure: SA score distribution + top5 highlighted
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram of all 100
ax = axes[0]
bins = np.arange(1, 8.5, 0.5)
ax.hist(cands['SA_Score'].dropna(), bins=bins, color='#2196F3', edgecolor='white', alpha=0.85, linewidth=0.8)
ax.axvline(3.0, color='#4CAF50', lw=1.5, ls='--', label='Easy (<3)')
ax.axvline(5.0, color='#F44336', lw=1.5, ls='--', label='Difficult (>5)')
ax.set_xlabel('SA Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('SA Score Distribution — 100 Candidates', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(1, 8)
mean_sa = cands['SA_Score'].mean()
ax.axvline(mean_sa, color='navy', lw=1.5, ls=':', label=f'Mean={mean_sa:.2f}')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right: top 5 bar chart
ax2 = axes[1]
top5_sa = []
top5_labels = []
colors_top5 = ['#1565C0', '#1976D2', '#1E88E5', '#42A5F5', '#90CAF9']
for tid in top5_ids:
    if 'Rank' in cands.columns:
        match = cands[cands['Rank'] == tid]
    else:
        match = cands.iloc[[tid-1]]
    if not match.empty:
        top5_sa.append(match.iloc[0]['SA_Score'])
        top5_labels.append(f"RET_{tid:03d}")

bars = ax2.bar(top5_labels, top5_sa, color=colors_top5, edgecolor='white', linewidth=0.8, zorder=3)
ax2.set_ylim(0, 6)
ax2.axhline(3.0, color='#4CAF50', lw=1.5, ls='--', label='Easy (<3)', zorder=2)
ax2.axhline(5.0, color='#F44336', lw=1.5, ls='--', label='Difficult (>5)', zorder=2)
ax2.set_xlabel('Lead Compound', fontsize=12, fontweight='bold')
ax2.set_ylabel('SA Score', fontsize=12, fontweight='bold')
ax2.set_title('SA Scores — Top 5 Consensus Leads', fontsize=13, fontweight='bold')
for bar, val in zip(bars, top5_sa):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08, f'{val:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.legend(fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', alpha=0.3, zorder=1)

plt.tight_layout()
os.makedirs(f"{BASE}/figures/revisions", exist_ok=True)
out_path = f"{BASE}/figures/revisions/SA_scores.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved: {out_path}")
