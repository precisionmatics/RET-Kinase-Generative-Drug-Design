"""
R1 Comment #1 — Full novelty check of all 100 candidates against training + curated sets.
Checks exact SMILES match AND high Tanimoto (>0.85) similarity.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import os

BASE = "/home/stalin/Desktop/Generative/RET_Generative"

train = pd.read_csv(f"{BASE}/data/processed/RET_training.csv")
curated = pd.read_csv(f"{BASE}/data/processed/RET_curated.csv")
cands = pd.read_csv(f"{BASE}/data/processed/RET_candidates.csv")

# Canonical SMILES for all training + curated compounds
all_known = set()
known_smiles_list = []
for df in [train, curated]:
    for smi in df['SMILES'].dropna():
        m = Chem.MolFromSmiles(smi)
        if m:
            can = Chem.MolToSmiles(m)
            all_known.add(can)
            known_smiles_list.append(can)

unique_known = list(set(known_smiles_list))
print(f"Total known compounds (train+curated): {len(unique_known)}")

# Fingerprints for known
known_fps = []
known_mols = []
for smi in unique_known:
    m = Chem.MolFromSmiles(smi)
    if m:
        known_mols.append(smi)
        known_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))

results = []
for _, row in cands.iterrows():
    smi = row['SMILES']
    m = Chem.MolFromSmiles(smi)
    if not m:
        results.append({'CandID': row.get('Rank', '?'), 'SMILES': smi,
                        'exact_match': False, 'max_tanimoto': 0.0, 'nearest_known': '', 'status': 'INVALID'})
        continue
    can = Chem.MolToSmiles(m)
    exact = can in all_known
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
    sims = DataStructs.BulkTanimotoSimilarity(fp, known_fps)
    max_sim = max(sims) if sims else 0.0
    idx = int(np.argmax(sims)) if sims else -1
    nearest = known_mols[idx] if idx >= 0 else ''
    status = 'EXACT_MATCH' if exact else ('HIGH_SIM' if max_sim > 0.85 else 'NOVEL')
    results.append({
        'CandID': f"RET_{row.get('Rank', '?'):03d}" if isinstance(row.get('Rank'), (int, float)) else str(row.get('Rank', '?')),
        'SMILES': smi,
        'exact_match': exact,
        'max_tanimoto': round(max_sim, 4),
        'nearest_known': nearest,
        'status': status
    })

out = pd.DataFrame(results)
out.to_csv(f"{BASE}/data/processed/RET_novelty_check.csv", index=False)

print("\n=== NOVELTY CHECK SUMMARY ===")
print(out['status'].value_counts())
print(f"\nExact matches: {out['exact_match'].sum()}")
print(f"High similarity (>0.85): {(out['max_tanimoto'] > 0.85).sum()}")
print(f"Novel (<0.85): {(out['max_tanimoto'] <= 0.85).sum()}")
print(f"\nNon-novel compounds:")
flagged = out[out['status'] != 'NOVEL']
for _, r in flagged.iterrows():
    print(f"  {r['CandID']}: Tc={r['max_tanimoto']:.3f}  status={r['status']}")
    print(f"    SMILES: {r['SMILES'][:80]}")
    print(f"    Nearest: {r['nearest_known'][:80]}")

# Also compute for top 5 by name
top5 = ['RET_091', 'RET_098', 'RET_092', 'RET_074', 'RET_099']
print("\n=== TOP 5 LEADS ===")
for cid in top5:
    match = out[out['CandID'] == cid]
    if not match.empty:
        r = match.iloc[0]
        print(f"  {r['CandID']}: Tc={r['max_tanimoto']:.3f} status={r['status']}")
