"""
R1 #10, R2 #9 — Reference compound docking with correct SMILES from PubChem.
Dock selpercatinib, pralsetinib, nintedanib using same AutoDock-GPU protocol.
"""
import os, subprocess, json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

BASE = "/home/stalin/Desktop/Generative/RET_Generative"
DOCK_DIR = f"{BASE}/docking"
RECEPTOR_FLD = f"{DOCK_DIR}/receptor/6NEC_receptor.maps.fld"
AUTODOCK_GPU = "/home/stalin/Downloads/Enamine/GIPR/autodock_gpu_ocl"
OUTDIR = f"{DOCK_DIR}/reference_compounds"
os.makedirs(OUTDIR, exist_ok=True)

# Verified SMILES from PubChem (correct CIDs, MWs confirmed)
COMPOUNDS = {
    'selpercatinib': {
        'smiles': 'CC(C)(COC1=CN2C(=C(C=N2)C#N)C(=C1)C3=CN=C(C=C3)N4CC5CC(C4)N5CC6=CN=C(C=C6)OC)O',
        'MW': 525.6, 'CID': 134436906
    },
    'pralsetinib': {
        'smiles': 'CC1=CC(=NN1)NC2=NC(=NC(=C2)C)C3CCC(CC3)(C(=O)N[C@@H](C)C4=CN=C(C=C4)N5C=C(C=N5)F)OC',
        'MW': 533.6, 'CID': 129073603
    },
    'nintedanib': {
        'smiles': 'CN1CCN(CC1)CC(=O)N(C)C2=CC=C(C=C2)N=C(C3=CC=CC=C3)C4=C(NC5=C4C=CC(=C5)C(=O)OC)O',
        'MW': 539.6, 'CID': 135423438
    },
}

def make_pdbqt(name, smiles):
    m = Chem.MolFromSmiles(smiles)
    if not m:
        print(f"  [ERROR] Invalid SMILES for {name}")
        return None
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(m, params) < 0:
        params.useRandomCoords = True
        AllChem.EmbedMolecule(m, params)
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    sdf = os.path.join(OUTDIR, f"{name}_correct.sdf")
    pdbqt = os.path.join(OUTDIR, f"{name}_correct.pdbqt")
    with Chem.SDWriter(sdf) as w:
        w.write(m)
    r = subprocess.run(f"mk_prepare_ligand.py -i {sdf} -o {pdbqt}",
                       shell=True, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(pdbqt):
        print(f"  [ERROR] meeko failed for {name}: {r.stderr[-300:]}")
        return None
    print(f"  {name}: PDBQT ready ({os.path.getsize(pdbqt)} bytes)")
    return pdbqt

def parse_dlg(dlg_path):
    """Extract best docking ΔG from DLG file."""
    best = None
    if not os.path.exists(dlg_path):
        return None
    with open(dlg_path) as f:
        for line in f:
            if 'DOCKED: USER' in line and 'Estimated Free Energy' in line:
                try:
                    best = float(line.split('=')[1].split('kcal')[0].strip())
                    break
                except:
                    pass
    if best is None:
        with open(dlg_path) as f:
            content = f.read()
        for line in content.split('\n'):
            s = line.strip()
            if s.startswith('1 ') and '|' in s:
                parts = s.split('|')
                if len(parts) >= 2:
                    try:
                        best = float(parts[1].strip().split()[0])
                        break
                    except:
                        pass
    return best

print("=== Reference Compound Docking ===\n")
results = []
for name, info in COMPOUNDS.items():
    print(f"--- {name} (MW={info['MW']}) ---")
    pdbqt = make_pdbqt(name, info['smiles'])
    if not pdbqt:
        results.append({'Name': name, 'MW': info['MW'], 'DockingScore': None})
        continue
    filelist = os.path.join(OUTDIR, f"{name}_filelist.txt")
    with open(filelist, 'w') as f:
        f.write(os.path.abspath(RECEPTOR_FLD) + "\n")
        f.write(os.path.abspath(pdbqt) + "\n")
    cmd = (f"{AUTODOCK_GPU} --filelist {filelist} "
           f"--nrun 50 --autostop 1 "
           f"--resnam {os.path.join(OUTDIR, name + '_correct')}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    # DLG file location
    dlg = os.path.join(OUTDIR, f"{name}_correct.dlg")
    if not os.path.exists(dlg):
        # AutoDock-GPU puts DLG next to filelist
        dlg = os.path.join(os.path.dirname(filelist), f"{name}_correct.dlg")
    dg = parse_dlg(dlg)
    print(f"  ΔG = {dg}")
    results.append({'Name': name, 'MW': info['MW'], 'DockingScore': dg,
                    'SMILES': info['smiles'], 'PubChemCID': info['CID']})

df = pd.DataFrame(results)
out_csv = f"{BASE}/data/processed/RET_reference_docking.csv"
df.to_csv(out_csv, index=False)
print(f"\n=== FINAL RESULTS ===")
print(df.to_string(index=False))
print(f"\nSaved: {out_csv}")
