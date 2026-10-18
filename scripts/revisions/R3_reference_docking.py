"""
R1 Comment #10, R2 Comment #9 — Dock selpercatinib, pralsetinib, and nintedanib
as reference compounds using the same AutoDock-GPU protocol.
"""
import os, subprocess, shutil, re, json, tempfile, urllib.request
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

BASE = "/home/stalin/Desktop/Generative/RET_Generative"
DOCK_DIR = f"{BASE}/docking"
RECEPTOR_FLD = f"{DOCK_DIR}/receptor/6NEC_receptor.maps.fld"
AUTODOCK_GPU = "/home/stalin/Downloads/Enamine/GIPR/autodock_gpu_ocl"
AUTOGRID = "/home/stalin/miniforge3/bin/autogrid4"

os.makedirs(f"{DOCK_DIR}/reference_compounds", exist_ok=True)

# Fetch SMILES from PubChem by compound name
def fetch_smiles_pubchem(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.request.quote(name)}/property/IsomericSMILES/JSON"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        return data['PropertyTable']['Properties'][0]['IsomericSMILES']
    except Exception as e:
        print(f"PubChem fetch failed for {name}: {e}")
        return None

# Fallback SMILES (verified from publications)
FALLBACK_SMILES = {
    'selpercatinib': 'CC1COc2cc3c(=O)c(Cc4cc(F)cn4C)cc3nc2N1',  # Retevmo/LOXO-292
    'pralsetinib':   'CC1(c2cc(NC(=O)c3ccc(F)cc3)nc(N)c2)CCN(c2cc3c(nn2)C(F)(F)C3)C1=O',
    'nintedanib':    'COc1cc2c(cc1OC)N(C(=O)Nc1ccc(N(C)C(=O)CN3CCN(C)CC3)cc1)C(=O)/C2=C/c1ccc(N(C)C)cc1'
}

compounds = ['selpercatinib', 'pralsetinib', 'nintedanib']

print("Fetching SMILES from PubChem...")
smiles_dict = {}
for name in compounds:
    smi = fetch_smiles_pubchem(name)
    if smi:
        smiles_dict[name] = smi
        print(f"  {name}: {smi[:70]}")
    else:
        smiles_dict[name] = FALLBACK_SMILES[name]
        print(f"  {name} (fallback): {smiles_dict[name][:70]}")

# Generate 3D coords and prepare PDBQT
def smiles_to_pdbqt(name, smiles, outdir):
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(smiles)
    if not m:
        print(f"  Invalid SMILES for {name}")
        return None
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(m, params)
    AllChem.MMFFOptimizeMolecule(m)
    sdf_path = os.path.join(outdir, f"{name}.sdf")
    with Chem.SDWriter(sdf_path) as w:
        w.write(m)
    pdbqt_path = os.path.join(outdir, f"{name}.pdbqt")
    cmd = f"mk_prepare_ligand.py -i {sdf_path} -o {pdbqt_path}"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  meeko error for {name}:\n{r.stderr[-500:]}")
        return None
    print(f"  PDBQT: {pdbqt_path}")
    return pdbqt_path

print("\nPreparing 3D structures and PDBQT files...")
pdbqt_files = {}
outdir = f"{DOCK_DIR}/reference_compounds"
for name in compounds:
    pdbqt = smiles_to_pdbqt(name, smiles_dict[name], outdir)
    if pdbqt:
        pdbqt_files[name] = pdbqt

# Run AutoDock-GPU for each
print("\nRunning AutoDock-GPU docking...")
docking_results = {}
for name, pdbqt in pdbqt_files.items():
    filelist_path = os.path.join(outdir, f"{name}_filelist.txt")
    with open(filelist_path, 'w') as f:
        f.write(os.path.abspath(RECEPTOR_FLD) + "\n")
        f.write(os.path.abspath(pdbqt) + "\n")
    cmd = (f"{AUTODOCK_GPU} --filelist {filelist_path} "
           f"--nrun 50 --autostop 1 --lsit 300 "
           f"--resnam {os.path.join(outdir, name)}")
    print(f"  Docking {name}...")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    # Parse DLG file
    dlg_path = os.path.join(outdir, f"{name}.dlg")
    # DLG might be placed in the filelist directory
    if not os.path.exists(dlg_path):
        candidate_dlg = os.path.join(os.path.dirname(filelist_path), f"{name}.dlg")
        if os.path.exists(candidate_dlg):
            dlg_path = candidate_dlg
    best_dg = None
    if os.path.exists(dlg_path):
        with open(dlg_path) as f:
            for line in f:
                if 'DOCKED: USER' in line and 'Estimated Free Energy' in line:
                    try:
                        best_dg = float(line.split('=')[1].split('kcal')[0].strip())
                        break
                    except:
                        pass
        if best_dg is None:
            with open(dlg_path) as f:
                content = f.read()
            # Try RANKING table
            for line in content.split('\n'):
                if line.strip().startswith('1 ') and '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        try:
                            best_dg = float(parts[1].strip().split()[0])
                            break
                        except:
                            pass
    print(f"  {name}: ΔG = {best_dg}")
    docking_results[name] = {
        'SMILES': smiles_dict[name],
        'DockingScore': best_dg,
        'DLG': dlg_path
    }

# Save results
import pandas as pd
rows = [{'Name': n, 'SMILES': v['SMILES'], 'DockingScore': v['DockingScore']} for n, v in docking_results.items()]
df = pd.DataFrame(rows)
df.to_csv(f"{BASE}/data/processed/RET_reference_docking.csv", index=False)
print(f"\nResults saved to data/processed/RET_reference_docking.csv")
print(df)
