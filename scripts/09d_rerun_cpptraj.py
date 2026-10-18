"""
Rerun cpptraj with proper RMS-fitting to get correct backbone RMSD / ligand RMSD / RMSF / Rg.
The original run used 'nofit' which bakes in translational+rotational drift.
"""
import subprocess, os, json

BASE   = "/home/stalin/Desktop/RET_Generative"
UNOMD  = "/home/stalin/miniforge3/envs/unomd"
CPPTRAJ = f"{UNOMD}/bin/cpptraj"

def write_cpptraj_in(cand, prmtop, dcd, ana_dir, n_lig_res=288):
    script = f"""parm {prmtop}
trajin {dcd}

# Strip solvent first, keep protein + ligand
strip :WAT,Na+,Cl-

# RMS-fit to average structure (iterative, backbone Cα only) — CORRECT approach
rms first @CA out /dev/null

# Backbone RMSD (already aligned above, ref = first stripped frame)
rmsd backbone @CA first out {ana_dir}/backbone_rmsd.dat

# Ligand RMSD (fit ligand atoms to their first-frame positions)
rmsd ligand_rmsd :{n_lig_res} first :{n_lig_res} out {ana_dir}/ligand_rmsd.dat

# RMSF per Cα (after fitting)
atomicfluct rmsf @CA byres out {ana_dir}/rmsf_ca.dat

# Radius of gyration (protein only, mass-weighted)
radgyr Rg :1-{n_lig_res-1} out {ana_dir}/rg.dat mass

run
quit
"""
    in_path = f"{ana_dir}/cpptraj_fixed.in"
    with open(in_path, 'w') as f:
        f.write(script)
    return in_path

for cand in ["RET_074", "RET_098"]:
    md_dir  = f"{BASE}/md/{cand}"
    ana_dir = f"{md_dir}/analysis"
    prmtop  = f"{md_dir}/solvated.prmtop"
    # pick the correct DCD name
    dcd = f"{md_dir}/{cand}_prod.dcd"
    if not os.path.exists(dcd):
        print(f"  WARNING: DCD not found at {dcd}")
        continue

    print(f"\n=== {cand} ===")
    in_path = write_cpptraj_in(cand, prmtop, dcd, ana_dir)
    log_path = f"{ana_dir}/cpptraj_fixed.log"
    cmd = [CPPTRAJ, "-i", in_path]
    print(f"  Running: {' '.join(cmd)}")
    with open(log_path, 'w') as logf:
        ret = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                             env={**os.environ, "AMBERHOME": UNOMD})
    if ret.returncode != 0:
        print(f"  ERROR (code {ret.returncode}), check {log_path}")
        with open(log_path) as f:
            print(f.read()[-2000:])
    else:
        # quick sanity check
        import numpy as np
        dat = []
        with open(f"{ana_dir}/backbone_rmsd.dat") as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    dat.append(float(line.split()[1]))
        arr = np.array(dat)
        print(f"  backbone RMSD: mean={arr.mean():.3f} Å  max={arr.max():.3f} Å  n={len(arr)}")

print("\nDone. Now regenerate figures.")
