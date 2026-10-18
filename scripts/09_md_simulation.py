#!/usr/bin/env python3
"""
Step 09: MD Simulation — Top 3 RET Kinase Leads
Pipeline: tleap solvation → OpenMM 100 ns production → cpptraj analysis → figures

Top 3 from 6-method consensus + decomp: RET_074, RET_098, RET_091
"""

import os
import re
import json
import textwrap
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path("/home/stalin/Desktop/RET_Generative")
MD_BASE     = ROOT / "md"
FIG_DIR     = ROOT / "figures" / "09_md"
RESULTS_DIR = ROOT / "results"
SCRIPTS_DIR = ROOT / "scripts"

MMPBSA_LIGS = ROOT / "mmpbsa" / "ligands"
RECEPTOR_PDB = ROOT / "docking" / "receptor" / "6NEC_chainA.pdb"

UNOMD_BIN  = Path("/home/stalin/miniforge3/envs/unomd/bin")
UNOMD_PY   = UNOMD_BIN / "python"
TLEAP      = UNOMD_BIN / "tleap"
CPPTRAJ    = UNOMD_BIN / "cpptraj"
AMBER_ENV  = {**os.environ, "AMBERHOME": str(UNOMD_BIN.parent)}

MD_SUITE_PY = Path("/home/stalin/miniforge3/envs/md-suite/bin/python")
OPENMM_RUN  = SCRIPTS_DIR / "09b_openmm_run.py"

TOP3 = ["RET_074", "RET_098", "RET_091"]

# Key binding site residues (AMBER 1-based)
KEY_RESIDUES = {93: "GLU805", 95: "ALA807", 46: "LYS758",
                94: "TYR806", 167: "ASP892", 92: "VAL804"}

for d in [MD_BASE, FIG_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Utilities ──────────────────────────────────────────────────────────────────
def run(cmd, cwd=None, log_path=None, check=True, env=None):
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, env=env or os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if log_path:
        Path(log_path).write_text(result.stdout)
    if check and result.returncode != 0:
        tail = result.stdout[-3000:] if result.stdout else "(no output)"
        raise RuntimeError(f"Command failed (rc={result.returncode}):\n{cmd}\n\n{tail}")
    return result.stdout


# ── Step 1: Solvate with tleap ────────────────────────────────────────────────
def solvate(cand_id, md_dir):
    prmtop  = md_dir / "solvated.prmtop"
    inpcrd  = md_dir / "solvated.inpcrd"

    if prmtop.exists() and inpcrd.exists():
        print(f"  [skip] {cand_id} — solvated topology exists")
        return prmtop, inpcrd

    lig_mol2   = MMPBSA_LIGS / cand_id / "lig.mol2"
    lig_frcmod = MMPBSA_LIGS / cand_id / "lig.frcmod"

    for f in [lig_mol2, lig_frcmod, RECEPTOR_PDB]:
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")

    tleap_in = textwrap.dedent(f"""\
        source leaprc.protein.ff14SB
        source leaprc.water.tip3p
        source leaprc.gaff2

        LIG = loadmol2 {lig_mol2}
        loadamberparams {lig_frcmod}
        check LIG

        rec = loadpdb {RECEPTOR_PDB}
        complex = combine {{rec LIG}}
        check complex

        solvateBox complex TIP3PBOX 12.0
        addIonsRand complex Na+ 0
        addIonsRand complex Cl- 0
        addIons2 complex Na+ 0
        addIons2 complex Cl- 0

        saveamberparm complex {prmtop} {inpcrd}
        savepdb complex {md_dir / 'solvated_complex.pdb'}
        quit
    """)
    leap_inp = md_dir / "tleap_solvate.in"
    leap_inp.write_text(tleap_in)

    print(f"  {cand_id}: solvating with tleap (ff14SB + GAFF2 + TIP3P)...")
    out = run(f"{TLEAP} -f {leap_inp}", cwd=md_dir,
              log_path=md_dir / "tleap_solvate.log", env=AMBER_ENV, check=False)

    if not prmtop.exists():
        raise RuntimeError(f"tleap solvation failed for {cand_id}.\n{out[-2000:]}")

    size = prmtop.stat().st_size / 1024
    print(f"  {cand_id}: solvated.prmtop = {size:.0f} KB")
    return prmtop, inpcrd


# ── Step 2: Run OpenMM MD ──────────────────────────────────────────────────────
def run_openmm(cand_id, md_dir):
    stats_file = md_dir / f"{cand_id}_md_stats.json"
    traj_file  = md_dir / f"{cand_id}_prod.dcd"

    if stats_file.exists() and traj_file.exists():
        print(f"  [skip] {cand_id} — MD complete (stats found)")
        return json.loads(stats_file.read_text())

    print(f"  {cand_id}: running OpenMM MD (100 ns, CUDA)...")
    out = run(
        f"{MD_SUITE_PY} {OPENMM_RUN} {cand_id} {md_dir}",
        cwd=md_dir,
        log_path=md_dir / f"{cand_id}_openmm_stdout.log"
    )
    if not stats_file.exists():
        raise RuntimeError(f"OpenMM MD failed for {cand_id}")
    return json.loads(stats_file.read_text())


# ── Step 3: cpptraj analysis ───────────────────────────────────────────────────
def run_cpptraj_analysis(cand_id, md_dir):
    """Compute RMSD, RMSF, Rg, ligand RMSD using cpptraj."""
    prmtop   = md_dir / "solvated.prmtop"
    traj_dcd = md_dir / f"{cand_id}_prod.dcd"
    out_dir  = md_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    cache = out_dir / "analysis_done.json"
    if cache.exists():
        print(f"  [skip] {cand_id} cpptraj analysis")
        return json.loads(cache.read_text())

    # Get total residue count and ligand residue number
    # Receptor = 287 residues, ligand = residue 288 (1-based AMBER)
    # Solvent starts at residue 289+
    N_REC = 287
    N_LIG_RES = N_REC + 1  # 288

    cpptraj_in = textwrap.dedent(f"""\
        parm {prmtop}
        trajin {traj_dcd}

        # Reference = first frame
        reference {traj_dcd} [ref]

        # Strip solvent for analysis
        strip :WAT,Na+,Cl-

        # Backbone RMSD (Cα, receptor residues)
        rmsd backbone @CA ref [ref] @CA mass nofit out {out_dir}/backbone_rmsd.dat

        # Ligand RMSD
        rmsd ligand_rmsd :{N_LIG_RES} ref [ref] :{N_LIG_RES} mass nofit out {out_dir}/ligand_rmsd.dat

        # RMSF per residue (Cα)
        atomicfluct rmsf @CA byres out {out_dir}/rmsf_ca.dat

        # Radius of gyration
        radgyr Rg :1-{N_LIG_RES} out {out_dir}/rg.dat mass

        # H-bond analysis (binding site: key residues ± ligand)
        hbond hb :46,92,93,94,95,96,97,98,99,167,{N_LIG_RES} solventdonor :WAT@O solventacceptor :WAT@O out {out_dir}/hbonds.dat avgout {out_dir}/hbonds_avg.dat

        run
        quit
    """)
    (out_dir / "cpptraj.in").write_text(cpptraj_in)

    print(f"  {cand_id}: cpptraj analysis...")
    run(f"{CPPTRAJ} -i {out_dir / 'cpptraj.in'}",
        log_path=out_dir / "cpptraj.log", env=AMBER_ENV)

    # Parse summary stats
    results = {"cand_id": cand_id}

    for fname, key in [("backbone_rmsd.dat", "backbone_rmsd"),
                        ("ligand_rmsd.dat",   "ligand_rmsd"),
                        ("rg.dat",            "rg")]:
        fpath = out_dir / fname
        if fpath.exists():
            data = np.loadtxt(fpath, comments="#")
            if data.ndim == 2 and data.shape[1] >= 2:
                vals = data[:, 1]
                results[f"{key}_mean"] = float(vals.mean())
                results[f"{key}_std"]  = float(vals.std())
                results[f"{key}_max"]  = float(vals.max())

    cache.write_text(json.dumps(results, indent=2))
    return results


# ── Figures ────────────────────────────────────────────────────────────────────
def generate_figures(all_analysis):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.linewidth": 1.2, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 300,
    })
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Fig 1 — Backbone RMSD over time (all 3 leads)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (cand_id, md_dir) in enumerate(all_analysis):
        rmsd_file = md_dir / "analysis" / "backbone_rmsd.dat"
        if rmsd_file.exists():
            data = np.loadtxt(rmsd_file, comments="#")
            if data.ndim == 2:
                time_ns = data[:, 0] * 0.002 / 1000  # steps × 2fs → ns
                rmsd_a  = data[:, 1]
                ax.plot(time_ns, rmsd_a, lw=0.8, color=colors[i],
                        alpha=0.9, label=cand_id)
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("Backbone RMSD (Å)", fontsize=12)
    ax.set_title("Backbone Cα RMSD — RET Kinase Complex", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(2.0, color="gray", ls="--", lw=0.8, label="2 Å threshold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "09_01_backbone_rmsd.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: 09_01_backbone_rmsd.png")

    # Fig 2 — Ligand RMSD over time
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (cand_id, md_dir) in enumerate(all_analysis):
        rmsd_file = md_dir / "analysis" / "ligand_rmsd.dat"
        if rmsd_file.exists():
            data = np.loadtxt(rmsd_file, comments="#")
            if data.ndim == 2:
                time_ns = data[:, 0] * 0.002 / 1000
                ax.plot(time_ns, data[:, 1], lw=0.8, color=colors[i],
                        alpha=0.9, label=cand_id)
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("Ligand RMSD (Å)", fontsize=12)
    ax.set_title("Ligand RMSD from Initial Docked Pose", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(2.0, color="gray", ls="--", lw=0.8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "09_02_ligand_rmsd.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: 09_02_ligand_rmsd.png")

    # Fig 3 — RMSF per residue (binding site region)
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, (cand_id, md_dir) in enumerate(all_analysis):
        rmsf_file = md_dir / "analysis" / "rmsf_ca.dat"
        if rmsf_file.exists():
            data = np.loadtxt(rmsf_file, comments="#")
            if data.ndim == 2:
                resnums = data[:, 0].astype(int)
                rmsf    = data[:, 1]
                # Focus on binding site region (res 40-170)
                mask = (resnums >= 40) & (resnums <= 175)
                ax.plot(resnums[mask], rmsf[mask], lw=1.2, color=colors[i],
                        alpha=0.85, label=cand_id)

    # Mark key residues
    for rnum, rname in KEY_RESIDUES.items():
        ax.axvline(rnum, color="red", ls="--", lw=0.5, alpha=0.5)
        ax.text(rnum, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1.0,
                rname, rotation=90, fontsize=6, color="red", ha="center")

    ax.set_xlabel("Residue (AMBER numbering)", fontsize=11)
    ax.set_ylabel("RMSF (Å)", fontsize=11)
    ax.set_title("Per-Residue RMSF — Binding Site Region (res 40–175)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "09_03_rmsf_binding_site.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: 09_03_rmsf_binding_site.png")

    # Fig 4 — Radius of gyration
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (cand_id, md_dir) in enumerate(all_analysis):
        rg_file = md_dir / "analysis" / "rg.dat"
        if rg_file.exists():
            data = np.loadtxt(rg_file, comments="#")
            if data.ndim == 2:
                time_ns = data[:, 0] * 0.002 / 1000
                ax.plot(time_ns, data[:, 1], lw=0.8, color=colors[i],
                        alpha=0.9, label=cand_id)
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("Radius of Gyration (Å)", fontsize=12)
    ax.set_title("Complex Radius of Gyration — Structural Stability", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "09_04_rg.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: 09_04_rg.png")

    # Fig 5 — Summary statistics bar chart
    stats_rows = []
    for cand_id, md_dir in all_analysis:
        cache = md_dir / "analysis" / "analysis_done.json"
        if cache.exists():
            d = json.loads(cache.read_text())
            stats_rows.append({
                "CandID":          cand_id,
                "Backbone RMSD":   d.get("backbone_rmsd_mean", np.nan),
                "Backbone RMSD σ": d.get("backbone_rmsd_std", np.nan),
                "Ligand RMSD":     d.get("ligand_rmsd_mean", np.nan),
                "Ligand RMSD σ":   d.get("ligand_rmsd_std", np.nan),
                "Rg":              d.get("rg_mean", np.nan),
            })

    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for ax, (col, errcol, title, unit_str) in zip(axes, [
            ("Backbone RMSD", "Backbone RMSD σ", "Mean Backbone RMSD", "Å"),
            ("Ligand RMSD",   "Ligand RMSD σ",   "Mean Ligand RMSD",   "Å"),
            ("Rg",            None,               "Mean Radius of Gyration", "Å"),
        ]):
            vals = stats_df[col].values
            errs = stats_df[errcol].values if errcol else np.zeros(len(vals))
            ax.bar(stats_df["CandID"], vals, yerr=errs,
                   color=colors[:len(vals)], edgecolor="k", lw=0.5,
                   capsize=4, error_kw={"lw": 1.2})
            ax.set_ylabel(f"{title} ({unit_str})", fontsize=10)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.tick_params(axis="x", rotation=15)

        plt.suptitle("MD Simulation Summary — Top 3 RET Leads",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "09_05_md_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved: 09_05_md_summary.png")

        stats_df.to_csv(RESULTS_DIR / "09_md_summary.csv", index=False)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Step 09: MD Simulation — Top 3 RET Kinase Leads")
    print(f"Candidates: {', '.join(TOP3)}")
    print("Protocol: ff14SB + GAFF2 + TIP3P | 100 ns | OpenMM CUDA")
    print("=" * 65)

    all_analysis = []
    failed = []

    for cand_id in TOP3:
        md_dir = MD_BASE / cand_id
        md_dir.mkdir(exist_ok=True)

        try:
            # 1. Solvate
            print(f"\n{'─'*50}")
            print(f"{cand_id}")
            print(f"{'─'*50}")
            print("[1/3] Solvation...")
            solvate(cand_id, md_dir)

            # 2. MD
            print("[2/3] OpenMM MD (100 ns)...")
            md_stats = run_openmm(cand_id, md_dir)
            print(f"  Wall time: {md_stats.get('wall_time_h', '?'):.1f} h")

            # 3. Analysis
            print("[3/3] cpptraj analysis...")
            analysis = run_cpptraj_analysis(cand_id, md_dir)
            bb_rmsd = analysis.get("backbone_rmsd_mean", "?")
            lig_rmsd = analysis.get("ligand_rmsd_mean", "?")
            print(f"  Backbone RMSD mean: {bb_rmsd:.2f} Å")
            print(f"  Ligand RMSD mean:   {lig_rmsd:.2f} Å")

            all_analysis.append((cand_id, md_dir))

        except Exception as exc:
            print(f"  ✗ {cand_id}: {exc}")
            failed.append({"CandID": cand_id, "error": str(exc)[:300]})

    if all_analysis:
        print("\nGenerating figures...")
        generate_figures(all_analysis)

    if failed:
        pd.DataFrame(failed).to_csv(RESULTS_DIR / "09_md_failed.csv", index=False)

    print("\n" + "=" * 65)
    print("Step 09 complete.")
    print(f"  Completed: {len(all_analysis)}/3")
    print(f"  5 figures → figures/09_md/")
    print("  Next → Step 10: Lead compound figures + pipeline workflow")
    print("=" * 65)


if __name__ == "__main__":
    main()
