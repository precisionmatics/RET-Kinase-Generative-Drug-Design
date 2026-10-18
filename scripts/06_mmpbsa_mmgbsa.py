#!/usr/bin/env python3
"""
Step 06: MM-PBSA / MM-GBSA — Top 20 RET Kinase Docked Candidates
AmberTools 22 (conda env: unomd)

Pitfall guards:
  - mk_export.py extracts DOCKED coordinates from DLG (not pre-docking SDF)
  - sander minimization mandatory before MMPBSA
  - parmed Python API combines rec+lig topologies (handles chain gap at 828→842)
  - antechamber: -fi sdf (avoids CONECT bond duplication)
  - parmchk2: -a Y (auto-fills missing GAFF2 torsion parameters)
"""

import os
import re
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import subprocess
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path("/home/stalin/Desktop/RET_Generative")
MMPBSA_DIR   = ROOT / "mmpbsa"
REC_DIR      = MMPBSA_DIR / "receptor"
LIG_DIR      = MMPBSA_DIR / "ligands"
FIG_DIR      = ROOT / "figures" / "06_mmpbsa"
RESULTS_DIR  = ROOT / "results"
DATA_DIR     = ROOT / "data" / "processed"

DOCK_CSV     = DATA_DIR / "RET_docking_results.csv"
BOLTZ_CSV    = DATA_DIR / "RET_boltz2_affinity.csv"
RECEPTOR_PDB = ROOT / "docking" / "receptor" / "6NEC_chainA.pdb"
DLG_DIR      = ROOT / "docking" / "results"
MK_EXPORT    = Path("/home/stalin/miniforge3/bin/mk_export.py")

# AmberTools in unomd conda env
UNOMD_BIN    = Path("/home/stalin/miniforge3/envs/unomd/bin")
UNOMD_PY     = UNOMD_BIN / "python"
ANTECHAMBER  = UNOMD_BIN / "antechamber"
PARMCHK2     = UNOMD_BIN / "parmchk2"
TLEAP        = UNOMD_BIN / "tleap"
SANDER       = UNOMD_BIN / "sander"
MMPBSA_PY    = UNOMD_BIN / "MMPBSA.py"

# AMBERHOME must be set for AmberTools to locate dat/ files
AMBER_ENV = {**os.environ, "AMBERHOME": str(UNOMD_BIN.parent)}

# 287 receptor residues (713–1012, gap 829–841 missing)
# tleap renumbers 1–287; ligand = residue 288
N_REC       = 287
TOP_N       = 20

for d in [REC_DIR, LIG_DIR, FIG_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Utilities ──────────────────────────────────────────────────────────────────
def run(cmd, cwd=None, log_path=None, check=True, use_amber_env=False):
    env = AMBER_ENV if use_amber_env else None
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if log_path:
        Path(log_path).write_text(result.stdout)
    if check and result.returncode != 0:
        tail = result.stdout[-3000:] if result.stdout else "(no output)"
        raise RuntimeError(f"Command failed (rc={result.returncode}):\n{cmd}\n\n{tail}")
    return result.stdout


def parmed_combine(rec_prmtop, rec_inpcrd, lig_prmtop, lig_inpcrd,
                   out_prmtop, out_inpcrd, log_path):
    """Use parmed Python API (unomd env) to combine rec+lig topologies and coords."""
    script = textwrap.dedent(f"""\
        import parmed as pmd
        rec = pmd.load_file('{rec_prmtop}', '{rec_inpcrd}')
        lig = pmd.load_file('{lig_prmtop}', '{lig_inpcrd}')
        cplx = rec + lig
        cplx.save('{out_prmtop}', overwrite=True)
        cplx.save('{out_inpcrd}', overwrite=True)
        print(f"Combined: {{len(cplx.atoms)}} atoms, {{len(cplx.residues)}} residues")
    """)
    py_script = Path(log_path).parent / "parmed_combine.py"
    py_script.write_text(script)
    run(f"{UNOMD_PY} {py_script}", log_path=log_path)


def parse_mmpbsa_dat(dat_file):
    """
    Parse MMPBSA.py FINAL_RESULTS.dat to extract DELTA TOTAL for PB and GB.
    AmberTools format: "DELTA TOTAL   -34.43   0.00   0.00" (no '=' sign).
    Order in file: GENERALIZED BORN section first, then POISSON BOLTZMANN.
    Returns (pb_val, gb_val) — both kcal/mol, None if section absent.
    """
    text = Path(dat_file).read_text()
    pb_val = gb_val = None

    # Split on method-section headers; order in file: GB then PB
    sections = re.split(r'(POISSON BOLTZMANN|GENERALIZED BORN)', text, flags=re.IGNORECASE)
    mode = None
    for seg in sections:
        upper = seg.upper().strip()
        if "POISSON BOLTZMANN" in upper:
            mode = "pb"
        elif "GENERALIZED BORN" in upper:
            mode = "gb"
        else:
            # Format: "DELTA TOTAL   <value>   ..."  — no equals sign
            m = re.search(r'DELTA TOTAL\s+([-+]?\d+\.?\d*)', seg)
            if m:
                val = float(m.group(1))
                if mode == "pb" and pb_val is None:
                    pb_val = val
                elif mode == "gb" and gb_val is None:
                    gb_val = val
    return pb_val, gb_val


# ── Receptor topology (built once) ────────────────────────────────────────────
def build_receptor_topology():
    rec_prmtop = REC_DIR / "receptor.prmtop"
    rec_inpcrd  = REC_DIR / "receptor.inpcrd"

    if rec_prmtop.exists() and rec_inpcrd.exists():
        print("  [skip] receptor.prmtop already exists")
        return rec_prmtop, rec_inpcrd

    tleap_in = textwrap.dedent(f"""\
        source leaprc.protein.ff14SB
        source leaprc.water.tip3p
        rec = loadpdb {RECEPTOR_PDB}
        check rec
        saveamberparm rec {rec_prmtop} {rec_inpcrd}
        quit
    """)
    inp = REC_DIR / "tleap_receptor.in"
    inp.write_text(tleap_in)

    print("  Running tleap for receptor topology...")
    out = run(f"{TLEAP} -f {inp}", cwd=REC_DIR,
              log_path=REC_DIR / "tleap_receptor.log", check=False, use_amber_env=True)

    if not rec_prmtop.exists():
        raise RuntimeError(f"tleap failed to build receptor topology.\n{out[-2000:]}")

    print(f"  receptor.prmtop: {rec_prmtop.stat().st_size / 1024:.0f} KB")
    return rec_prmtop, rec_inpcrd


# ── Single-ligand pipeline ─────────────────────────────────────────────────────
def process_one_ligand(cand_id, dlg_file, rec_prmtop, rec_inpcrd):
    """
    Full MM-PBSA/MM-GBSA pipeline for one docked candidate.
    Returns dict with DELTA_TOTAL_pb and DELTA_TOTAL_gb.
    """
    wdir = LIG_DIR / cand_id
    wdir.mkdir(parents=True, exist_ok=True)

    cache = wdir / "result.json"
    if cache.exists():
        print(f"  [skip] {cand_id} — cached")
        return json.loads(cache.read_text())

    dlg_path   = DLG_DIR / dlg_file
    docked_sdf = wdir / "docked_lig.sdf"
    lig_mol2   = wdir / "lig.mol2"
    lig_frcmod = wdir / "lig.frcmod"
    lig_prmtop = wdir / "lig.prmtop"
    lig_inpcrd = wdir / "lig.inpcrd"
    cplx_prmtop = wdir / "complex.prmtop"
    cplx_inpcrd = wdir / "complex.inpcrd"
    min_rst     = wdir / "min.rst"
    mmpbsa_out  = wdir / "FINAL_RESULTS_MMPBSA.dat"

    # 2a. Extract docked coordinates from DLG (never use pre-docking SDF)
    print(f"  {cand_id}: extract docked pose...")
    run(f"python {MK_EXPORT} {dlg_path} -s {docked_sdf}",
        log_path=wdir / "mk_export.log")

    # 2b. antechamber: GAFF2 (-fi sdf, never pdb)
    print(f"  {cand_id}: antechamber GAFF2...")
    run(
        f"{ANTECHAMBER} -i {docked_sdf} -fi sdf "
        f"-o {lig_mol2} -fo mol2 "
        f"-c bcc -at gaff2 -nc 0 -rn LIG -dr n -pf y",
        cwd=wdir, log_path=wdir / "antechamber.log", use_amber_env=True
    )

    # 2c. parmchk2 -a Y (fills missing torsions for heterocyclic ring systems)
    print(f"  {cand_id}: parmchk2...")
    run(
        f"{PARMCHK2} -i {lig_mol2} -f mol2 -o {lig_frcmod} -a Y",
        cwd=wdir, log_path=wdir / "parmchk2.log", use_amber_env=True
    )

    # 2d. tleap: ligand-only topology
    print(f"  {cand_id}: tleap ligand topology...")
    tleap_lig = textwrap.dedent(f"""\
        source leaprc.gaff2
        LIG = loadmol2 {lig_mol2}
        loadamberparams {lig_frcmod}
        check LIG
        saveamberparm LIG {lig_prmtop} {lig_inpcrd}
        quit
    """)
    (wdir / "tleap_lig.in").write_text(tleap_lig)
    run(f"{TLEAP} -f {wdir / 'tleap_lig.in'}",
        cwd=wdir, log_path=wdir / "tleap_lig.log", use_amber_env=True)

    if not lig_prmtop.exists():
        raise RuntimeError(f"tleap failed to create {lig_prmtop}")

    # 2e. parmed (Python API): combine rec + lig topologies with docked coordinates
    print(f"  {cand_id}: parmed combine rec+lig...")
    parmed_combine(rec_prmtop, rec_inpcrd,
                   lig_prmtop, lig_inpcrd,
                   cplx_prmtop, cplx_inpcrd,
                   wdir / "parmed_combine.log")

    if not cplx_prmtop.exists() or not cplx_inpcrd.exists():
        raise RuntimeError(f"parmed failed to create complex topology for {cand_id}")

    # 2f. sander: gas-phase minimization with receptor frozen
    print(f"  {cand_id}: sander minimization...")
    sander_in = textwrap.dedent(f"""\
        Minimize complex (receptor frozen, ligand free)
         &cntrl
          imin=1, maxcyc=2000, ncyc=1000,
          ntb=0, igb=0, cut=999.0,
          ntr=1,
          restraintmask=':{1}-{N_REC}',
          restraint_wt=10.0,
         /
    """)
    (wdir / "min.in").write_text(sander_in)
    run(
        f"{SANDER} -O "
        f"-i {wdir / 'min.in'} "
        f"-o {wdir / 'min.out'} "
        f"-p {cplx_prmtop} "
        f"-c {cplx_inpcrd} "
        f"-r {min_rst} "
        f"-ref {cplx_inpcrd}",
        cwd=wdir, log_path=wdir / "sander.log", use_amber_env=True
    )

    # 2g. MMPBSA.py: both PB and GB
    print(f"  {cand_id}: MMPBSA.py (PB + GB)...")
    mmpbsa_in = textwrap.dedent(f"""\
        MM-PBSA and MM-GBSA for {cand_id}
        &general
          startframe=1, endframe=1, interval=1,
          verbose=2,
        /
        &gb
          igb=5, saltcon=0.150,
        /
        &pb
          istrng=0.150, inp=1,
        /
    """)
    (wdir / "mmpbsa.in").write_text(mmpbsa_in)

    run(
        f"{UNOMD_PY} {MMPBSA_PY} -O "
        f"-i {wdir / 'mmpbsa.in'} "
        f"-o {mmpbsa_out} "
        f"-cp {cplx_prmtop} "
        f"-rp {rec_prmtop} "
        f"-lp {lig_prmtop} "
        f"-y {min_rst}",
        cwd=wdir, log_path=wdir / "mmpbsa.log", use_amber_env=True
    )

    # 2h. Parse results
    pb_val, gb_val = parse_mmpbsa_dat(mmpbsa_out)
    result = {"CandID": cand_id, "DELTA_TOTAL_pb": pb_val, "DELTA_TOTAL_gb": gb_val}
    cache.write_text(json.dumps(result, indent=2))
    return result


# ── Figures ────────────────────────────────────────────────────────────────────
def generate_figures(df):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.linewidth": 1.2, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 300,
    })
    pal = plt.cm.Set2.colors

    # Fig 1 — MM-PBSA bar
    fig, ax = plt.subplots(figsize=(12, 6))
    vals = df["DELTA_TOTAL_pb"].values
    colors = [pal[0] if v < -30 else pal[1] if v < -20 else pal[3] for v in vals]
    ax.barh(df["CandID"][::-1], vals[::-1], color=colors[::-1], edgecolor="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("MM-PBSA ΔG$_{bind}$ (kcal/mol)", fontsize=12)
    ax.set_title("MM-PBSA Binding Free Energy — Top 20 RET Candidates",
                 fontsize=13, fontweight="bold")
    ax.invert_xaxis()
    patches = [mpatches.Patch(color=pal[0], label="< −30"),
               mpatches.Patch(color=pal[1], label="−30 to −20"),
               mpatches.Patch(color=pal[3], label="≥ −20")]
    ax.legend(handles=patches, title="ΔG (kcal/mol)", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_01_mmpbsa_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 2 — MM-GBSA bar
    fig, ax = plt.subplots(figsize=(12, 6))
    vals_gb = df["DELTA_TOTAL_gb"].values
    colors_gb = [pal[2] if v < -30 else pal[1] if v < -20 else pal[3] for v in vals_gb]
    ax.barh(df["CandID"][::-1], vals_gb[::-1], color=colors_gb[::-1], edgecolor="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("MM-GBSA ΔG$_{bind}$ (kcal/mol)", fontsize=12)
    ax.set_title("MM-GBSA Binding Free Energy — Top 20 RET Candidates",
                 fontsize=13, fontweight="bold")
    ax.invert_xaxis()
    patches_gb = [mpatches.Patch(color=pal[2], label="< −30"),
                  mpatches.Patch(color=pal[1], label="−30 to −20"),
                  mpatches.Patch(color=pal[3], label="≥ −20")]
    ax.legend(handles=patches_gb, title="ΔG (kcal/mol)", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_02_mmgbsa_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 3 — AutoDock vs MM-PBSA & MM-GBSA scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title, cmap_ in zip(
        axes,
        ["DELTA_TOTAL_pb", "DELTA_TOTAL_gb"],
        ["AutoDock-GPU vs MM-PBSA", "AutoDock-GPU vs MM-GBSA"],
        ["viridis", "plasma"]
    ):
        sc = ax.scatter(df["DockingScore"], df[col],
                        c=df["QED"], cmap=cmap_, s=75, edgecolors="k", lw=0.5, zorder=3)
        for _, r in df.iterrows():
            ax.annotate(r["CandID"], (r["DockingScore"], r[col]),
                        xytext=(4, 4), textcoords="offset points", fontsize=6, color="gray")
        plt.colorbar(sc, ax=ax, label="QED")
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("AutoDock-GPU ΔG (kcal/mol)", fontsize=11)
        ax.set_ylabel(f"{'MM-PBSA' if 'pb' in col else 'MM-GBSA'} ΔG$_{{bind}}$ (kcal/mol)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_03_docking_vs_mmpbsa_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 4 — PB vs GB correlation
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df["DELTA_TOTAL_pb"], df["DELTA_TOTAL_gb"],
                    c=range(len(df)), cmap="tab20", s=80, edgecolors="k", lw=0.5, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["CandID"], (r["DELTA_TOTAL_pb"], r["DELTA_TOTAL_gb"]),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)
    mn = min(df["DELTA_TOTAL_pb"].min(), df["DELTA_TOTAL_gb"].min()) - 5
    mx = max(df["DELTA_TOTAL_pb"].max(), df["DELTA_TOTAL_gb"].max()) + 5
    ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, label="y = x")
    corr = df["DELTA_TOTAL_pb"].corr(df["DELTA_TOTAL_gb"])
    ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=10)
    ax.set_xlabel("MM-PBSA ΔG$_{bind}$ (kcal/mol)", fontsize=11)
    ax.set_ylabel("MM-GBSA ΔG$_{bind}$ (kcal/mol)", fontsize=11)
    ax.set_title("MM-PBSA vs MM-GBSA Correlation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_04_pb_vs_gb_corr.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 5 — Four-method consensus heatmap
    boltz = pd.read_csv(BOLTZ_CSV)[["CandID", "pIC50_boltz"]]
    merged = df.merge(boltz, on="CandID", how="left")
    merged["pIC50_boltz"] = merged["pIC50_boltz"].fillna(merged["pIC50_boltz"].min())

    def norm_rank(s, ascending=True):
        r = s.rank(ascending=ascending, method="min")
        return (r - 1) / max(len(s) - 1, 1)

    heat = np.column_stack([
        norm_rank(merged["DockingScore"], ascending=True),
        norm_rank(merged["DELTA_TOTAL_pb"], ascending=True),
        norm_rank(merged["DELTA_TOTAL_gb"], ascending=True),
        norm_rank(merged["pIC50_boltz"], ascending=False),
    ])

    fig, ax = plt.subplots(figsize=(7, 10))
    im = ax.imshow(heat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["AutoDock-GPU", "MM-PBSA", "MM-GBSA", "Boltz-2"],
                       fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(merged)))
    ax.set_yticklabels(merged["CandID"].tolist(), fontsize=9)
    plt.colorbar(im, ax=ax, label="Normalized rank (0=best, 1=worst)")
    ax.set_title("Four-Method Consensus Ranking\nTop 20 RET Candidates",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_05_consensus_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  5 figures saved → figures/06_mmpbsa/")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Step 06: MM-PBSA / MM-GBSA — Top 20 RET Kinase Candidates")
    print("=" * 65)

    dock_df = pd.read_csv(DOCK_CSV)
    top20   = dock_df[dock_df["DockRank"] <= TOP_N].copy().reset_index(drop=True)

    print(f"\nTop {TOP_N} candidates:")
    print(top20[["DockRank", "CandID", "DockingScore", "DLGFile"]].to_string(index=False))

    print("\n[1/3] Building receptor topology (ff14SB)...")
    rec_prmtop, rec_inpcrd = build_receptor_topology()

    print(f"\n[2/3] MM-PBSA/MM-GBSA for {TOP_N} candidates...")
    results = []
    failed  = []

    for _, row in top20.iterrows():
        cand_id  = row["CandID"]
        dlg_file = row["DLGFile"]
        try:
            res = process_one_ligand(cand_id, dlg_file, rec_prmtop, rec_inpcrd)
            res.update({
                "DockRank":     int(row["DockRank"]),
                "DockingScore": float(row["DockingScore"]),
                "QED":          float(row["QED"]),
                "MW":           float(row["MW"]),
                "LogP":         float(row["LogP"]),
                "SMILES":       row["SMILES"],
            })
            results.append(res)
            print(f"  ✓ {cand_id}: PB={res['DELTA_TOTAL_pb']:.2f}  GB={res['DELTA_TOTAL_gb']:.2f}")
        except Exception as exc:
            print(f"  ✗ {cand_id}: {exc}")
            failed.append({"CandID": cand_id, "error": str(exc)[:300]})

    if not results:
        print("\nERROR: All ligands failed. Check logs in mmpbsa/ligands/<ID>/")
        return

    res_df = pd.DataFrame(results)

    # Consensus score (average of PB + GB + AutoDock ranks)
    res_df["rank_pb"]   = res_df["DELTA_TOTAL_pb"].rank(ascending=True)
    res_df["rank_gb"]   = res_df["DELTA_TOTAL_gb"].rank(ascending=True)
    res_df["rank_dock"] = res_df["DockingScore"].rank(ascending=True)
    res_df["consensus"] = (res_df["rank_pb"] + res_df["rank_gb"] + res_df["rank_dock"]) / 3
    res_df.sort_values("consensus", inplace=True, ignore_index=True)

    res_df.to_csv(DATA_DIR / "RET_mmpbsa_results.csv", index=False)
    res_df.to_csv(RESULTS_DIR / "06_mmpbsa_results.csv", index=False)
    print(f"\nSaved → data/processed/RET_mmpbsa_results.csv ({len(res_df)} rows)")

    if failed:
        pd.DataFrame(failed).to_csv(RESULTS_DIR / "06_mmpbsa_failed.csv", index=False)
        print(f"Failed: {len(failed)} → results/06_mmpbsa_failed.csv")

    print("\n" + "─" * 60)
    print(f"{'#':<4} {'CandID':<10} {'ΔG_PB':>10} {'ΔG_GB':>10} {'AutoDock':>10} {'Consensus':>10}")
    print("─" * 60)
    for i, r in res_df.head(10).iterrows():
        print(f"{i+1:<4} {r['CandID']:<10} {r['DELTA_TOTAL_pb']:>10.2f} "
              f"{r['DELTA_TOTAL_gb']:>10.2f} {r['DockingScore']:>10.2f} {r['consensus']:>10.2f}")

    print("\n[3/3] Generating figures...")
    generate_figures(res_df.copy())

    best_pb  = res_df.sort_values("DELTA_TOTAL_pb").iloc[0]
    best_gb  = res_df.sort_values("DELTA_TOTAL_gb").iloc[0]
    top5     = res_df.head(5)["CandID"].tolist()

    stats = {
        "n_processed": len(results), "n_failed": len(failed),
        "best_pb_cand": best_pb["CandID"], "best_pb_val": float(best_pb["DELTA_TOTAL_pb"]),
        "best_gb_cand": best_gb["CandID"], "best_gb_val": float(best_gb["DELTA_TOTAL_gb"]),
        "mean_pb": float(res_df["DELTA_TOTAL_pb"].mean()),
        "mean_gb": float(res_df["DELTA_TOTAL_gb"].mean()),
        "top5_consensus": top5,
    }
    (RESULTS_DIR / "06_mmpbsa_stats.json").write_text(json.dumps(stats, indent=2))

    print("\n" + "=" * 65)
    print("Step 06 complete.")
    print(f"  Best MM-PBSA : {stats['best_pb_cand']} ({stats['best_pb_val']:.2f} kcal/mol)")
    print(f"  Best MM-GBSA : {stats['best_gb_cand']} ({stats['best_gb_val']:.2f} kcal/mol)")
    print(f"  Top-5 consensus : {top5}")
    print("  Next → Step 07: GNINA CNN docking (top 20)")
    print("=" * 65)


if __name__ == "__main__":
    main()
