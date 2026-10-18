#!/usr/bin/env python3
"""
Step 08: Per-Residue MM-GBSA Decomposition — Top 5 RET Kinase Leads
Reuses complex.prmtop, receptor.prmtop, lig.prmtop, min.rst from Step 06.
Runs MMPBSA.py with &decomp (idecomp=2, GB, per-residue) for each lead.

Top 5 from 6-method consensus: RET_091, RET_098, RET_092, RET_074, RET_099
Key RET hinge residues (AMBER numbering, 1-based from GLU713=1):
  LYS758=46, VAL804=92, GLU805=93, TYR806=94, ALA807=95, ASP892=167
"""

import re
import json
import textwrap
import subprocess
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path("/home/stalin/Desktop/RET_Generative")
MMPBSA_DIR  = ROOT / "mmpbsa"
REC_DIR     = MMPBSA_DIR / "receptor"
LIG_DIR     = MMPBSA_DIR / "ligands"
FIG_DIR     = ROOT / "figures" / "08_decomp"
RESULTS_DIR = ROOT / "results"
DATA_DIR    = ROOT / "data" / "processed"

UNOMD_BIN  = Path("/home/stalin/miniforge3/envs/unomd/bin")
UNOMD_PY   = UNOMD_BIN / "python"
MMPBSA_PY  = UNOMD_BIN / "MMPBSA.py"
AMBER_ENV  = {**os.environ, "AMBERHOME": str(UNOMD_BIN.parent)}

TOP5 = ["RET_091", "RET_098", "RET_092", "RET_074", "RET_099"]

# Key RET binding site residues (AMBER 1-based numbering from GLU713=1)
KEY_RESIDUES = {
    46:  "LYS758",
    92:  "VAL804",
    93:  "GLU805",
    94:  "TYR806",
    95:  "ALA807",
    96:  "LYS808",
    97:  "TYR809",
    98:  "GLY810",
    99:  "SER811",
    167: "ASP892",
}

for d in [FIG_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Run MMPBSA decomp ─────────────────────────────────────────────────────────
def run(cmd, log_path=None, check=True, cwd=None):
    result = subprocess.run(
        cmd, shell=True, env=AMBER_ENV, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if log_path:
        Path(log_path).write_text(result.stdout)
    if check and result.returncode != 0:
        tail = result.stdout[-3000:] if result.stdout else "(no output)"
        raise RuntimeError(f"Command failed (rc={result.returncode}):\n{cmd}\n\n{tail}")
    return result.stdout


def run_decomp(cand_id):
    wdir       = LIG_DIR / cand_id
    decomp_dir = wdir / "decomp"
    decomp_dir.mkdir(exist_ok=True)

    decomp_out = decomp_dir / "DECOMP_RESULTS.dat"
    decomp_dat = decomp_dir / "PER_RESIDUE_DECOMP.dat"
    cache      = decomp_dir / "decomp_result.json"

    if cache.exists():
        print(f"  [skip] {cand_id} decomp — cached")
        return json.loads(cache.read_text())

    cplx_prmtop = wdir / "complex.prmtop"
    rec_prmtop  = REC_DIR / "receptor.prmtop"
    lig_prmtop  = wdir / "lig.prmtop"
    min_rst     = wdir / "min.rst"

    # Fast-path: if PER_RESIDUE_DECOMP.dat already exists, just re-parse
    if decomp_dat.exists() and decomp_dat.stat().st_size > 1000:
        print(f"  {cand_id}: re-parsing existing PER_RESIDUE_DECOMP.dat...")
        residues = parse_decomp(decomp_dat)
        if not residues:
            residues = parse_decomp_v2(decomp_dat)
        result = {"cand_id": cand_id, "residues": residues}
        cache.write_text(json.dumps(result, indent=2))
        return result

    for f in [cplx_prmtop, rec_prmtop, lig_prmtop, min_rst]:
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")

    decomp_in = textwrap.dedent(f"""\
        Per-residue MM-GBSA decomposition for {cand_id}
        &general
          startframe=1, endframe=1, interval=1,
          verbose=2,
        /
        &gb
          igb=5, saltcon=0.150,
        /
        &decomp
          idecomp=2,
          dec_verbose=3,
          print_res="1-288",
        /
    """)
    (decomp_dir / "decomp.in").write_text(decomp_in)

    print(f"  {cand_id}: running MMPBSA.py decomp...")
    run(
        f"{UNOMD_PY} {MMPBSA_PY} -O "
        f"-i {decomp_dir / 'decomp.in'} "
        f"-o {decomp_out} "
        f"-do {decomp_dat} "
        f"-cp {cplx_prmtop} "
        f"-rp {rec_prmtop} "
        f"-lp {lig_prmtop} "
        f"-y {min_rst}",
        cwd=decomp_dir,
        log_path=decomp_dir / "decomp.log"
    )

    residues = parse_decomp(decomp_dat)
    result = {"cand_id": cand_id, "residues": residues}
    cache.write_text(json.dumps(result, indent=2))
    return result


# ── Parse decomp output ───────────────────────────────────────────────────────
def parse_decomp(dat_file):
    """
    Parse MMPBSA.py per-residue decomp CSV output (AmberTools 22 format).
    File contains multiple sections; DELTAS: section has complex-receptor-ligand.
    CSV format per data row:
      Residue, Location, Internal(Avg,SD,SE), vdW(Avg,SD,SE), Elec(Avg,SD,SE),
      Polar(Avg,SD,SE), NonPolar(Avg,SD,SE), TOTAL(Avg,SD,SE)
    Column indices (0-based): 0=Residue, 1=Location, 2=Internal_Avg,
      5=vdW_Avg, 8=Elec_Avg, 11=Polar_Avg, 14=NonPolar_Avg, 17=TOTAL_Avg
    """
    text = Path(dat_file).read_text()
    residues = {}

    # Find DELTAS section
    delta_idx = text.find("DELTAS:")
    if delta_idx == -1:
        delta_idx = 0
    delta_text = text[delta_idx:]

    in_data = False
    for line in delta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip header lines
        if line.startswith("Residue,Location"):
            in_data = False  # reset: next non-header line starts data
            continue
        if line.startswith(",,Avg"):
            in_data = True  # data rows follow
            continue
        if not in_data:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 18:
            continue

        # parts[0] = "GLU   1", parts[1] = "R GLU   1"
        # Extract residue name and number from parts[0]
        res_field = parts[0].strip()
        m = re.match(r'([A-Z]+)\s+(\d+)', res_field)
        if not m:
            continue
        resname = m.group(1)
        resnum  = int(m.group(2))

        try:
            vdw      = float(parts[5])
            elec     = float(parts[8])
            polar    = float(parts[11])
            nonpolar = float(parts[14])
            total    = float(parts[17])
        except (ValueError, IndexError):
            continue

        residues[resnum] = {
            "resnum":   resnum,
            "resname":  resname,
            "total":    total,
            "vdw":      vdw,
            "elec":     elec,
            "polar":    polar,
            "nonpolar": nonpolar,
        }

    return residues


def parse_decomp_v2(dat_file):
    """Fallback: same CSV logic but without requiring DELTAS: header."""
    text = Path(dat_file).read_text()
    residues = {}
    in_data = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Residue,Location"):
            in_data = False
            continue
        if line.startswith(",,Avg"):
            in_data = True
            continue
        if not in_data or not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 18:
            continue
        m = re.match(r'([A-Z]+)\s+(\d+)', parts[0].strip())
        if not m:
            continue
        resname, resnum = m.group(1), int(m.group(2))
        try:
            residues[resnum] = {
                "resnum": resnum, "resname": resname,
                "total":    float(parts[17]),
                "vdw":      float(parts[5]),
                "elec":     float(parts[8]),
                "polar":    float(parts[11]),
                "nonpolar": float(parts[14]),
            }
        except (ValueError, IndexError):
            continue
    return residues


# ── Figures ───────────────────────────────────────────────────────────────────
def generate_figures(all_results):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.linewidth": 1.2, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 300,
    })
    pal = plt.cm.Set2.colors

    # Build a combined DataFrame: each row = one residue × one candidate
    rows = []
    for res in all_results:
        cand = res["cand_id"]
        for rnum, rdata in res["residues"].items():
            rows.append({
                "CandID":   cand,
                "resnum":   rnum,
                "resname":  rdata["resname"],
                "total":    rdata["total"],
                "vdw":      rdata["vdw"],
                "elec":     rdata["elec"],
                "polar":    rdata["polar"],
                "nonpolar": rdata["nonpolar"],
                "is_key":   rnum in KEY_RESIDUES,
            })
    df = pd.DataFrame(rows)

    if df.empty:
        print("  WARNING: no residue data parsed — check decomp output format")
        return

    # Fig 1 — Per-candidate bar chart (top 15 contributors each, stacked vdw/elec/polar)
    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 6),
                              sharey=False)
    if len(all_results) == 1:
        axes = [axes]

    for ax, res in zip(axes, all_results):
        cand = res["cand_id"]
        sub = df[df["CandID"] == cand].copy()
        if sub.empty:
            ax.set_title(f"{cand}\n(no data)")
            continue
        sub = sub.sort_values("total").head(15)  # most negative = best contributors
        labels = [f"{r['resname']}\n({r['resnum']})" for _, r in sub.iterrows()]
        colors = ["#d62728" if r["resnum"] in KEY_RESIDUES else "#1f77b4"
                  for _, r in sub.iterrows()]
        ax.barh(range(len(sub)), sub["total"].values, color=colors, edgecolor="k", lw=0.4)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("ΔG contribution\n(kcal/mol)", fontsize=9)
        ax.set_title(cand, fontsize=11, fontweight="bold")
        ax.invert_xaxis()

    # Legend
    patches = [mpatches.Patch(color="#d62728", label="Key hinge/binding residue"),
               mpatches.Patch(color="#1f77b4", label="Other residue")]
    fig.legend(handles=patches, loc="upper right", fontsize=9)
    fig.suptitle("Per-Residue MM-GBSA Decomposition — Top 5 RET Leads",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "08_01_per_residue_bars.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: 08_01_per_residue_bars.png")

    # Fig 2 — Heatmap: key residues × candidates
    # Get union of key residues present across all results
    key_resnums = sorted(KEY_RESIDUES.keys())
    heatmap_rows = []
    for rnum in key_resnums:
        row = {"resnum": rnum, "label": f"{KEY_RESIDUES[rnum]}\n(res {rnum})"}
        for res in all_results:
            cand = res["cand_id"]
            row[cand] = res["residues"].get(rnum, {}).get("total", 0.0)
        heatmap_rows.append(row)

    heat_df = pd.DataFrame(heatmap_rows).set_index("label")
    cand_cols = [r["cand_id"] for r in all_results]
    heat_vals = heat_df[cand_cols].values

    vmax = max(abs(heat_vals.min()), abs(heat_vals.max())) + 0.1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heat_vals, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(range(len(cand_cols)))
    ax.set_xticklabels(cand_cols, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(heat_df)))
    ax.set_yticklabels(heat_df.index.tolist(), fontsize=9)
    for i in range(heat_vals.shape[0]):
        for j in range(heat_vals.shape[1]):
            val = heat_vals[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(val) > vmax * 0.5 else "black")
    plt.colorbar(im, ax=ax, label="ΔG contribution (kcal/mol)")
    ax.set_title("Per-Residue MM-GBSA — Key RET Binding Site Residues",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "08_02_key_residues_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: 08_02_key_residues_heatmap.png")

    # Fig 3 — Stacked energy decomposition (vdw + elec + polar) for top 5 key residues
    # Average across all candidates
    avg_rows = []
    all_resnums = df["resnum"].unique()
    for rnum in all_resnums:
        sub = df[df["resnum"] == rnum]
        avg_rows.append({
            "resnum":  rnum,
            "resname": sub["resname"].iloc[0],
            "total":   sub["total"].mean(),
            "vdw":     sub["vdw"].mean(),
            "elec":    sub["elec"].mean(),
            "polar":   sub["polar"].mean(),
            "nonpolar":sub["nonpolar"].mean(),
            "is_key":  rnum in KEY_RESIDUES,
        })
    avg_df = pd.DataFrame(avg_rows).sort_values("total").head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(avg_df))
    labels = [f"{r['resname']}" for _, r in avg_df.iterrows()]
    w = 0.6
    ax.bar(x, avg_df["vdw"],     width=w, label="vdW",         color="#2196F3", edgecolor="k", lw=0.4)
    ax.bar(x, avg_df["elec"],    width=w, bottom=avg_df["vdw"], label="Electrostatic", color="#FF9800", edgecolor="k", lw=0.4)
    ax.bar(x, avg_df["polar"],   width=w,
           bottom=avg_df["vdw"] + avg_df["elec"],
           label="Polar Solv.", color="#9C27B0", edgecolor="k", lw=0.4)
    ax.bar(x, avg_df["nonpolar"],width=w,
           bottom=avg_df["vdw"] + avg_df["elec"] + avg_df["polar"],
           label="Nonpolar Solv.", color="#4CAF50", edgecolor="k", lw=0.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean ΔG contribution (kcal/mol)", fontsize=11)
    ax.set_title("Energy Decomposition by Component — Top 15 Residues (Mean, 5 Leads)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    # Mark key residues
    for i, (_, r) in enumerate(avg_df.iterrows()):
        if r["resnum"] in KEY_RESIDUES:
            ax.get_xticklabels()[i].set_color("#d62728")
            ax.get_xticklabels()[i].set_fontweight("bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "08_03_stacked_decomp.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: 08_03_stacked_decomp.png")

    # Fig 4 — Radar chart for top 5 key residue contributions per candidate
    key_present = [r for r in key_resnums if any(
        r in res["residues"] for res in all_results)]
    if len(key_present) >= 3:
        categories = [KEY_RESIDUES[r] for r in key_present]
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        colors_r = plt.cm.tab10.colors

        for i, res in enumerate(all_results):
            cand = res["cand_id"]
            vals = [abs(res["residues"].get(r, {}).get("total", 0.0)) for r in key_present]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", lw=2, color=colors_r[i], label=cand)
            ax.fill(angles, vals, alpha=0.1, color=colors_r[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_title("|ΔG contribution| at Key RET Binding Residues",
                     fontsize=12, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "08_04_radar_key_residues.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved: 08_04_radar_key_residues.png")

    # Fig 5 — Top contributor table figure
    top_contributors = (
        df.groupby(["resnum","resname"])["total"]
        .mean()
        .reset_index()
        .sort_values("total")
        .head(20)
    )
    top_contributors["is_key"] = top_contributors["resnum"].isin(KEY_RESIDUES)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors_t = ["#d62728" if k else "#1f77b4" for k in top_contributors["is_key"]]
    bars = ax.barh(
        top_contributors["resname"] + " (" + top_contributors["resnum"].astype(str) + ")",
        top_contributors["total"],
        color=colors_t, edgecolor="k", lw=0.4
    )
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Mean ΔG contribution (kcal/mol)", fontsize=11)
    ax.set_title("Top 20 Residue Contributors — Mean Across 5 Leads",
                 fontsize=12, fontweight="bold")
    ax.invert_xaxis()
    patches = [mpatches.Patch(color="#d62728", label="Key hinge/binding residue"),
               mpatches.Patch(color="#1f77b4", label="Other residue")]
    ax.legend(handles=patches, fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "08_05_top_contributors_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: 08_05_top_contributors_bar.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Step 08: Per-Residue MM-GBSA Decomposition — Top 5 RET Leads")
    print(f"Candidates: {', '.join(TOP5)}")
    print("=" * 65)

    all_results = []
    failed = []

    for cand_id in TOP5:
        try:
            res = run_decomp(cand_id)
            # Fallback parser if primary found nothing
            if not res["residues"]:
                print(f"  {cand_id}: primary parser found 0 residues, trying fallback...")
                dat = LIG_DIR / cand_id / "decomp" / "PER_RESIDUE_DECOMP.dat"
                res["residues"] = parse_decomp_v2(dat)
            n = len(res["residues"])
            print(f"  ✓ {cand_id}: {n} residues parsed")
            all_results.append(res)
        except Exception as exc:
            print(f"  ✗ {cand_id}: {exc}")
            failed.append({"CandID": cand_id, "error": str(exc)[:300]})

    if not all_results:
        print("ERROR: all decomp runs failed.")
        return

    # Save combined CSV
    rows = []
    for res in all_results:
        for rnum, rdata in res["residues"].items():
            rows.append({"CandID": res["cand_id"], **rdata,
                          "is_key": rnum in KEY_RESIDUES,
                          "key_name": KEY_RESIDUES.get(rnum, "")})
    decomp_df = pd.DataFrame(rows)
    decomp_df.to_csv(DATA_DIR / "RET_decomp_results.csv", index=False)
    decomp_df.to_csv(RESULTS_DIR / "08_decomp_results.csv", index=False)
    print(f"\nSaved → data/processed/RET_decomp_results.csv ({len(decomp_df)} rows)")

    # Key residue summary
    print("\nKey RET binding site residue contributions:")
    print(f"{'Residue':<12}", end="")
    for r in all_results:
        print(f"  {r['cand_id']:>10}", end="")
    print()
    print("-" * (12 + 12 * len(all_results)))
    for rnum, rname in KEY_RESIDUES.items():
        print(f"{rname:<12}", end="")
        for res in all_results:
            val = res["residues"].get(rnum, {}).get("total", None)
            print(f"  {val:>10.2f}" if val is not None else f"  {'N/A':>10}", end="")
        print()

    print("\nGenerating figures...")
    generate_figures(all_results)

    # Stats
    stats = {
        "top5": TOP5,
        "n_processed": len(all_results),
        "n_failed": len(failed),
        "key_residues_found": {
            res["cand_id"]: [KEY_RESIDUES[r] for r in KEY_RESIDUES if r in res["residues"]]
            for res in all_results
        },
    }
    (RESULTS_DIR / "08_decomp_stats.json").write_text(json.dumps(stats, indent=2))

    print("\n" + "=" * 65)
    print("Step 08 complete.")
    print(f"  {len(all_results)}/5 candidates decomposed")
    print(f"  5 figures saved → figures/08_decomp/")
    print("  Next → Step 09: MD simulation (GROMACS) — top 3 leads")
    print("=" * 65)


if __name__ == "__main__":
    main()
