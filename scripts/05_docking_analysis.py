"""
RET Kinase Generative Drug Design Pipeline
Step 05: AutoDock-GPU Docking Results Analysis & Figures
Target: RET kinase (P07949) — PDB 6NEC
"""

import os, sys, re, json, glob, logging, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, QED
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit import RDLogger, DataStructs
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ─── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR  = "/home/stalin/Desktop/RET_Generative"
DATA_DIR     = os.path.join(PROJECT_DIR, "data/processed")
DOCK_DIR     = os.path.join(PROJECT_DIR, "docking")
RESULTS_DIR  = os.path.join(PROJECT_DIR, "docking/results")
FIG_DIR      = os.path.join(PROJECT_DIR, "figures/05_docking")
LOG_DIR      = os.path.join(PROJECT_DIR, "logs")
OUT_DIR      = os.path.join(PROJECT_DIR, "results")

CAND_CSV     = os.path.join(DATA_DIR, "RET_candidates.csv")
DOCK_CSV_OUT = os.path.join(DATA_DIR, "RET_docking_results.csv")

for d in [FIG_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("step05_ret")
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(LOG_DIR, "05_docking_analysis.log"), mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S"))
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(fh); log.addHandler(ch)

PALETTE = {
    "primary":   "#1565C0",
    "secondary": "#C62828",
    "accent":    "#2E7D32",
    "highlight": "#F57F17",
    "gray":      "#757575",
    "top":       "#E53935",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.28, "grid.linestyle": "--",
    "figure.dpi": 100, "savefig.dpi": 300,
})


# ══════════════════════════════════════════════════════════════════════════════
# DLG PARSING
# ══════════════════════════════════════════════════════════════════════════════
def parse_dlg(dlg_path):
    """Extract best binding energy, RMSD, and all cluster data from a DLG file."""
    best_energy      = None
    cluster_energies = []
    cluster_rmsds    = []
    ligand_name      = os.path.basename(dlg_path).replace(".dlg", "")
    ligand_file      = None
    n_runs           = 0
    all_energies     = []

    energy_re = re.compile(
        r"DOCKED: USER\s+Estimated Free Energy of Binding\s+=\s+([-\d.]+)")
    rmsd_re   = re.compile(
        r"DOCKED: USER\s+RMSD from best mode\s+=\s+([\d.]+)")
    cluster_re = re.compile(
        r"CLUSTERING HISTOGRAM.*?(?=RMSD TABLE|\Z)", re.DOTALL)
    lfile_re   = re.compile(r"Ligand file:\s+(.+)")
    run_re     = re.compile(r"^Run\s+(\d+)/", re.MULTILINE)

    try:
        with open(dlg_path) as f:
            content = f.read()
    except Exception as e:
        return None

    # Ligand file
    m = lfile_re.search(content)
    if m:
        ligand_file = os.path.basename(m.group(1).strip())

    # All binding energies (one per pose/run)
    all_energies = [float(x) for x in energy_re.findall(content)]
    if all_energies:
        best_energy = min(all_energies)

    # All RMSDs
    all_rmsds = [float(x) for x in rmsd_re.findall(content)]

    # Count runs
    runs = run_re.findall(content)
    n_runs = max([int(r) for r in runs]) if runs else len(all_energies)

    return {
        "dlg":          dlg_path,
        "ligand_name":  ligand_name,
        "ligand_file":  ligand_file,
        "best_energy":  best_energy,
        "mean_energy":  float(np.mean(all_energies)) if all_energies else None,
        "std_energy":   float(np.std(all_energies))  if all_energies else None,
        "n_poses":      len(all_energies),
        "n_runs":       n_runs,
        "all_energies": all_energies,
    }


def _save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
def fig_binding_energy_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax = axes[0]
    n_top = (df["DockingScore"] <= -9.0).sum()
    ax.hist(df["DockingScore"], bins=25, color=PALETTE["primary"],
            edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(df["DockingScore"].mean(), color=PALETTE["secondary"],
               lw=2, ls="--", label=f"Mean: {df['DockingScore'].mean():.2f} kcal/mol")
    ax.axvline(-9.0, color=PALETTE["highlight"], lw=1.5, ls=":",
               label=f"−9.0 threshold (n={n_top})")
    ax.set_xlabel("Estimated Free Energy of Binding (kcal/mol)", fontsize=12)
    ax.set_ylabel("Number of Compounds", fontsize=12)
    ax.set_title("Docking Score Distribution\nRET Kinase (PDB: 6NEC)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # Ranked bar chart — top 25
    ax2 = axes[1]
    top25 = df.nsmallest(25, "DockingScore").reset_index(drop=True)
    colors = [PALETTE["top"] if v <= -10.0 else PALETTE["primary"]
              for v in top25["DockingScore"]]
    bars = ax2.barh(top25["CandID"][::-1], top25["DockingScore"][::-1],
                    color=colors[::-1], edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, top25["DockingScore"][::-1]):
        ax2.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", ha="right", fontsize=7.5,
                 color="white", fontweight="bold")
    ax2.set_xlabel("Docking Score (kcal/mol)", fontsize=12)
    ax2.set_title("Top 25 Candidates by Docking Score", fontsize=12, fontweight="bold")
    ax2.axvline(-9.0, color=PALETTE["highlight"], lw=1.5, ls=":", alpha=0.7)
    patch_top = mpatches.Patch(color=PALETTE["top"],     label="Score ≤ −10.0")
    patch_ok  = mpatches.Patch(color=PALETTE["primary"], label="Score −10.0 to −9.0")
    ax2.legend(handles=[patch_top, patch_ok], fontsize=9, loc="lower right")

    fig.suptitle("AutoDock-GPU Docking Results — RET Kinase Generative Candidates",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig01_docking_score_distribution.png")


def fig_score_vs_properties(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    props = [("QED",  "QED",                  "tab:blue"),
             ("MW",   "Molecular Weight (Da)", "tab:green"),
             ("LogP", "LogP",                  "tab:orange"),
             ("TPSA", "TPSA (Å²)",             "tab:purple")]

    for ax, (col, label, color) in zip(axes, props):
        sc = ax.scatter(df[col], df["DockingScore"],
                        c=df["DockingScore"], cmap="RdYlGn_r",
                        s=55, alpha=0.80, edgecolors="white", linewidths=0.4,
                        vmin=df["DockingScore"].min(), vmax=df["DockingScore"].max())
        # Annotate top 5
        top5 = df.nsmallest(5, "DockingScore")
        for _, row in top5.iterrows():
            ax.annotate(row["CandID"], (row[col], row["DockingScore"]),
                        fontsize=7, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points", color="darkred")
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Docking Score (kcal/mol)", fontsize=11)
        ax.set_title(f"Docking Score vs {label}", fontsize=11, fontweight="bold")
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="ΔG (kcal/mol)")
        ax.axhline(-9.0, color="gray", lw=1, ls="--", alpha=0.6)

    fig.suptitle("Docking Score vs Physicochemical Properties — RET Kinase Candidates",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig02_score_vs_properties.png")


def fig_top20_structures(df, n=20):
    top = df.nsmallest(n, "DockingScore").reset_index(drop=True)
    mols, legends = [], []
    for _, row in top.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol:
            mols.append(mol)
            legends.append(f"{row['CandID']}\n{row['DockingScore']:.2f} kcal/mol\nQED={row['QED']:.3f}")
    if not mols:
        return
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(440, 320),
                                legends=legends, returnPNG=False)
    out = os.path.join(FIG_DIR, "fig03_top20_structures.png")
    img.save(out, dpi=(300, 300))
    log.info("  Saved: fig03_top20_structures.png")


def fig_energy_convergence(df):
    """Plot energy distributions per candidate (violin-like using all runs)."""
    top15 = df.nsmallest(15, "DockingScore").reset_index(drop=True)
    data  = [row["AllEnergies"] for _, row in top15.iterrows()]
    labels= [f"{row['CandID']}\n{row['DockingScore']:.2f}" for _, row in top15.iterrows()]

    fig, ax = plt.subplots(figsize=(14, 6))
    parts = ax.violinplot(data, positions=range(len(data)), showmedians=True,
                          showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(PALETTE["primary"])
        pc.set_alpha(0.65)
    parts["cmedians"].set_colors(PALETTE["secondary"])
    parts["cmedians"].set_linewidth(2)

    # Overlay best energy dots
    for i, (_, row) in enumerate(top15.iterrows()):
        ax.scatter(i, row["DockingScore"], color=PALETTE["top"], s=60,
                   zorder=5, marker="*")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Binding Energy (kcal/mol)", fontsize=12)
    ax.set_title("Docking Energy Convergence — Top 15 Candidates\n"
                 "(violin = all 50 LGA runs; ★ = best pose)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig04_energy_convergence.png")


def fig_summary_dashboard(df):
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

    # Panel A: Score histogram
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.hist(df["DockingScore"], bins=20, color=PALETTE["primary"],
             edgecolor="white", linewidth=0.5, alpha=0.85)
    ax0.axvline(df["DockingScore"].mean(), color=PALETTE["secondary"],
                lw=2, ls="--", label=f"Mean {df['DockingScore'].mean():.2f}")
    ax0.axvline(-9.0, color=PALETTE["highlight"], lw=1.5, ls=":",
                label="−9.0 kcal/mol")
    ax0.set_xlabel("ΔG (kcal/mol)", fontsize=11)
    ax0.set_ylabel("Count", fontsize=11)
    ax0.set_title("Score Distribution", fontsize=12, fontweight="bold")
    ax0.legend(fontsize=9)

    # Panel B: QED vs score
    ax1 = fig.add_subplot(gs[0, 2:])
    sc  = ax1.scatter(df["QED"], df["DockingScore"],
                      c=df["DockingScore"], cmap="RdYlGn_r", s=50, alpha=0.8,
                      edgecolors="white", linewidths=0.3)
    plt.colorbar(sc, ax=ax1, fraction=0.04, pad=0.02)
    ax1.set_xlabel("QED", fontsize=11)
    ax1.set_ylabel("ΔG (kcal/mol)", fontsize=11)
    ax1.set_title("QED vs Docking Score", fontsize=12, fontweight="bold")
    ax1.axhline(-9.0, color="gray", lw=1, ls="--", alpha=0.5)

    # Panel C: Top 10 table
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis("off")
    top10 = df.nsmallest(10, "DockingScore")[
        ["Rank", "CandID", "DockingScore", "QED", "MW", "LogP", "TPSA", "HBD", "HBA", "SMILES"]]
    top10 = top10.copy()
    top10["DockingScore"] = top10["DockingScore"].apply(lambda x: f"{x:.2f}")
    top10["QED"]  = top10["QED"].apply(lambda x: f"{x:.3f}")
    top10["MW"]   = top10["MW"].apply(lambda x: f"{x:.1f}")
    top10["LogP"] = top10["LogP"].apply(lambda x: f"{x:.2f}")
    top10["TPSA"] = top10["TPSA"].apply(lambda x: f"{x:.1f}")
    top10["SMILES"] = top10["SMILES"].apply(lambda s: s[:38] + "…" if len(s) > 38 else s)
    top10.columns = ["Rank", "ID", "ΔG (kcal/mol)", "QED", "MW", "LogP", "TPSA", "HBD", "HBA", "SMILES"]
    tbl = ax2.table(cellText=top10.values,
                    colLabels=top10.columns,
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1565C0"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#E3F2FD")
        if r > 0 and c == 2:   # docking score col
            val = float(cell.get_text().get_text())
            if val <= -10.0:
                cell.set_facecolor("#FFCDD2")
    ax2.set_title("Top 10 Docked Candidates", fontsize=13, fontweight="bold", pad=6)

    # Panel D: Properties of top 20 vs all
    ax3 = fig.add_subplot(gs[2, :2])
    top20 = df.nsmallest(20, "DockingScore")
    props_lbl = ["QED", "MW", "LogP", "TPSA"]
    top20_means = [top20[p].mean() for p in props_lbl]
    all_means   = [df[p].mean()    for p in props_lbl]
    x = np.arange(len(props_lbl)); w = 0.32
    ax3.bar(x - w/2, all_means,   w, label="All 100", color=PALETTE["primary"],  alpha=0.75)
    ax3.bar(x + w/2, top20_means, w, label="Top 20",  color=PALETTE["secondary"], alpha=0.75)
    ax3.set_xticks(x); ax3.set_xticklabels(props_lbl, fontsize=11)
    ax3.set_title("Mean Properties: All vs Top 20", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=10)

    # Panel E: Summary stats
    ax4 = fig.add_subplot(gs[2, 2:])
    ax4.axis("off")
    n_below_9  = (df["DockingScore"] <= -9.0).sum()
    n_below_10 = (df["DockingScore"] <= -10.0).sum()
    stats_txt  = (
        f"AutoDock-GPU Results — RET Kinase (PDB: 6NEC)\n"
        f"{'─'*38}\n"
        f"Total docked compounds : {len(df)}\n"
        f"ΔG best                : {df['DockingScore'].min():.2f} kcal/mol\n"
        f"ΔG worst               : {df['DockingScore'].max():.2f} kcal/mol\n"
        f"ΔG mean ± std          : {df['DockingScore'].mean():.2f} ± {df['DockingScore'].std():.2f}\n"
        f"ΔG ≤ −9.0 kcal/mol     : {n_below_9} compounds\n"
        f"ΔG ≤ −10.0 kcal/mol    : {n_below_10} compounds\n"
        f"{'─'*38}\n"
        f"Top hit QED            : {df.nsmallest(1,'DockingScore')['QED'].values[0]:.3f}\n"
        f"Top hit MW             : {df.nsmallest(1,'DockingScore')['MW'].values[0]:.1f} Da\n"
        f"Top hit LogP           : {df.nsmallest(1,'DockingScore')['LogP'].values[0]:.2f}\n"
    )
    ax4.text(0.05, 0.95, stats_txt, transform=ax4.transAxes,
             fontsize=10, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3E5F5", edgecolor="#9C27B0"))

    fig.suptitle("Step 05: AutoDock-GPU Docking Summary — RET Kinase Generative Pipeline",
                 fontsize=14, fontweight="bold", y=1.01)
    _save(fig, "fig05_docking_summary_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  RET KINASE GENERATIVE DRUG DESIGN PIPELINE                 ║")
    log.info("║  Step 05: AutoDock-GPU Docking Analysis                     ║")
    log.info(f"║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                          ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")

    # ── Parse DLG files ───────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 1: Parsing DLG Files")
    log.info("=" * 70)

    dlg_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.dlg")))
    log.info(f"Found {len(dlg_files)} DLG files")

    records = []
    for dlg in dlg_files:
        rec = parse_dlg(dlg)
        if rec and rec["best_energy"] is not None:
            records.append(rec)
        else:
            log.warning(f"  Skipping {os.path.basename(dlg)} — no energy found")

    log.info(f"Parsed {len(records)} valid DLG files")
    log.info(f"Best binding energy : {min(r['best_energy'] for r in records):.2f} kcal/mol")
    log.info(f"Worst binding energy: {max(r['best_energy'] for r in records):.2f} kcal/mol")

    # ── Map back to candidates ────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 2: Merging with Candidate Properties")
    log.info("=" * 70)

    cand_df = pd.read_csv(CAND_CSV)
    log.info(f"Candidates CSV: {len(cand_df)} rows")

    # Build mapping: DLG job index → ligand file → candidate ID
    # DLG job numbering is sequential but ligand assignment depends on batch order
    # Extract ligand name from dlg records
    dlg_records = []
    for rec in records:
        lf = rec.get("ligand_file", "")
        # e.g. "RET_cand-27.pdbqt" → rank 27
        m = re.search(r"RET_cand-(\d+)", lf if lf else rec["ligand_name"])
        if m:
            cand_rank = int(m.group(1))
        else:
            # Try from dlg filename: RET_cand_20.dlg → job 20 (not same as cand rank)
            # Use all_energies best for ordering
            cand_rank = None
        dlg_records.append({
            "CandRank":    cand_rank,
            "DockingScore":rec["best_energy"],
            "MeanScore":   rec["mean_energy"],
            "StdScore":    rec["std_energy"],
            "NPoses":      rec["n_poses"],
            "AllEnergies": rec["all_energies"],
            "DLGFile":     os.path.basename(rec["dlg"]),
        })

    dlg_df = pd.DataFrame(dlg_records)
    log.info(f"DLG records with rank: {dlg_df['CandRank'].notna().sum()} / {len(dlg_df)}")

    # Merge on candidate rank
    merged = cand_df.copy()
    merged["CandRank"] = merged["Rank"]
    merged = merged.merge(
        dlg_df[dlg_df["CandRank"].notna()].astype({"CandRank": int}),
        on="CandRank", how="left"
    )

    # Create CandID label
    merged["CandID"] = merged["Rank"].apply(lambda x: f"RET_{int(x):03d}")

    # Report merge stats
    n_docked = merged["DockingScore"].notna().sum()
    log.info(f"Merged {n_docked} / {len(merged)} candidates with docking scores")

    # For any unmatched (shouldn't happen), fill from remaining dlg_df rows
    unmatched_dlg = dlg_df[dlg_df["CandRank"].isna()]
    if not unmatched_dlg.empty:
        log.warning(f"  {len(unmatched_dlg)} DLG files could not be matched to candidates by name")

    # Drop rows with no docking score
    result_df = merged[merged["DockingScore"].notna()].copy()
    result_df = result_df.sort_values("DockingScore").reset_index(drop=True)
    result_df["DockRank"] = result_df.index + 1

    log.info(f"Final docked set: {len(result_df)} candidates")
    log.info(f"  Best  ΔG: {result_df['DockingScore'].min():.2f} kcal/mol")
    log.info(f"  Worst ΔG: {result_df['DockingScore'].max():.2f} kcal/mol")
    log.info(f"  Mean  ΔG: {result_df['DockingScore'].mean():.2f} ± {result_df['DockingScore'].std():.2f}")
    log.info(f"  ΔG ≤ −9.0 : {(result_df['DockingScore'] <= -9.0).sum()} candidates")
    log.info(f"  ΔG ≤ −10.0: {(result_df['DockingScore'] <= -10.0).sum()} candidates")

    log.info("\nTop 15 by docking score:")
    top15_show = result_df.nsmallest(15, "DockingScore")[
        ["DockRank", "CandID", "DockingScore", "QED", "MW", "LogP", "SMILES"]]
    for _, row in top15_show.iterrows():
        log.info(f"  #{int(row['DockRank']):3d} {row['CandID']}  "
                 f"ΔG={row['DockingScore']:6.2f}  QED={row['QED']:.3f}  "
                 f"MW={row['MW']:.0f}  LogP={row['LogP']:.2f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    save_cols = ["DockRank", "CandID", "DockingScore", "MeanScore", "StdScore",
                 "NPoses", "SMILES", "MW", "LogP", "TPSA", "QED",
                 "HBD", "HBA", "RotB", "Scaffold", "DLGFile"]
    save_cols = [c for c in save_cols if c in result_df.columns]
    result_df[save_cols].to_csv(DOCK_CSV_OUT, index=False)
    log.info(f"\nDocking CSV → {DOCK_CSV_OUT}")

    # ── Figures ───────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 3: Figures")
    log.info("=" * 70)
    fig_binding_energy_distribution(result_df)
    fig_score_vs_properties(result_df)
    fig_top20_structures(result_df, n=20)
    fig_energy_convergence(result_df)
    fig_summary_dashboard(result_df)

    # ── Results JSON ──────────────────────────────────────────────────────────
    top5 = result_df.nsmallest(5, "DockingScore")
    results = {
        "step":    "05_autodock_gpu",
        "target":  "RET kinase (P07949)",
        "pdb":     "6NEC",
        "box_center": [5.72, 4.62, -3.27],
        "box_size":   [22.5, 22.5, 22.5],
        "timestamp": datetime.now().isoformat(),
        "n_docked":  int(len(result_df)),
        "best_score":  round(float(result_df["DockingScore"].min()), 3),
        "worst_score": round(float(result_df["DockingScore"].max()), 3),
        "mean_score":  round(float(result_df["DockingScore"].mean()), 3),
        "std_score":   round(float(result_df["DockingScore"].std()), 3),
        "n_below_9":   int((result_df["DockingScore"] <= -9.0).sum()),
        "n_below_10":  int((result_df["DockingScore"] <= -10.0).sum()),
        "top5": [
            {
                "rank":  int(row["DockRank"]),
                "id":    row["CandID"],
                "score": round(float(row["DockingScore"]), 3),
                "qed":   round(float(row["QED"]), 4),
                "mw":    round(float(row["MW"]), 1),
                "smiles":row["SMILES"],
            }
            for _, row in top5.iterrows()
        ],
    }
    results_path = os.path.join(OUT_DIR, "05_docking_stats.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results JSON → {results_path}")

    log.info("=" * 70)
    log.info(f"Step 05 complete")
    log.info(f"Docked candidates : {len(result_df)}")
    log.info(f"Best ΔG           : {result_df['DockingScore'].min():.2f} kcal/mol")
    log.info(f"ΔG ≤ −9.0         : {(result_df['DockingScore'] <= -9.0).sum()} compounds")
    log.info(f"ΔG ≤ −10.0        : {(result_df['DockingScore'] <= -10.0).sum()} compounds")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
