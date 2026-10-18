"""
RET Kinase Generative Drug Design Pipeline
Step 05b: Boltz-2 Binding Affinity Prediction
Target: RET kinase (P07949) — Top 20 docked candidates
Reference: Boltz-2 (MIT, 2025) — first deep-learning model approaching FEP accuracy
"""

import os, sys, json, glob, logging, subprocess, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ─── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR  = "/home/stalin/Desktop/RET_Generative"
DATA_DIR     = os.path.join(PROJECT_DIR, "data/processed")
BOLTZ_DIR    = os.path.join(PROJECT_DIR, "boltz")
BOLTZ_INPUTS = os.path.join(BOLTZ_DIR, "inputs")
BOLTZ_OUTPUT = os.path.join(BOLTZ_DIR, "outputs")
FIG_DIR      = os.path.join(PROJECT_DIR, "figures/05b_boltz2")
LOG_DIR      = os.path.join(PROJECT_DIR, "logs")
RESULTS_DIR  = os.path.join(PROJECT_DIR, "results")

DOCK_CSV     = os.path.join(DATA_DIR, "RET_docking_results.csv")
PDB_6NEC     = os.path.join(PROJECT_DIR, "data/raw/6NEC.pdb")
OUT_CSV      = os.path.join(DATA_DIR, "RET_boltz2_affinity.csv")

for d in [BOLTZ_INPUTS, BOLTZ_OUTPUT, FIG_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("step05b_ret")
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(LOG_DIR, "05b_boltz2_affinity.log"), mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S"))
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(fh); log.addHandler(ch)

# ─── RET kinase sequence (6NEC chain A, 287 residues) ─────────────────────────
RET_SEQUENCE = (
    "EDPKWEFPRKNLVLGKTLGEGEFGKVVKATAFHLKGRAGYTTVAVKMLKENASPSELRDLLSEFNVLK"
    "QVNHPHVIKLYGACSQDGPLLLIVEYAKYGSLRGFLRESRKVGPGYLGDERALTMGDLISFAWQISQGM"
    "QYLAEMKLVHRDLAARNILVAEGRKMKISDFGLSRDVYEEDSYVKRSQGRIPVKWMAIESLFDHIYTTQ"
    "SDVWSFGVLLWEIVTLGGNPYPGIPPERLFNLLKTGHRMERPDNCSEEMYRLMLQCWKQEPDKRPVFAD"
    "ISKDLEKMMVKR"
)

# ─── Pocket constraint residues (seq indices, 1-based) ────────────────────────
# Hinge: 93(GLU805), 94(TYR806), 95(ALA807), 96(LYS808), 97(TYR809), 98(GLY810), 99(SER811)
# Key:   46(LYS758), 92(VAL804/gatekeeper), 167(ASP892)
POCKET_RESIDUES = [46, 92, 93, 94, 95, 96, 97, 98, 99, 167]

N_TOP = 20   # run Boltz-2 on top N candidates by docking score

PALETTE = {
    "boltz":  "#7B1FA2",
    "dock":   "#1565C0",
    "top":    "#C62828",
    "accent": "#2E7D32",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.28, "grid.linestyle": "--",
    "figure.dpi": 100, "savefig.dpi": 300,
})


# ══════════════════════════════════════════════════════════════════════════════
# YAML CREATION
# ══════════════════════════════════════════════════════════════════════════════
def write_boltz_yaml(out_path, ligand_name, smiles, use_template=True,
                     use_pocket=True, pdb_path=None):
    """Write a Boltz-2 YAML input file for protein-ligand affinity prediction."""
    pocket_str = ""
    if use_pocket:
        contacts = "\n".join(
            f"      - [A, {idx}]" for idx in POCKET_RESIDUES
        )
        pocket_str = f"""
constraints:
  - pocket:
      binder: B
      contacts:
{contacts}
      max_distance: 8.0
"""

    template_str = ""
    if use_template and pdb_path:
        template_str = f"""
templates:
  - pdb: {pdb_path}
    chain_id: [A]
"""

    yaml_content = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {RET_SEQUENCE}
      msa: empty
  - ligand:
      id: B
      smiles: '{smiles}'
{pocket_str}{template_str}
properties:
  - affinity:
      binder: B
"""
    with open(out_path, "w") as f:
        f.write(yaml_content)


# ══════════════════════════════════════════════════════════════════════════════
# RESULT PARSING
# ══════════════════════════════════════════════════════════════════════════════
def parse_boltz_affinity(output_dir, name):
    """Find and parse Boltz-2 affinity JSON for a given prediction name."""
    # Boltz-2 output structure:
    # output_dir/boltz_results_<inputs_basename>/predictions/<name>/affinity_<name>.json
    # or: output_dir/predictions/<name>/affinity_*.json
    patterns = [
        os.path.join(output_dir, "**", f"affinity_{name}.json"),
        os.path.join(output_dir, "**", f"affinity_*.json"),
        os.path.join(output_dir, f"*{name}*", "**", "affinity*.json"),
    ]
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        # Filter to match our name
        matched = [f for f in found if name in os.path.basename(f)]
        if matched:
            with open(matched[0]) as fp:
                data = json.load(fp)
            return data
    # Broader search
    all_json = glob.glob(os.path.join(output_dir, "**", "*.json"), recursive=True)
    affinity_jsons = [f for f in all_json if "affinity" in os.path.basename(f) and name in f]
    if affinity_jsons:
        with open(affinity_jsons[0]) as fp:
            return json.load(fp)
    return None


def _save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
def fig_affinity_overview(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: pIC50 bar chart (sorted)
    ax = axes[0]
    df_sorted = df.sort_values("pIC50_boltz", ascending=False).reset_index(drop=True)
    colors = [PALETTE["top"] if v >= 7.0 else PALETTE["boltz"] for v in df_sorted["pIC50_boltz"]]
    bars = ax.barh(df_sorted["CandID"][::-1], df_sorted["pIC50_boltz"][::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, df_sorted["pIC50_boltz"][::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=8)
    ax.axvline(7.0, color="gray", lw=1.5, ls="--", alpha=0.7, label="pIC50=7 (100 nM)")
    ax.axvline(8.0, color=PALETTE["top"], lw=1.5, ls=":", alpha=0.7, label="pIC50=8 (10 nM)")
    ax.set_xlabel("Boltz-2 Predicted pIC50", fontsize=12)
    ax.set_title("Boltz-2 Predicted pIC50\n(Top 20 Docked Candidates)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel B: Binder probability
    ax2 = axes[1]
    df2 = df.sort_values("binder_prob", ascending=False).reset_index(drop=True)
    colors2 = [PALETTE["top"] if v >= 0.7 else PALETTE["boltz"] for v in df2["binder_prob"]]
    bars2 = ax2.barh(df2["CandID"][::-1], df2["binder_prob"][::-1],
                     color=colors2[::-1], edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars2, df2["binder_prob"][::-1]):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", ha="left", fontsize=8)
    ax2.axvline(0.7, color=PALETTE["top"], lw=1.5, ls="--", alpha=0.7, label="p=0.70 threshold")
    ax2.set_xlabel("Binder Probability", fontsize=12)
    ax2.set_xlim(0, 1.08)
    ax2.set_title("Boltz-2 Binder Probability\n(Top 20 Docked Candidates)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle("Boltz-2 Binding Affinity Predictions — RET Kinase Generative Candidates",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig01_boltz2_affinity_overview.png")


def fig_boltz_vs_autodock(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    sc = ax.scatter(df["DockingScore"], df["pIC50_boltz"],
                    c=df["binder_prob"], cmap="RdYlGn", s=80, alpha=0.85,
                    edgecolors="white", linewidths=0.5, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="Binder Probability")
    for _, row in df.iterrows():
        ax.annotate(row["CandID"], (row["DockingScore"], row["pIC50_boltz"]),
                    fontsize=6.5, ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points", color="#424242")
    ax.set_xlabel("AutoDock-GPU ΔG (kcal/mol)", fontsize=12)
    ax.set_ylabel("Boltz-2 Predicted pIC50", fontsize=12)
    ax.set_title("AutoDock-GPU vs Boltz-2\nBinding Affinity Correlation", fontsize=12, fontweight="bold")
    ax.axhline(7.0, color="gray", lw=1, ls="--", alpha=0.5)
    ax.axvline(-9.0, color="gray", lw=1, ls="--", alpha=0.5)

    # Correlation annotation
    corr = df[["DockingScore", "pIC50_boltz"]].corr().iloc[0, 1]
    ax.text(0.05, 0.92, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Panel B: Consensus ranking (average of normalized scores)
    ax2 = axes[1]
    df_plot = df.copy()
    # Normalize: docking score (lower=better) → rank; pIC50 (higher=better) → rank
    df_plot["dock_rank"]   = df_plot["DockingScore"].rank()              # lower ΔG = lower rank = better
    df_plot["boltz_rank"]  = df_plot["pIC50_boltz"].rank(ascending=False) # higher pIC50 = lower rank
    df_plot["consensus_rank"] = (df_plot["dock_rank"] + df_plot["boltz_rank"]) / 2
    df_plot = df_plot.sort_values("consensus_rank").reset_index(drop=True)
    df_plot["rank_display"] = range(1, len(df_plot)+1)

    colors3 = [PALETTE["top"] if i < 5 else PALETTE["boltz"] for i in range(len(df_plot))]
    bars3 = ax2.barh(df_plot["CandID"][::-1], df_plot["consensus_rank"][::-1],
                     color=colors3[::-1], edgecolor="white", linewidth=0.4)
    ax2.set_xlabel("Consensus Rank Score (lower = better)", fontsize=12)
    ax2.set_title("Consensus Ranking\n(AutoDock-GPU + Boltz-2)", fontsize=12, fontweight="bold")

    fig.suptitle("Multi-Platform Docking + Boltz-2 Affinity Consensus — RET Kinase",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig02_boltz_vs_autodock.png")
    return df_plot.sort_values("consensus_rank")


def fig_top_structures_boltz(df, n=10):
    top = df.nlargest(n, "pIC50_boltz").reset_index(drop=True)
    mols, legends = [], []
    for _, row in top.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol:
            mols.append(mol)
            legends.append(
                f"{row['CandID']}\npIC50={row['pIC50_boltz']:.2f}\n"
                f"ΔG={row['DockingScore']:.2f}\nP(binder)={row['binder_prob']:.3f}"
            )
    if not mols:
        return
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(450, 340),
                                legends=legends, returnPNG=False)
    out = os.path.join(FIG_DIR, "fig03_top_structures_boltz2.png")
    img.save(out, dpi=(300, 300))
    log.info("  Saved: fig03_top_structures_boltz2.png")


def fig_summary_dashboard(df, consensus_df):
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.45)

    # pIC50 histogram
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(df["pIC50_boltz"], bins=12, color=PALETTE["boltz"],
             edgecolor="white", alpha=0.85)
    ax0.axvline(df["pIC50_boltz"].mean(), color=PALETTE["top"], lw=2, ls="--",
                label=f"Mean {df['pIC50_boltz'].mean():.2f}")
    ax0.axvline(7.0, color="gray", lw=1.5, ls=":", label="pIC50=7")
    ax0.set_xlabel("Boltz-2 pIC50", fontsize=11)
    ax0.set_ylabel("Count", fontsize=11)
    ax0.set_title("pIC50 Distribution", fontsize=12, fontweight="bold")
    ax0.legend(fontsize=9)

    # Binder probability distribution
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(df["binder_prob"], bins=10, color=PALETTE["accent"],
             edgecolor="white", alpha=0.85)
    ax1.axvline(0.5, color="gray", lw=1.5, ls="--", label="p=0.5")
    ax1.axvline(0.7, color=PALETTE["top"], lw=1.5, ls=":", label="p=0.7")
    ax1.set_xlabel("Binder Probability", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Binder Probability Distribution", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)

    # Scatter: dock vs boltz
    ax2 = fig.add_subplot(gs[0, 2])
    sc = ax2.scatter(df["DockingScore"], df["pIC50_boltz"],
                     c=df["binder_prob"], cmap="RdYlGn", s=60, alpha=0.8,
                     edgecolors="white", linewidths=0.3, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax2, fraction=0.04, pad=0.02)
    ax2.set_xlabel("AutoDock-GPU ΔG", fontsize=11)
    ax2.set_ylabel("Boltz-2 pIC50", fontsize=11)
    ax2.set_title("Dock Score vs pIC50", fontsize=12, fontweight="bold")
    corr = df[["DockingScore", "pIC50_boltz"]].corr().iloc[0, 1]
    ax2.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax2.transAxes, fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Consensus top 10 table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    top10 = consensus_df.head(10)[[
        "CandID", "DockingScore", "pIC50_boltz", "binder_prob", "QED", "MW", "LogP", "SMILES"
    ]].copy()
    top10["DockingScore"]  = top10["DockingScore"].apply(lambda x: f"{x:.2f}")
    top10["pIC50_boltz"]   = top10["pIC50_boltz"].apply(lambda x: f"{x:.2f}")
    top10["binder_prob"]   = top10["binder_prob"].apply(lambda x: f"{x:.3f}")
    top10["QED"]           = top10["QED"].apply(lambda x: f"{x:.3f}")
    top10["MW"]            = top10["MW"].apply(lambda x: f"{x:.0f}")
    top10["LogP"]          = top10["LogP"].apply(lambda x: f"{x:.2f}")
    top10["SMILES"]        = top10["SMILES"].apply(lambda s: s[:40] + "…" if len(s) > 40 else s)
    top10.columns = ["ID", "ΔG (kcal/mol)", "pIC50 (Boltz-2)", "P(binder)", "QED", "MW", "LogP", "SMILES"]
    tbl = ax3.table(cellText=top10.values, colLabels=top10.columns,
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4A148C"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F3E5F5")
    ax3.set_title("Top 10 Consensus Leads (AutoDock-GPU + Boltz-2 Affinity)",
                  fontsize=13, fontweight="bold", pad=6)

    fig.suptitle("Step 05b: Boltz-2 Affinity Prediction Summary — RET Kinase",
                 fontsize=14, fontweight="bold", y=1.01)
    _save(fig, "fig04_summary_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  RET KINASE GENERATIVE DRUG DESIGN PIPELINE                 ║")
    log.info("║  Step 05b: Boltz-2 Binding Affinity Prediction              ║")
    log.info(f"║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                          ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")

    # ── Load docking results ──────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 1: Loading Top Docked Candidates")
    log.info("=" * 70)
    dock_df = pd.read_csv(DOCK_CSV)
    top_df  = dock_df.nsmallest(N_TOP, "DockingScore").reset_index(drop=True)
    log.info(f"Top {N_TOP} candidates selected:")
    for _, row in top_df.iterrows():
        log.info(f"  {row['CandID']}  ΔG={row['DockingScore']:.2f}  QED={row['QED']:.3f}  "
                 f"MW={row['MW']:.0f}")

    # Check atom count constraint (Boltz-2 limit: ≤128 atoms with Hs)
    for _, row in top_df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol:
            molH = Chem.AddHs(mol)
            n_atoms = molH.GetNumAtoms()
            if n_atoms > 128:
                log.warning(f"  {row['CandID']} has {n_atoms} atoms (>128) — may be unreliable")
            else:
                log.info(f"  {row['CandID']}: {n_atoms} atoms (OK)")

    # ── Create YAML inputs ────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 2: Creating Boltz-2 YAML Input Files")
    log.info("=" * 70)
    yaml_dir = os.path.join(BOLTZ_INPUTS, "top20")
    os.makedirs(yaml_dir, exist_ok=True)

    yaml_files = []
    for _, row in top_df.iterrows():
        name     = row["CandID"].replace("_", "-")   # RET-016
        yaml_out = os.path.join(yaml_dir, f"{name}.yaml")
        write_boltz_yaml(yaml_out, name, row["SMILES"],
                         use_template=True, use_pocket=True, pdb_path=PDB_6NEC)
        yaml_files.append(yaml_out)
        log.info(f"  Written: {os.path.basename(yaml_out)}")

    log.info(f"Created {len(yaml_files)} YAML files → {yaml_dir}")

    # ── Run Boltz-2 ───────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info(f"SECTION 3: Running Boltz-2 (model=boltz2, affinity_mw_correction)")
    log.info("=" * 70)

    boltz_out = os.path.join(BOLTZ_OUTPUT, "top20")
    cmd = [
        "conda", "run", "-n", "boltz",
        "boltz", "predict", yaml_dir,
        "--out_dir",   boltz_out,
        "--model",     "boltz2",
        "--accelerator", "gpu",
        "--devices",   "1",
        "--recycling_steps", "3",
        "--diffusion_samples_affinity", "5",
        "--affinity_mw_correction",
        "--output_format", "pdb",
        "--override",
    ]
    log.info(f"Command: {' '.join(cmd[3:])}")
    log.info(f"Running Boltz-2 on {len(yaml_files)} complexes (~3-5 min each)...")

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0:
        log.error(f"Boltz-2 failed:\n{proc.stderr[-3000:]}")
        sys.exit(1)

    log.info("Boltz-2 run complete")
    if proc.stdout:
        for line in proc.stdout.split("\n")[-30:]:
            if line.strip():
                log.info(f"  [boltz] {line}")

    # ── Parse results ─────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 4: Parsing Boltz-2 Affinity Outputs")
    log.info("=" * 70)

    # Discover all affinity JSON files in the output directory
    all_affinity_jsons = glob.glob(
        os.path.join(boltz_out, "**", "affinity*.json"), recursive=True)
    log.info(f"Found {len(all_affinity_jsons)} affinity JSON files")
    for f in sorted(all_affinity_jsons):
        log.info(f"  {f}")

    # Parse each
    records = []
    for _, row in top_df.iterrows():
        cand_id = row["CandID"]
        name    = cand_id.replace("_", "-")

        data = parse_boltz_affinity(boltz_out, name)
        if data is None:
            # Try with underscores
            data = parse_boltz_affinity(boltz_out, cand_id)

        if data is None:
            log.warning(f"  {cand_id}: no affinity JSON found")
            records.append({
                "CandID": cand_id, "affinity_pred_value": None,
                "binder_prob": None, "pIC50_boltz": None,
            })
            continue

        aff_val   = data.get("affinity_pred_value",       None)
        binder_p  = data.get("affinity_probability_binary", None)
        # pIC50 = 6 - log10(IC50 in μM) → if aff_val = log10(IC50 in μM)
        # then pIC50 = 6 - aff_val  (note: more negative aff_val = better binder)
        pic50 = (6.0 - float(aff_val)) if aff_val is not None else None

        log.info(f"  {cand_id}: affinity_pred={aff_val:.4f}  "
                 f"pIC50={pic50:.2f}  P(binder)={binder_p:.4f}")
        records.append({
            "CandID":             cand_id,
            "affinity_pred_value":float(aff_val) if aff_val is not None else None,
            "binder_prob":        float(binder_p) if binder_p is not None else None,
            "pIC50_boltz":        float(pic50) if pic50 is not None else None,
        })

    boltz_df = pd.DataFrame(records)
    result_df = top_df.merge(boltz_df, on="CandID", how="left")
    result_df = result_df[result_df["pIC50_boltz"].notna()].copy()

    log.info(f"\nParsed {len(result_df)} / {len(top_df)} with valid affinity predictions")
    log.info(f"pIC50 range: {result_df['pIC50_boltz'].min():.2f} – {result_df['pIC50_boltz'].max():.2f}")
    log.info(f"Binder prob range: {result_df['binder_prob'].min():.3f} – {result_df['binder_prob'].max():.3f}")

    n_high = (result_df["pIC50_boltz"] >= 7.0).sum()
    log.info(f"pIC50 ≥ 7 (≤100 nM): {n_high} candidates")

    # Sort by pIC50 descending
    result_df = result_df.sort_values("pIC50_boltz", ascending=False).reset_index(drop=True)
    log.info("\nTop 10 by Boltz-2 pIC50:")
    for _, row in result_df.head(10).iterrows():
        log.info(f"  {row['CandID']}  pIC50={row['pIC50_boltz']:.2f}  "
                 f"ΔG={row['DockingScore']:.2f}  P(binder)={row['binder_prob']:.3f}")

    # Save CSV
    save_cols = ["CandID", "DockingScore", "pIC50_boltz", "affinity_pred_value",
                 "binder_prob", "QED", "MW", "LogP", "TPSA", "HBD", "HBA", "SMILES"]
    save_cols = [c for c in save_cols if c in result_df.columns]
    result_df[save_cols].to_csv(OUT_CSV, index=False)
    log.info(f"\nBoltz-2 CSV → {OUT_CSV}")

    # ── Figures ───────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 5: Figures")
    log.info("=" * 70)
    fig_affinity_overview(result_df)
    consensus_df = fig_boltz_vs_autodock(result_df)
    fig_top_structures_boltz(result_df, n=10)
    fig_summary_dashboard(result_df, consensus_df)

    # ── Results JSON ──────────────────────────────────────────────────────────
    top3 = result_df.nlargest(3, "pIC50_boltz")
    results = {
        "step":      "05b_boltz2_affinity",
        "target":    "RET kinase (P07949)",
        "pdb":       "6NEC",
        "timestamp": datetime.now().isoformat(),
        "model":     "boltz2",
        "n_predicted": int(len(result_df)),
        "pIC50_best":  round(float(result_df["pIC50_boltz"].max()), 3),
        "pIC50_mean":  round(float(result_df["pIC50_boltz"].mean()), 3),
        "n_above_7":   int((result_df["pIC50_boltz"] >= 7.0).sum()),
        "dock_boltz_correlation": round(float(
            result_df[["DockingScore", "pIC50_boltz"]].corr().iloc[0, 1]), 3),
        "top3": [
            {"id":  row["CandID"],
             "pIC50_boltz":   round(float(row["pIC50_boltz"]), 3),
             "docking_score": round(float(row["DockingScore"]), 3),
             "binder_prob":   round(float(row["binder_prob"]), 4),
             "smiles":        row["SMILES"]}
            for _, row in top3.iterrows()
        ],
    }
    results_path = os.path.join(RESULTS_DIR, "05b_boltz2_stats.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results JSON → {results_path}")

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info(f"Step 05b complete in {elapsed/60:.1f} min")
    log.info(f"Predicted: {len(result_df)} candidates")
    log.info(f"Best pIC50: {result_df['pIC50_boltz'].max():.2f}")
    log.info(f"pIC50 ≥ 7.0: {(result_df['pIC50_boltz'] >= 7.0).sum()} candidates")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
