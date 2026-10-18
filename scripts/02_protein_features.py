#!/usr/bin/env python3
"""
Step 02: Protein Structure Analysis and Binding-Site Feature Extraction
Pipeline : Generative AI-Based Drug Design for RET Kinase Type I Inhibitors
Target   : RET kinase domain — PDB 6NEC (Nintedanib complex, 1.87 Å)
Ligand   : XIN (Nintedanib / BIBF-1120)
Disease  : RET-fusion+ NSCLC, Medullary Thyroid Carcinoma

Method:
  - Identify protein residues within 5 Å of co-crystallized ligand (XIN)
  - Select 20 closest residues for BiLSTM conditioning
  - Encode each as 13-feature vector (x, y, z + 10 physicochemical features)
  - Output: 20×13 numpy array used as attention conditioning input

Outputs
-------
data/raw/6NEC.pdb                        Downloaded PDB file
data/processed/binding_site_residues.csv 20 residues + all features
data/processed/binding_site_matrix.npy   20×13 raw feature matrix
data/processed/binding_site_matrix_norm.npy  Normalized 20×13 matrix
results/02_protein_features_stats.json   Statistics and metadata
logs/02_protein_features.log             Full run log
figures/02_protein_features/             6 journal-quality figures
"""

import os
import sys
import json
import logging
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime

from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import protein_letters_3to1

warnings.filterwarnings("ignore")

# ─────────────────────────────── PATHS ────────────────────────────────────────
ROOT     = Path("/home/stalin/Desktop/RET_Generative")
DIR_RAW  = ROOT / "data" / "raw"
DIR_PROC = ROOT / "data" / "processed"
DIR_FIG  = ROOT / "figures" / "02_protein_features"
DIR_LOG  = ROOT / "logs"
DIR_RES  = ROOT / "results"

# ─────────────────────────────── CONSTANTS ────────────────────────────────────
PDB_ID       = "6NEC"
CHAIN_ID     = "A"
LIGAND_CODE  = "XIN"         # Nintedanib (BIBF-1120)
LIGAND_NAME  = "Nintedanib (BIBF-1120)"
CUTOFF_ANG   = 5.0
N_RESIDUES   = 20
N_FEATURES   = 13
PDB_URL      = f"https://files.rcsb.org/download/{PDB_ID}.pdb"

# RET hinge residue range (equivalent to FGFR1 562–572)
HINGE_RANGE  = (805, 812)
# Gatekeeper: V804 (V804M mutation = acquired selpercatinib resistance)
GATEKEEPER   = 804

# ── Amino acid property tables ────────────────────────────────────────────────
KD_HYDRO = {
    "ALA": 1.8,  "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8,  "LYS": -3.9, "MET": 1.9,  "PHE": 2.8,  "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}
CHARGE = {
    "ALA": 0,  "ARG": 1,  "ASN": 0,  "ASP": -1, "CYS": 0,
    "GLN": 0,  "GLU": -1, "GLY": 0,  "HIS": 0,  "ILE": 0,
    "LEU": 0,  "LYS": 1,  "MET": 0,  "PHE": 0,  "PRO": 0,
    "SER": 0,  "THR": 0,  "TRP": 0,  "TYR": 0,  "VAL": 0,
}
HBD = {
    "ALA": 0, "ARG": 4, "ASN": 1, "ASP": 0, "CYS": 1,
    "GLN": 1, "GLU": 0, "GLY": 0, "HIS": 1, "ILE": 0,
    "LEU": 0, "LYS": 1, "MET": 0, "PHE": 0, "PRO": 0,
    "SER": 1, "THR": 1, "TRP": 1, "TYR": 1, "VAL": 0,
}
HBA = {
    "ALA": 0, "ARG": 0, "ASN": 1, "ASP": 2, "CYS": 0,
    "GLN": 1, "GLU": 2, "GLY": 0, "HIS": 1, "ILE": 0,
    "LEU": 0, "LYS": 0, "MET": 1, "PHE": 0, "PRO": 0,
    "SER": 1, "THR": 1, "TRP": 0, "TYR": 1, "VAL": 0,
}
AROMATIC = {
    "ALA": 0, "ARG": 0, "ASN": 0, "ASP": 0, "CYS": 0,
    "GLN": 0, "GLU": 0, "GLY": 0, "HIS": 1, "ILE": 0,
    "LEU": 0, "LYS": 0, "MET": 0, "PHE": 1, "PRO": 0,
    "SER": 0, "THR": 0, "TRP": 1, "TYR": 1, "VAL": 0,
}
RES_MW = {
    "ALA": 89.1,  "ARG": 174.2, "ASN": 132.1, "ASP": 133.1, "CYS": 121.2,
    "GLN": 146.2, "GLU": 147.1, "GLY": 75.0,  "HIS": 155.2, "ILE": 131.2,
    "LEU": 131.2, "LYS": 146.2, "MET": 149.2, "PHE": 165.2, "PRO": 115.1,
    "SER": 105.1, "THR": 119.1, "TRP": 204.2, "TYR": 181.2, "VAL": 117.1,
}
SC_HEAVY = {
    "ALA": 1,  "ARG": 7,  "ASN": 4,  "ASP": 4,  "CYS": 2,
    "GLN": 5,  "GLU": 5,  "GLY": 0,  "HIS": 6,  "ILE": 4,
    "LEU": 4,  "LYS": 5,  "MET": 4,  "PHE": 7,  "PRO": 3,
    "SER": 2,  "THR": 3,  "TRP": 10, "TYR": 8,  "VAL": 3,
}
POLARITY = {
    "ALA": 0.0, "ARG": 1.0, "ASN": 0.5, "ASP": 1.0, "CYS": 0.5,
    "GLN": 0.5, "GLU": 1.0, "GLY": 0.0, "HIS": 1.0, "ILE": 0.0,
    "LEU": 0.0, "LYS": 1.0, "MET": 0.0, "PHE": 0.0, "PRO": 0.0,
    "SER": 0.5, "THR": 0.5, "TRP": 0.5, "TYR": 0.5, "VAL": 0.0,
}
SC_TYPE = {
    "ALA": 0, "ARG": 2, "ASN": 1, "ASP": 2, "CYS": 1,
    "GLN": 1, "GLU": 2, "GLY": 0, "HIS": 2, "ILE": 0,
    "LEU": 0, "LYS": 2, "MET": 0, "PHE": 0, "PRO": 0,
    "SER": 1, "THR": 1, "TRP": 1, "TYR": 1, "VAL": 0,
}

# ── Figure style ──────────────────────────────────────────────────────────────
ACCENT      = "#1A237E"
ACCENT2     = "#0277BD"
DPI         = 300
TYPE_COLORS = {"Hydrophobic": "#E64A19", "Polar": "#1976D2", "Charged": "#388E3C"}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.linewidth":    1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linewidth":    0.6,
    "figure.dpi":        DPI,
    "savefig.dpi":       DPI,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})


# ─────────────────────────────── LOGGING ──────────────────────────────────────
def setup_logging() -> logging.Logger:
    log_path = DIR_LOG / "02_protein_features.log"
    logger = logging.getLogger("RET_Step02")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def savefig(fig, name: str, log):
    path = DIR_FIG / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved: {path.name}")


# ─────────────────────────────── HELPERS ──────────────────────────────────────
def get_residue_type(res3: str) -> str:
    return ["Hydrophobic", "Polar", "Charged"][SC_TYPE.get(res3, 0)]


def get_ca_coords(residue) -> np.ndarray:
    try:
        return residue["CA"].get_vector().get_array()
    except KeyError:
        for atom in residue.get_atoms():
            return atom.get_vector().get_array()
    return np.zeros(3)


def get_ca_bfactor(residue) -> float:
    try:
        return residue["CA"].get_bfactor()
    except KeyError:
        bfs = [a.get_bfactor() for a in residue.get_atoms()]
        return float(np.mean(bfs)) if bfs else 0.0


def build_feature_vector(residue) -> np.ndarray:
    """
    13-feature vector:
      0-2 : Cα x, y, z
      3   : Kyte-Doolittle hydrophobicity
      4   : Formal charge at pH 7.4
      5   : H-bond donor (side chain)
      6   : H-bond acceptor (side chain)
      7   : Aromaticity (0/1)
      8   : Residue MW (Da)
      9   : Side chain heavy atom count
      10  : B-factor (Cα)
      11  : Polarity score (0/0.5/1)
      12  : Side chain type (0=hydrophobic, 1=polar, 2=charged)
    """
    r3  = residue.get_resname().strip()
    xyz = get_ca_coords(residue)
    return np.array([
        xyz[0], xyz[1], xyz[2],
        KD_HYDRO.get(r3, 0.0),
        float(CHARGE.get(r3, 0)),
        float(HBD.get(r3, 0)),
        float(HBA.get(r3, 0)),
        float(AROMATIC.get(r3, 0)),
        RES_MW.get(r3, 120.0),
        float(SC_HEAVY.get(r3, 2)),
        get_ca_bfactor(residue),
        POLARITY.get(r3, 0.0),
        float(SC_TYPE.get(r3, 0)),
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DOWNLOAD PDB
# ══════════════════════════════════════════════════════════════════════════════
def download_pdb(log) -> Path:
    log.info("=" * 70)
    log.info(f"SECTION 1: Loading PDB {PDB_ID} (RET kinase + Nintedanib, 1.87 Å)")
    log.info("=" * 70)
    pdb_path = DIR_RAW / f"{PDB_ID}.pdb"
    if pdb_path.exists():
        log.info(f"PDB already present: {pdb_path}")
        return pdb_path
    log.info(f"Downloading from RCSB: {PDB_URL}")
    r = requests.get(PDB_URL, timeout=60)
    r.raise_for_status()
    pdb_path.write_text(r.text)
    lines = r.text.splitlines()
    log.info(f"Saved: {pdb_path}  ({len(lines):,} lines)")
    for l in lines:
        if l.startswith(("TITLE", "REMARK   2 RESOLUTION")):
            log.info(f"  {l.strip()}")
    return pdb_path


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PARSE + IDENTIFY BINDING SITE
# ══════════════════════════════════════════════════════════════════════════════
def parse_and_find_site(pdb_path: Path, log):
    log.info("=" * 70)
    log.info("SECTION 2: Structure Parsing and Binding Site Identification")
    log.info("=" * 70)

    parser    = PDBParser(QUIET=True)
    structure = parser.get_structure(PDB_ID, str(pdb_path))
    model     = structure[0]

    log.info(f"Chains: {[c.id for c in model.get_chains()]}")

    # Find XIN ligand atoms in chain A
    ligand_residues = []
    for chain in model.get_chains():
        if chain.id != CHAIN_ID:
            continue
        for res in chain.get_residues():
            if res.get_resname().strip() == LIGAND_CODE:
                ligand_residues.append(res)

    if not ligand_residues:
        raise ValueError(f"Ligand {LIGAND_CODE} not found in chain {CHAIN_ID} of {PDB_ID}")

    primary_lig       = ligand_residues[0]
    primary_lig_atoms = list(primary_lig.get_atoms())
    lig_coords        = np.array([a.get_vector().get_array() for a in primary_lig_atoms])
    centroid          = lig_coords.mean(axis=0)

    log.info(f"Ligand {LIGAND_CODE} ({LIGAND_NAME}): {len(primary_lig_atoms)} atoms")
    log.info(f"Ligand centroid (docking box): ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")

    # Neighbor search: protein atoms in chain A within CUTOFF_ANG
    protein_atoms = [
        a for a in model.get_atoms()
        if a.get_parent().get_parent().id == CHAIN_ID
        and a.get_parent().id[0] == " "
    ]
    ns = NeighborSearch(protein_atoms)

    nearby = set()
    for lig_atom in primary_lig_atoms:
        hits = ns.search(lig_atom.get_vector().get_array(), CUTOFF_ANG, "R")
        nearby.update(hits)

    log.info(f"Protein residues within {CUTOFF_ANG} Å: {len(nearby)}")

    # Min distance from each residue to any ligand atom
    res_dist = {}
    for res in nearby:
        min_d = min(
            (ra.get_vector() - la.get_vector()).norm()
            for ra in res.get_atoms()
            for la in primary_lig_atoms
        )
        res_dist[res] = min_d

    selected = sorted(res_dist.items(), key=lambda x: x[1])[:N_RESIDUES]

    log.info(f"\nTop {N_RESIDUES} binding site residues:")
    log.info(f"{'Rank':>4}  {'Res':>8}  {'Dist (Å)':>10}  {'Type':>12}  {'B-factor':>9}  {'Role'}")
    log.info("-" * 68)

    records = []
    for rank, (res, dist) in enumerate(selected, 1):
        r3    = res.get_resname().strip()
        rnum  = res.get_id()[1]
        rtype = get_residue_type(r3)
        bf    = get_ca_bfactor(res)
        try:
            r1 = protein_letters_3to1.get(r3, "X")
        except Exception:
            r1 = "X"
        is_hinge     = HINGE_RANGE[0] <= rnum <= HINGE_RANGE[1]
        is_gatekeeper = rnum == GATEKEEPER
        role = "HINGE" if is_hinge else ("GATEKEEPER" if is_gatekeeper else "")
        log.info(f"{rank:>4}  {r3+str(rnum):>8}  {dist:>10.3f}  {rtype:>12}  "
                 f"{bf:>9.2f}  {role}")
        records.append({
            "rank":       rank,
            "chain":      CHAIN_ID,
            "res_num":    rnum,
            "res_name":   r3,
            "res_1let":   r1,
            "min_dist_A": float(dist),
            "type":       rtype,
            "b_factor":   float(bf),
            "is_hinge":   is_hinge,
            "is_gatekeeper": is_gatekeeper,
        })

    hinge_found = [f"{r['res_name']}{r['res_num']}" for r in records if r["is_hinge"]]
    log.info(f"\nHinge residues identified: {hinge_found}")
    gk = [r for r in records if r["is_gatekeeper"]]
    if gk:
        log.info(f"Gatekeeper residue (V804M site): {gk[0]['res_name']}{gk[0]['res_num']}")

    return structure, model, primary_lig, primary_lig_atoms, selected, records, centroid


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_matrix(selected, records, log):
    log.info("=" * 70)
    log.info("SECTION 3: Building 20×13 Feature Matrix")
    log.info("=" * 70)

    feature_names = [
        "x (Å)", "y (Å)", "z (Å)",
        "KD Hydrophobicity", "Formal Charge",
        "HB Donor", "HB Acceptor", "Aromaticity",
        "Residue MW (Da)", "SC Heavy Atoms",
        "B-factor", "Polarity", "SC Type",
    ]
    matrix = np.zeros((N_RESIDUES, N_FEATURES), dtype=np.float32)

    for i, ((res, dist), rec) in enumerate(zip(selected, records)):
        fv = build_feature_vector(res)
        matrix[i] = fv
        rec.update({fn: float(fv[j]) for j, fn in enumerate(feature_names)})

    log.info(f"Matrix shape: {matrix.shape}")
    log.info(f"\n{'Feature':>22}  {'Min':>8}  {'Max':>8}  {'Mean':>8}  {'Std':>8}")
    log.info("-" * 58)
    for j, fn in enumerate(feature_names):
        col = matrix[:, j]
        log.info(f"{fn:>22}  {col.min():>8.3f}  {col.max():>8.3f}  "
                 f"{col.mean():>8.3f}  {col.std():>8.3f}")

    # Min-max normalization per feature
    matrix_norm = np.zeros_like(matrix)
    for j in range(N_FEATURES):
        col = matrix[:, j]
        rng = col.max() - col.min()
        matrix_norm[:, j] = (col - col.min()) / rng if rng > 1e-6 else 0.0

    np.save(DIR_PROC / "binding_site_matrix.npy",      matrix)
    np.save(DIR_PROC / "binding_site_matrix_norm.npy", matrix_norm)
    log.info(f"\nSaved: binding_site_matrix.npy  ({matrix.shape})")
    log.info(f"Saved: binding_site_matrix_norm.npy (min-max normalized)")
    return matrix, matrix_norm, feature_names


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — FIGURES (6 journal-quality)
# ══════════════════════════════════════════════════════════════════════════════
def make_figures(records, matrix, matrix_norm, feature_names, ligand_atoms, centroid, log):
    log.info("=" * 70)
    log.info("SECTION 4: Generating 6 Journal-Quality Figures")
    log.info("=" * 70)

    df     = pd.DataFrame(records)
    labels = [f"{r['res_name']}{r['res_num']}" for r in records]
    types  = [r["type"] for r in records]
    colors = [TYPE_COLORS[t] for t in types]
    dists  = [r["min_dist_A"] for r in records]
    bfacts = [r["b_factor"] for r in records]

    # ── FIG 01: Residue Distance from Ligand ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(range(N_RESIDUES), dists, color=colors,
                  edgecolor="white", linewidth=0.7, width=0.72)
    ax.axhline(CUTOFF_ANG, color="#444", ls="--", lw=1.5, alpha=0.65,
               label=f"{CUTOFF_ANG} Å cutoff")

    # Annotate hinge and gatekeeper residues
    for i, r in enumerate(records):
        if r["is_hinge"]:
            ax.text(i, dists[i] + 0.06, "H", ha="center", fontsize=8,
                    color="#1A237E", fontweight="bold")
        if r["is_gatekeeper"]:
            ax.text(i, dists[i] + 0.06, "GK", ha="center", fontsize=7.5,
                    color="#C62828", fontweight="bold")

    ax.set_xticks(range(N_RESIDUES))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Minimum Distance to Ligand (Å)", fontsize=12)
    ax.set_title(f"Binding Site Residues — Distance from {LIGAND_NAME}\n"
                 f"RET Kinase PDB {PDB_ID}, Chain {CHAIN_ID}  |  "
                 f"H = hinge residue, GK = gatekeeper (V804M resistance site)",
                 fontsize=12, fontweight="bold")
    handles = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
    handles += [
        Line2D([0], [0], color="#444", ls="--", lw=1.5, label=f"{CUTOFF_ANG} Å cutoff"),
        Line2D([0], [0], color="none", label="H = hinge  |  GK = gatekeeper"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper right", framealpha=0.9)
    ax.set_xlim(-0.6, N_RESIDUES - 0.4)
    ax.set_ylim(0, CUTOFF_ANG * 1.22)
    fig.tight_layout()
    savefig(fig, "fig01_residue_distances", log)

    # ── FIG 02: 20×13 Normalized Feature Heatmap ─────────────────────────────
    short_fn = ["x", "y", "z", "KD", "Chg", "HBD", "HBA", "Ar",
                "MW", "SC", "Bfac", "Pol", "SCT"]
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(matrix_norm, aspect="auto", cmap="RdYlBu_r",
                   interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(range(N_FEATURES))
    ax.set_xticklabels(short_fn, fontsize=9.5)
    ax.set_yticks(range(N_RESIDUES))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Binding Site Residue", fontsize=12)
    ax.set_title(f"Normalized 20×13 Binding Site Feature Matrix — RET Kinase\n"
                 f"PDB {PDB_ID} ({LIGAND_NAME}) | BiLSTM Attention Conditioning Input",
                 fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalized Value (0–1)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    # Residue type bar on left
    for i, (t, clr) in enumerate(zip(types, colors)):
        ax.add_patch(mpatches.Rectangle((-1.5, i - 0.5), 1.0, 1.0,
                                         color=clr, clip_on=False, alpha=0.82))
    # Highlight hinge rows
    for i, r in enumerate(records):
        if r["is_hinge"]:
            ax.add_patch(mpatches.Rectangle((-0.5, i - 0.5), N_FEATURES, 1.0,
                                             color="#E8EAF6", clip_on=True,
                                             alpha=0.35, zorder=0))
    for x in [3, 5, 7, 8, 10, 11, 12]:
        ax.axvline(x - 0.5, color="white", lw=1.5, alpha=0.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    savefig(fig, "fig02_feature_matrix_heatmap", log)

    # ── FIG 03: B-factor Profile ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.bar(range(N_RESIDUES), bfacts, color=colors,
           edgecolor="white", linewidth=0.7, width=0.72)
    ax.plot(range(N_RESIDUES), bfacts, "o-", color=ACCENT,
            ms=5, lw=1.5, alpha=0.8, zorder=5)
    mean_bf = np.mean(bfacts)
    ax.axhline(mean_bf, color="#444", ls="--", lw=1.2, alpha=0.65,
               label=f"Mean B-factor = {mean_bf:.1f} Å²")
    for i, r in enumerate(records):
        if r["is_hinge"]:
            ax.annotate("H", xy=(i, bfacts[i]), xytext=(i, bfacts[i] + 1.5),
                        ha="center", fontsize=8, color=ACCENT, fontweight="bold")
    ax.set_xticks(range(N_RESIDUES))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("B-factor (Å²)", fontsize=12)
    ax.set_title("Cα B-factor Profile — RET Kinase Binding Site Residues\n"
                 "(Lower = more rigid → more reliable for BiLSTM conditioning)",
                 fontsize=12, fontweight="bold")
    handles = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
    handles.append(Line2D([0], [0], color="#444", ls="--", lw=1.2,
                          label=f"Mean ({mean_bf:.1f} Å²)"))
    ax.legend(handles=handles, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig03_bfactor_profile", log)

    # ── FIG 04: Residue Type Composition + Hydrophobicity ────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    type_counts = df["type"].value_counts()
    tc_vals     = [type_counts.get(t, 0) for t in TYPE_COLORS]
    tc_colors   = list(TYPE_COLORS.values())
    wedges, texts, autotexts = ax1.pie(
        tc_vals, labels=None, colors=tc_colors,
        autopct="%1.0f%%", startangle=90, pctdistance=0.72,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight("bold"); at.set_color("white")
    ax1.text(0, 0, f"n={N_RESIDUES}\nresidues", ha="center", va="center",
             fontsize=10.5, fontweight="bold")
    ax1.legend(wedges, [f"{l}  (n={v})" for l, v in zip(TYPE_COLORS, tc_vals)],
               loc="lower center", bbox_to_anchor=(0.5, -0.16),
               fontsize=10, framealpha=0.9)
    ax1.set_title("Residue Type Composition", fontsize=12, fontweight="bold")

    kd_vals = [KD_HYDRO.get(r["res_name"], 0) for r in records]
    ax2.barh(range(N_RESIDUES), kd_vals, color=colors,
             edgecolor="white", linewidth=0.4, height=0.7)
    ax2.axvline(0, color="black", lw=1.2, alpha=0.7)
    # Highlight hinge residues
    for i, r in enumerate(records):
        if r["is_hinge"]:
            ax2.axhspan(i - 0.5, i + 0.5, color="#E3F2FD", alpha=0.5, zorder=0)
    ax2.set_yticks(range(N_RESIDUES))
    ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Kyte-Doolittle Hydrophobicity", fontsize=11)
    ax2.set_title("Hydrophobicity of Binding Site Residues\n(blue bands = hinge residues)",
                  fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig04_residue_type_hydrophobicity", log)

    # ── FIG 05: Feature Correlation Heatmap ──────────────────────────────────
    feat_df = pd.DataFrame(matrix, columns=feature_names)
    corr    = feat_df.corr()
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(N_FEATURES))
    ax.set_xticklabels(short_fn, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(N_FEATURES))
    ax.set_yticklabels(short_fn, fontsize=9)
    for i in range(N_FEATURES):
        for j in range(N_FEATURES):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if abs(val) > 0.6 else "black")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Pearson r", fontsize=10)
    ax.set_title("Feature-to-Feature Correlation — RET Kinase Binding Site Matrix",
                 fontsize=12, fontweight="bold")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    savefig(fig, "fig05_feature_correlation", log)

    # ── FIG 06: Binding Site Summary Table ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.axis("off")
    col_labels = ["Rank", "Residue", "Min Dist (Å)", "Type",
                  "B-factor (Ų)", "HBD", "HBA", "KD Hydro", "Role"]
    table_data = []
    for r in records:
        role = "HINGE" if r["is_hinge"] else ("GATEKEEPER" if r["is_gatekeeper"] else "—")
        table_data.append([
            r["rank"],
            f"{r['res_name']}{r['res_num']}",
            f"{r['min_dist_A']:.3f}",
            r["type"],
            f"{r['b_factor']:.1f}",
            HBD.get(r["res_name"], 0),
            HBA.get(r["res_name"], 0),
            f"{KD_HYDRO.get(r['res_name'], 0.0):.1f}",
            role,
        ])

    # Row colours: hinge=blue, gatekeeper=red, others from type
    def row_bg(r):
        if r["is_hinge"]:      return "#BBDEFB"
        if r["is_gatekeeper"]: return "#FFCDD2"
        return {"Hydrophobic": "#FFE0B240", "Polar": "#E3F2FD40",
                "Charged": "#E8F5E940"}.get(r["type"], "#FAFAFA")

    row_colors = [[row_bg(r)] * len(col_labels) for r in records]
    tbl = ax.table(
        cellText=table_data, colLabels=col_labels,
        cellLoc="center", loc="center", cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.32)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(ACCENT)
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#E0E0E0")
    ax.set_title(
        f"Binding Site Residues — PDB {PDB_ID} (RET Kinase + {LIGAND_NAME})\n"
        f"Blue = hinge residues (GLU805–ALA807)  |  Red = gatekeeper V804 (V804M resistance)",
        fontsize=12, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    savefig(fig, "fig06_residue_verification_table", log)

    log.info(f"All 6 figures saved to: {DIR_FIG}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log = setup_logging()
    t0  = datetime.now()
    log.info("╔" + "═" * 68 + "╗")
    log.info("║  RET KINASE GENERATIVE DRUG DESIGN PIPELINE                   ║")
    log.info("║  Step 02: Protein Structure Analysis & Feature Extraction      ║")
    log.info(f"║  PDB: {PDB_ID} (RET + {LIGAND_NAME})                    ║")
    log.info(f"║  Started: {t0.strftime('%Y-%m-%d %H:%M:%S')}                          ║")
    log.info("╚" + "═" * 68 + "╝")

    # 1. Download
    pdb_path = download_pdb(log)

    # 2. Parse + binding site
    structure, model, lig_res, lig_atoms, selected, records, centroid = \
        parse_and_find_site(pdb_path, log)

    # 3. Feature matrix
    matrix, matrix_norm, feature_names = build_feature_matrix(selected, records, log)

    # 4. Save residue CSV
    df_res  = pd.DataFrame(records)
    res_csv = DIR_PROC / "binding_site_residues.csv"
    df_res.to_csv(res_csv, index=False)
    log.info(f"Residue table saved: {res_csv}")

    # 5. Stats JSON
    hinge_res = [f"{r['res_name']}{r['res_num']}" for r in records if r["is_hinge"]]
    gk_res    = [f"{r['res_name']}{r['res_num']}" for r in records if r["is_gatekeeper"]]
    stats = {
        "pdb_id":           PDB_ID,
        "ligand_code":      LIGAND_CODE,
        "ligand_name":      LIGAND_NAME,
        "resolution_A":     1.87,
        "chain":            CHAIN_ID,
        "cutoff_angstrom":  CUTOFF_ANG,
        "n_residues":       N_RESIDUES,
        "n_features":       N_FEATURES,
        "matrix_shape":     list(matrix.shape),
        "ligand_centroid":  centroid.tolist(),
        "ligand_n_atoms":   len(lig_atoms),
        "min_dist_A":       float(min(r["min_dist_A"] for r in records)),
        "max_dist_A":       float(max(r["min_dist_A"] for r in records)),
        "mean_bfactor":     float(np.mean([r["b_factor"] for r in records])),
        "residue_types":    {t: int(sum(1 for r in records if r["type"] == t))
                             for t in TYPE_COLORS},
        "hinge_residues":   hinge_res,
        "gatekeeper":       gk_res,
        "residues": [
            {"rank": r["rank"], "res": f"{r['res_name']}{r['res_num']}",
             "dist_A": round(r["min_dist_A"], 3), "type": r["type"],
             "is_hinge": r["is_hinge"], "is_gatekeeper": r["is_gatekeeper"]}
            for r in records
        ],
    }
    json_path = DIR_RES / "02_protein_features_stats.json"
    with open(json_path, "w") as jf:
        json.dump(stats, jf, indent=2)
    log.info(f"Stats JSON saved: {json_path}")

    # 6. Summary
    log.info("=" * 70)
    log.info("BINDING SITE SUMMARY — RET KINASE")
    log.info("=" * 70)
    log.info(f"PDB           : {PDB_ID} (1.87 Å, Chain {CHAIN_ID})")
    log.info(f"Ligand        : {LIGAND_CODE} ({LIGAND_NAME}), {len(lig_atoms)} atoms")
    log.info(f"Centroid      : ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
    log.info(f"Residues      : {N_RESIDUES} within {CUTOFF_ANG} Å")
    log.info(f"Dist range    : {stats['min_dist_A']:.3f} – {stats['max_dist_A']:.3f} Å")
    log.info(f"Mean B-factor : {stats['mean_bfactor']:.2f} Å²")
    log.info(f"Types         : {stats['residue_types']}")
    log.info(f"Hinge residues: {hinge_res}")
    log.info(f"Gatekeeper    : {gk_res}")
    log.info(f"Matrix        : {matrix.shape} → saved as .npy")

    # 7. Figures
    make_figures(records, matrix, matrix_norm, feature_names, lig_atoms, centroid, log)

    elapsed = (datetime.now() - t0).total_seconds()
    log.info("=" * 70)
    log.info(f"Step 02 complete in {elapsed:.1f}s")
    log.info(f"Feature matrix: {matrix.shape} → ready for BiLSTM conditioning")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
