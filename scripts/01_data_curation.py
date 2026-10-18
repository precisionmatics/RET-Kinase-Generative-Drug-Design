#!/usr/bin/env python3
"""
Step 01: Data Curation and Preprocessing
Pipeline : Generative AI-Based Drug Design for RET Kinase Type I Inhibitors
Target   : RET kinase (UniProt P07949) — Rearranged During Transfection
Disease  : RET-fusion+ NSCLC, Medullary Thyroid Carcinoma
Source   : BindingDB May 2026 (BindingDB_All_202605_tsv.zip)

Outputs
-------
data/raw/RET_raw.tsv                  Raw extracted rows
data/interim/RET_standardized.csv     After SMILES standardization
data/processed/RET_curated.csv        Full curated set (3 activity classes)
data/processed/RET_training.csv       Training set (Active + Highly Active, pIC50 >= 7.0)
results/01_curation_stats.json        All filter statistics
logs/01_data_curation.log             Full run log
figures/01_data_curation/             13 journal-quality figures
"""

import os
import sys
import json
import logging
import zipfile
import warnings
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    Descriptors, rdMolDescriptors, AllChem, QED,
    FilterCatalog, RDConfig
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ─────────────────────────────── PATHS ────────────────────────────────────────
ROOT        = Path("/home/stalin/Desktop/RET_Generative")
BINDINGDB   = Path("/home/stalin/Downloads/BindingDB/BindingDB_All_202605_tsv.zip")
DIR_RAW     = ROOT / "data" / "raw"
DIR_INTERIM = ROOT / "data" / "interim"
DIR_PROC    = ROOT / "data" / "processed"
DIR_FIG     = ROOT / "figures" / "01_data_curation"
DIR_LOG     = ROOT / "logs"
DIR_RES     = ROOT / "results"

# ─────────────────────────────── CONSTANTS ────────────────────────────────────
UNIPROT_ID  = "P07949"          # RET kinase
UNIPROT_COL = 44
SMILES_COL  = 1
IC50_COL    = 9
KI_COL      = 8
KD_COL      = 10
LIGAND_ID   = 0

MAX_IC50_NM       = 10_000.0
LIP_HBD           = 5
LIP_HBA           = 10
LIP_MW            = 550.0
LIP_LOGP          = 5.0
MIN_CURATED_PIC50 = 6.5
MIN_TRAIN_PIC50   = 7.0

ACT_BINS   = [6.5, 7.0, 8.0, 15]
ACT_LABELS = ["Weakly Active (6.5–7)", "Active (7–8)", "Highly Active (≥8)"]

# RET Type I hinge-binding pharmacophore SMARTS
# Hinge residues: GLU805 (H-bond acceptor), ALA807 (H-bond donor)
# Key scaffolds: aminopyrimidine, pyrrolopyrimidine, indazole, quinazoline,
#                pyrazolopyridine, benzimidazole, anilinoquinoline, naphthyridine,
#                pyrazole-NH, imidazole-NH, broad aromatic-NH (selpercatinib class)
TYPE_I_SMARTS = {
    # classic aminoheterocycles
    "aminopyrimidine":     "[NH2,NH1;!$(NC=O)]c1ncccn1",
    "aminopyridine":       "[NH2,NH1;!$(NC=O)]c1ccncc1",
    "aminoquinoline":      "[NH2,NH1;!$(NC=O)]c1ccc2ncccc2c1",
    "aminoquinazoline":    "[NH2,NH1;!$(NC=O)]c1nc2ccccc2nc1",
    "anilinopyrimidine":   "Nc1ccnc(N)n1",
    # fused bicyclics (hinge-NH via aromatic nitrogen)
    "quinazoline":         "c1cnc2ccccc2n1",
    "pyrrolopyrimidine":   "c1cnc2[nH]ccc21",
    "pyrazolopyrimidine":  "c1nn2ccncc2c1",
    "imidazopyridazine":   "c1cnc2ccnn21",
    "pyridopyrimidine":    "c1cnc2ncccc21",
    "naphthyridine":       "c1cncc2ncccc12",
    "phthalazine":         "c1cncc2ccccc12",
    # NH-containing aromatics (dominant in modern RET inhibitors)
    "indazole_NH":         "c1ccc2[nH]ncc2c1",
    "benzimidazole_NH":    "c1cnc2ccccc2[nH]1",
    "pyrazolopyridine_NH": "c1cc2cn[nH]c2nc1",
    # broad aromatic NH — captures pyrazoles, imidazoles, pyrrolopyridines,
    # indoles, carbazoles used in selpercatinib-class scaffolds
    "aromatic_NH":         "a[nH]a",
    # pyrazole explicitly (1H-pyrazole; key scaffold in >30% of RET inhibitors)
    "pyrazole_NH":         "c1c[nH]nc1",
    "imidazole_NH":        "c1c[nH]cn1",
    # pyridinone / lactam hinge donor
    "pyridinone":          "O=c1cc[nH]cc1",
    "pyrimidinone":        "O=c1[nH]ccnc1",
}

# Type II / DFG-out marker (urea/thiourea) — compounds with these are excluded
TYPE_II_SMARTS = [
    "[NX3H1,NX3H2]C(=O)[NX3H1,NX3H2]",
    "[NX3H1,NX3H2]C(=S)[NX3H1,NX3H2]",
]

# ─────────────────────────────── STYLE ────────────────────────────────────────
PALETTE = {
    "Weakly Active (6.5–7)": "#5B8DB8",
    "Active (7–8)":           "#4CAF50",
    "Highly Active (≥8)":     "#E53935",
}
ACCENT      = "#1A237E"
ACCENT2     = "#0277BD"
ACCENT3     = "#37474F"
GRID_ALPHA  = 0.22
DPI         = 300

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
    "grid.alpha":        GRID_ALPHA,
    "grid.linewidth":    0.6,
    "legend.frameon":    True,
    "legend.framealpha": 0.9,
    "legend.fontsize":   10,
    "figure.dpi":        DPI,
    "savefig.dpi":       DPI,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})


# ─────────────────────────────── LOGGING ──────────────────────────────────────
def setup_logging() -> logging.Logger:
    log_path = DIR_LOG / "01_data_curation.log"
    logger = logging.getLogger("RET_Step01")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────── HELPERS ──────────────────────────────────────
def parse_affinity(s: str):
    s = s.strip()
    if not s:
        return None, None
    censored = s.startswith(">") or s.startswith("<")
    try:
        return float(s.lstrip("><").strip()), censored
    except ValueError:
        return None, None


def standardize_smiles(smi: str):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        lfc = rdMolStandardize.LargestFragmentChooser()
        mol = lfc.choose(mol)
        un  = rdMolStandardize.Uncharger()
        mol = un.uncharge(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def compute_properties(mol):
    return {
        "MW":         Descriptors.MolWt(mol),
        "ExactMW":    Descriptors.ExactMolWt(mol),
        "LogP":       Descriptors.MolLogP(mol),
        "HBD":        rdMolDescriptors.CalcNumHBD(mol),
        "HBA":        rdMolDescriptors.CalcNumHBA(mol),
        "TPSA":       Descriptors.TPSA(mol),
        "RotBonds":   rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Rings":      rdMolDescriptors.CalcNumRings(mol),
        "ArRings":    rdMolDescriptors.CalcNumAromaticRings(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "QED":        QED.qed(mol),
        "SA":         sascorer.calculateScore(mol),
        "Fsp3":       rdMolDescriptors.CalcFractionCSP3(mol),
        "MR":         Descriptors.MolMR(mol),
    }


def lipinski_pass(props: dict) -> bool:
    return (
        props["HBD"]  <= LIP_HBD  and
        props["HBA"]  <= LIP_HBA  and
        props["MW"]   <= LIP_MW   and
        props["LogP"] <= LIP_LOGP
    )


def setup_pains_filter():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog.FilterCatalog(params)


def is_pains(mol, catalog) -> bool:
    return catalog.HasMatch(mol)


def detect_type_i(mol) -> bool:
    compiled = {k: Chem.MolFromSmarts(v) for k, v in TYPE_I_SMARTS.items()}
    for patt in compiled.values():
        if patt and mol.HasSubstructMatch(patt):
            return True
    return False


def detect_type_ii(mol) -> bool:
    for smarts in TYPE_II_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            return True
    return False


def get_murcko_scaffold(mol):
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), canonical=True)
    except Exception:
        return ""


def assign_activity_class(pic50: float) -> str:
    if pic50 < 6.5:
        return "Inactive (<6.5)"
    elif pic50 < 7.0:
        return "Weakly Active (6.5–7)"
    elif pic50 < 8.0:
        return "Active (7–8)"
    else:
        return "Highly Active (≥8)"


def savefig(fig, name: str, log):
    path = DIR_FIG / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved: {path.name}")


def morgan_fp(mol, radius=2, nbits=2048):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — EXTRACT FROM BINDINGDB
# ══════════════════════════════════════════════════════════════════════════════
def extract_bindingdb(log) -> pd.DataFrame:
    log.info("=" * 70)
    log.info("SECTION 1: Extracting RET (P07949) from BindingDB")
    log.info("=" * 70)

    rows = []
    total_scanned = 0

    with zipfile.ZipFile(BINDINGDB) as z:
        with z.open("BindingDB_All.tsv") as f:
            header = f.readline().decode("utf-8", errors="replace").strip().split("\t")
            log.info(f"BindingDB columns   : {len(header)}")

            for line in f:
                total_scanned += 1
                parts = line.decode("utf-8", errors="replace").strip().split("\t")
                if len(parts) <= UNIPROT_COL:
                    continue
                if UNIPROT_ID not in parts[UNIPROT_COL]:
                    continue

                def get(col):
                    return parts[col].strip() if len(parts) > col else ""

                ic50_v, ic50_c = parse_affinity(get(IC50_COL))
                ki_v,   ki_c   = parse_affinity(get(KI_COL))
                kd_v,   kd_c   = parse_affinity(get(KD_COL))

                rows.append({
                    "BindingDB_ID":  get(LIGAND_ID),
                    "SMILES_raw":    get(SMILES_COL),
                    "IC50_nM":       ic50_v,
                    "IC50_censored": ic50_c,
                    "Ki_nM":         ki_v,
                    "Ki_censored":   ki_c,
                    "Kd_nM":         kd_v,
                    "Kd_censored":   kd_c,
                    "UniProt":       parts[UNIPROT_COL].strip(),
                })

    df = pd.DataFrame(rows)
    log.info(f"Total rows scanned      : {total_scanned:,}")
    log.info(f"RET rows extracted      : {len(df):,}")
    log.info(f"  With SMILES           : {(df.SMILES_raw != '').sum():,}")
    log.info(f"  With IC50             : {df.IC50_nM.notna().sum():,}")
    log.info(f"  With Ki               : {df.Ki_nM.notna().sum():,}")
    log.info(f"  With Kd               : {df.Kd_nM.notna().sum():,}")

    raw_path = DIR_RAW / "RET_raw.tsv"
    df.to_csv(raw_path, sep="\t", index=False)
    log.info(f"Raw data saved          : {raw_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — IC50 FILTER + DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════════════
def filter_and_dedup(df: pd.DataFrame, log) -> pd.DataFrame:
    log.info("=" * 70)
    log.info("SECTION 2: IC50 Filtering and Deduplication")
    log.info("=" * 70)

    n0 = len(df)
    df_ic = df[df.IC50_nM.notna() & (df.IC50_censored == False)].copy()
    log.info(f"Exact (uncensored) IC50 : {len(df_ic):,} / {n0:,}")

    df_ic = df_ic[df_ic.SMILES_raw.str.strip() != ""]
    log.info(f"After SMILES filter     : {len(df_ic):,}")

    df_ic = df_ic[df_ic.IC50_nM <= MAX_IC50_NM].copy()
    log.info(f"After IC50 <= 10,000 nM : {len(df_ic):,}")

    df_ic["pIC50"] = 9.0 - np.log10(df_ic["IC50_nM"].clip(lower=1e-3))

    grp = df_ic.groupby("SMILES_raw").agg(
        pIC50_median=("pIC50", "median"),
        pIC50_mean=("pIC50", "mean"),
        pIC50_std=("pIC50", "std"),
        IC50_nM_median=("IC50_nM", "median"),
        n_measurements=("pIC50", "count"),
    ).reset_index()
    grp.columns = ["SMILES_raw", "pIC50", "pIC50_mean", "pIC50_std",
                   "IC50_nM", "n_measurements"]

    log.info(f"After deduplication     : {len(grp):,} unique SMILES")
    log.info(f"  Multi-measurement     : {(grp.n_measurements > 1).sum():,}")
    log.info(f"  pIC50 range           : {grp.pIC50.min():.2f} – {grp.pIC50.max():.2f}")
    log.info(f"  pIC50 mean ± std      : {grp.pIC50.mean():.2f} ± {grp.pIC50.std():.2f}")
    return grp


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — SMILES STANDARDIZATION + PROPERTY COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def standardize_and_compute(df: pd.DataFrame, log) -> pd.DataFrame:
    log.info("=" * 70)
    log.info("SECTION 3: SMILES Standardization + Property Computation")
    log.info("=" * 70)

    n0 = len(df)
    records = []
    for _, row in df.iterrows():
        can_smi = standardize_smiles(row["SMILES_raw"])
        if can_smi is None:
            continue
        mol = Chem.MolFromSmiles(can_smi)
        if mol is None:
            continue
        props = compute_properties(mol)
        records.append({
            "SMILES_raw":     row["SMILES_raw"],
            "SMILES":         can_smi,
            "pIC50":          row["pIC50"],
            "IC50_nM":        row["IC50_nM"],
            "n_measurements": row["n_measurements"],
            **props,
        })

    df2 = pd.DataFrame(records)
    log.info(f"Input unique SMILES     : {n0:,}")
    log.info(f"After standardization   : {len(df2):,}")
    log.info(f"  Failed               : {n0 - len(df2):,}")

    before = len(df2)
    df2 = df2.sort_values("pIC50", ascending=False).drop_duplicates(
        subset="SMILES", keep="first"
    ).reset_index(drop=True)
    log.info(f"After canonical dedup   : {len(df2):,}  (removed {before - len(df2)})")

    interim_path = DIR_INTERIM / "RET_standardized.csv"
    df2.to_csv(interim_path, index=False)
    log.info(f"Interim saved           : {interim_path}")
    return df2


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — LIPINSKI + PAINS + TYPE I FILTER
# ══════════════════════════════════════════════════════════════════════════════
def apply_filters(df: pd.DataFrame, log) -> tuple:
    log.info("=" * 70)
    log.info("SECTION 4: Lipinski, PAINS, and Type I Pharmacophore Filters")
    log.info("=" * 70)

    pains_cat = setup_pains_filter()
    n0 = len(df)

    lip_pass  = []
    pains_hit = []
    type_i    = []
    type_ii   = []
    scaffold  = []
    activity  = []

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        lip_pass.append(lipinski_pass(row.to_dict()))
        pains_hit.append(is_pains(mol, pains_cat))
        type_i.append(detect_type_i(mol))
        type_ii.append(detect_type_ii(mol))
        scaffold.append(get_murcko_scaffold(mol))
        activity.append(assign_activity_class(row["pIC50"]))

    df = df.copy()
    df["Lipinski"]     = lip_pass
    df["PAINS"]        = pains_hit
    df["TypeI"]        = type_i
    df["TypeII"]       = type_ii
    df["Scaffold"]     = scaffold
    df["ActivityClass"] = activity

    log.info(f"Before filters          : {n0:,}")
    log.info(f"  Lipinski pass         : {sum(lip_pass):,}")
    log.info(f"  PAINS hits            : {sum(pains_hit):,}")
    log.info(f"  Type I pharmacophore  : {sum(type_i):,}")
    log.info(f"  Type II (urea)        : {sum(type_ii):,}")

    log.info("\nLipinski violations:")
    log.info(f"  HBD > {LIP_HBD}   : {(df['HBD']  > LIP_HBD).sum():,}")
    log.info(f"  HBA > {LIP_HBA}  : {(df['HBA']  > LIP_HBA).sum():,}")
    log.info(f"  MW > {LIP_MW}  : {(df['MW']   > LIP_MW).sum():,}")
    log.info(f"  LogP > {LIP_LOGP} : {(df['LogP'] > LIP_LOGP).sum():,}")

    df_filt = df[df["Lipinski"] & ~df["PAINS"] & df["TypeI"]].copy()
    log.info(f"\nAfter Lip + PAINS + TypeI: {len(df_filt):,}")

    n_inactive = (df_filt["ActivityClass"] == "Inactive (<6.5)").sum()
    df_filt    = df_filt[df_filt["ActivityClass"] != "Inactive (<6.5)"].copy()
    log.info(f"Removed pIC50 < 6.5    : {n_inactive:,}")
    log.info(f"Curated set (3 classes): {len(df_filt):,}")

    log.info("\nActivity class distribution:")
    ac_counts = df_filt["ActivityClass"].value_counts()
    for cls in ACT_LABELS:
        cnt = ac_counts.get(cls, 0)
        pct = 100 * cnt / len(df_filt) if len(df_filt) > 0 else 0
        log.info(f"  {cls:30s}: {cnt:5,}  ({pct:.1f}%)")

    n_scaffolds = df_filt["Scaffold"].nunique()
    log.info(f"\nUnique Bemis-Murcko scaffolds : {n_scaffolds:,}")
    log.info(f"Scaffold diversity ratio       : {n_scaffolds / len(df_filt):.3f}")

    fcounts = {
        "n_lipinski":   int(sum(lip_pass)),
        "n_pains":      int(sum(pains_hit)),
        "n_type_i_raw": int(sum(type_i)),
        "n_inactive":   int(n_inactive),
        "n_before_act": int(len(df_filt) + n_inactive),
    }
    return df_filt, fcounts


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — FIGURES (13 journal-quality)
# ══════════════════════════════════════════════════════════════════════════════
def make_figures(df_all: pd.DataFrame, df_filt: pd.DataFrame,
                 stats_dict: dict, log) -> None:
    log.info("=" * 70)
    log.info("SECTION 5: Generating 13 Journal-Quality Figures")
    log.info("=" * 70)

    present_cls  = [c for c in ACT_LABELS if (df_filt["ActivityClass"] == c).sum() >= 2]
    short_labels = {"Weakly Active (6.5–7)": "Weak",
                    "Active (7–8)": "Active",
                    "Highly Active (≥8)": "High"}

    # ── FIG 01: pIC50 distribution with KDE overlay ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, clr in PALETTE.items():
        sub = df_filt[df_filt["ActivityClass"] == cls]["pIC50"]
        if len(sub) > 1:
            ax.hist(sub, bins=50, color=clr, alpha=0.70, edgecolor="white",
                    linewidth=0.4, label=cls, zorder=3)
    kde_x = np.linspace(df_filt["pIC50"].min() - 0.3,
                        df_filt["pIC50"].max() + 0.3, 500)
    kde   = gaussian_kde(df_filt["pIC50"], bw_method=0.30)
    ax2   = ax.twinx()
    ax2.plot(kde_x, kde(kde_x), color=ACCENT, lw=2.2, alpha=0.85, zorder=5)
    ax2.set_ylabel("Density", color=ACCENT, fontsize=11)
    ax2.tick_params(axis="y", colors=ACCENT)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(ACCENT)
    ax2.spines["top"].set_visible(False)
    for thresh, ls, label in [(6.5, ":", "6.5 (Weakly Active)"),
                               (7.0, "--", "7.0 (Active)"),
                               (8.0, "-.", "8.0 (Highly Active)")]:
        ax.axvline(thresh, color="#333", ls=ls, lw=1.3, alpha=0.55, zorder=6)
    mu, sd = df_filt["pIC50"].mean(), df_filt["pIC50"].std()
    med    = df_filt["pIC50"].median()
    ax.text(0.97, 0.95,
            f"n = {len(df_filt):,}\nμ = {mu:.2f}  σ = {sd:.2f}\nMedian = {med:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#bbb", alpha=0.92),
            fontsize=9.5)
    ax.set_xlabel("pIC$_{50}$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("pIC$_{50}$ Distribution — RET Kinase Type I Inhibitors",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig01_pIC50_distribution", log)

    # ── FIG 02: Molecular Weight distribution ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, clr in PALETTE.items():
        sub = df_filt[df_filt["ActivityClass"] == cls]["MW"]
        if len(sub) > 1:
            ax.hist(sub, bins=50, color=clr, alpha=0.70, edgecolor="white",
                    linewidth=0.4, label=cls, zorder=3)
    ax.axvline(500, color="#555", ls="--", lw=1.5, alpha=0.65, label="500 Da (Lipinski)")
    ax.axvline(df_filt["MW"].mean(), color=ACCENT2, ls="-", lw=1.8, alpha=0.85,
               label=f"Mean ({df_filt['MW'].mean():.0f} Da)")
    ax.set_xlabel("Molecular Weight (Da)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Molecular Weight Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.text(0.97, 0.95,
            f"Min: {df_filt['MW'].min():.0f} Da\nMax: {df_filt['MW'].max():.0f} Da\n"
            f"Mean: {df_filt['MW'].mean():.0f} Da\nStd: {df_filt['MW'].std():.0f} Da",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#bbb", alpha=0.92),
            fontsize=9.5)
    fig.tight_layout()
    savefig(fig, "fig02_MW_distribution", log)

    # ── FIG 03: LogP distribution ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, clr in PALETTE.items():
        sub = df_filt[df_filt["ActivityClass"] == cls]["LogP"]
        if len(sub) > 1:
            ax.hist(sub, bins=50, color=clr, alpha=0.70, edgecolor="white",
                    linewidth=0.4, label=cls, zorder=3)
    ax.axvline(5.0, color="#555", ls="--", lw=1.5, alpha=0.65, label="LogP = 5 (Lipinski)")
    ax.axvline(df_filt["LogP"].mean(), color=ACCENT2, ls="-", lw=1.8, alpha=0.85,
               label=f"Mean ({df_filt['LogP'].mean():.2f})")
    ax.set_xlabel("LogP (Crippen)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("LogP Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig03_LogP_distribution", log)

    # ── FIG 04: TPSA distribution ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, clr in PALETTE.items():
        sub = df_filt[df_filt["ActivityClass"] == cls]["TPSA"]
        if len(sub) > 1:
            ax.hist(sub, bins=50, color=clr, alpha=0.70, edgecolor="white",
                    linewidth=0.4, label=cls, zorder=3)
    ax.axvline(140, color="#555", ls="--", lw=1.5, alpha=0.65, label="140 Å² (Veber)")
    ax.set_xlabel("TPSA (Ų)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Topological Polar Surface Area Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig04_TPSA_distribution", log)

    # ── FIG 05: QED distribution ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, clr in PALETTE.items():
        sub = df_filt[df_filt["ActivityClass"] == cls]["QED"]
        if len(sub) > 1:
            ax.hist(sub, bins=45, color=clr, alpha=0.70, edgecolor="white",
                    linewidth=0.4, label=cls, zorder=3)
    ax.axvline(0.5, color="#555", ls="--", lw=1.5, alpha=0.65, label="QED = 0.5")
    ax.set_xlabel("QED Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Quantitative Estimate of Drug-likeness (QED)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.text(0.03, 0.95,
            f"Mean: {df_filt['QED'].mean():.3f}\n"
            f"QED ≥ 0.5: {(df_filt['QED'] >= 0.5).sum():,} ({100*(df_filt['QED'] >= 0.5).mean():.0f}%)",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#bbb", alpha=0.92),
            fontsize=9.5)
    fig.tight_layout()
    savefig(fig, "fig05_QED_distribution", log)

    # ── FIG 06: Property Correlation Matrix (scatter + hexbin + hist) ─────
    props_for_corr = ["MW", "LogP", "TPSA", "HBD", "HBA", "QED", "pIC50"]
    prop_labels    = ["MW (Da)", "LogP", "TPSA (Ų)", "HBD", "HBA", "QED", "pIC$_{50}$"]
    n_p = len(props_for_corr)
    fig, axes = plt.subplots(n_p, n_p, figsize=(15, 14))
    fig.suptitle("Property Correlation Matrix — RET Kinase Type I Inhibitors",
                 fontsize=14, fontweight="bold", y=1.002)
    for i in range(n_p):
        for j in range(n_p):
            ax = axes[i][j]
            xi, yi = props_for_corr[j], props_for_corr[i]
            if i == j:
                ax.hist(df_filt[xi], bins=30, color=ACCENT2, alpha=0.75,
                        edgecolor="white", linewidth=0.3)
                ax.set_facecolor("#F8F9FA")
            elif i > j:
                for cls, clr in PALETTE.items():
                    sub = df_filt[df_filt["ActivityClass"] == cls]
                    ax.scatter(sub[xi], sub[yi], s=1.5, c=clr,
                               alpha=0.30, rasterized=True)
                r, p = stats.pearsonr(df_filt[xi], df_filt[yi])
                sign = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                ax.text(0.05, 0.93, f"r={r:.2f}{sign}",
                        transform=ax.transAxes, fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="#ccc", alpha=0.85))
            else:
                ax.hexbin(df_filt[xi], df_filt[yi], gridsize=18,
                          cmap="Blues", mincnt=1, linewidths=0.1)
            if i == n_p - 1:
                ax.set_xlabel(prop_labels[j], fontsize=7.5)
            if j == 0:
                ax.set_ylabel(prop_labels[i], fontsize=7.5)
            ax.tick_params(labelsize=6.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(color=c, label=l, alpha=0.8) for l, c in PALETTE.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=8.5,
               title="Activity Class", title_fontsize=8.5,
               framealpha=0.9, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    savefig(fig, "fig06_property_correlation_matrix", log)

    # ── FIG 07: Activity Class Donut Chart ────────────────────────────────
    ac_counts = df_filt["ActivityClass"].value_counts()
    ac_vals   = [ac_counts.get(c, 0) for c in ACT_LABELS]
    ac_colors = [PALETTE[c] for c in ACT_LABELS]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    wedges, texts, autotexts = ax.pie(
        ac_vals, labels=None, colors=ac_colors,
        autopct="%1.1f%%", startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2.0),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.text(0, 0, f"n = {len(df_filt):,}\ncompounds", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#333")
    legend_labels = [f"{l}  (n={v:,})" for l, v in zip(ACT_LABELS, ac_vals)]
    ax.legend(wedges, legend_labels, loc="lower center", fontsize=9,
              bbox_to_anchor=(0.5, -0.12), ncol=1, framealpha=0.9)
    ax.set_title("Activity Class Distribution", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    savefig(fig, "fig07_activity_class_donut", log)

    # ── FIG 08: Data Curation Funnel ──────────────────────────────────────
    stages = [
        ("Raw RET rows",                     stats_dict["n_raw"]),
        ("Has SMILES",                        stats_dict["n_has_smiles"]),
        ("Exact IC50 (no censoring)",         stats_dict["n_exact_ic50"]),
        ("IC50 ≤ 10,000 nM",                  stats_dict["n_pchembl5"]),
        ("After deduplication",              stats_dict["n_dedup"]),
        ("After SMILES standardization",     stats_dict["n_standardized"]),
        ("Lipinski filter (MW ≤ 550 Da)",    stats_dict["n_lipinski"]),
        ("PAINS-free + Type I pharmacophore",stats_dict["n_type_i"]),
        ("Curated (3 classes, pIC50 ≥ 6.5)",stats_dict["n_curated"]),
        ("Training (Active + Highly Active)",stats_dict["n_training"]),
    ]
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    blues  = plt.cm.Blues(np.linspace(0.30, 0.88, len(stages)))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.barh(range(len(stages)), values, color=blues,
                   edgecolor="white", linewidth=0.8, height=0.62)
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + max(values) * 0.01, i, f"{val:,}",
                va="center", ha="left", fontsize=9.5, fontweight="bold", color="#222")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Number of Compounds", fontsize=12)
    ax.set_title("Data Curation Funnel — RET Kinase Type I Inhibitors",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    savefig(fig, "fig08_data_curation_funnel", log)

    # ── FIG 09: Scaffold Diversity ────────────────────────────────────────
    sc_counts = df_filt["Scaffold"].value_counts().head(25)
    sc_vals   = sc_counts.values
    sc_labels = [f"Scaffold {i+1}" for i in range(len(sc_vals))]
    n_sc      = df_filt["Scaffold"].nunique()
    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(14, 5.5),
                                          gridspec_kw={"width_ratios": [2, 1]})
    blues2 = plt.cm.Blues(np.linspace(0.28, 0.88, len(sc_vals)))
    ax_bar.barh(sc_labels[::-1], sc_vals[::-1], color=blues2[::-1],
                edgecolor="white", linewidth=0.5, height=0.68)
    for i, v in enumerate(sc_vals[::-1]):
        ax_bar.text(v + sc_vals.max() * 0.01, i, str(v), va="center", fontsize=8.5)
    ax_bar.set_xlabel("Number of Compounds", fontsize=11)
    ax_bar.set_title("Top 25 Bemis–Murcko Scaffolds", fontsize=12, fontweight="bold")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.set_xlim(0, sc_vals.max() * 1.15)

    top10 = sc_counts.head(10).sum()
    rest  = len(df_filt) - top10
    ax_pie.pie(
        [top10, rest],
        labels=[f"Top 10\n({top10:,})", f"Remaining\n({rest:,})"],
        colors=[ACCENT2, "#CFD8DC"],
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=10),
    )
    ax_pie.set_title(f"Scaffold Coverage\n({n_sc:,} unique scaffolds)",
                     fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig09_scaffold_diversity", log)

    # ── FIG 10: pIC50 vs MW and pIC50 vs LogP ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for cls, clr in PALETTE.items():
        sub = df_filt[df_filt["ActivityClass"] == cls]
        axes[0].scatter(sub["MW"], sub["pIC50"], s=5, c=clr,
                        alpha=0.35, label=cls, rasterized=True)
        axes[1].scatter(sub["LogP"], sub["pIC50"], s=5, c=clr,
                        alpha=0.35, label=cls, rasterized=True)
    for ax, xcol, xlabel in [(axes[0], "MW", "Molecular Weight (Da)"),
                              (axes[1], "LogP", "LogP")]:
        r, p = stats.pearsonr(df_filt[xcol], df_filt["pIC50"])
        m, b = np.polyfit(df_filt[xcol], df_filt["pIC50"], 1)
        xrng = np.linspace(df_filt[xcol].min(), df_filt[xcol].max(), 200)
        ax.plot(xrng, m * xrng + b, color=ACCENT, lw=1.8, ls="--", alpha=0.75,
                label=f"Trend (r = {r:.2f})")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("pIC$_{50}$", fontsize=12)
        ax.legend(fontsize=8.5, markerscale=2.5, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_title("pIC$_{50}$ vs. Molecular Weight", fontsize=12, fontweight="bold")
    axes[1].set_title("pIC$_{50}$ vs. LogP", fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig10_pIC50_vs_MW_LogP", log)

    # ── FIG 11: SA Score and Fsp3 distributions ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(df_filt["SA"], bins=45, color=ACCENT2, alpha=0.78,
                 edgecolor="white", linewidth=0.4)
    axes[0].axvline(3.0, color="black", ls="--", lw=1.5, alpha=0.6,
                    label="SA = 3 (easy)")
    axes[0].axvline(6.0, color="red", ls="--", lw=1.5, alpha=0.6,
                    label="SA = 6 (complex)")
    axes[0].set_xlabel("Synthetic Accessibility Score", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Synthetic Accessibility Distribution", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].text(0.97, 0.95,
                 f"Mean: {df_filt['SA'].mean():.2f}\nSA ≤ 4: {(df_filt['SA']<=4).sum():,}",
                 transform=axes[0].transAxes, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbb", alpha=0.92),
                 fontsize=9.5)
    axes[1].hist(df_filt["Fsp3"], bins=40, color=ACCENT, alpha=0.78,
                 edgecolor="white", linewidth=0.4)
    axes[1].axvline(0.25, color="black", ls="--", lw=1.5, alpha=0.6,
                    label="Fsp3 = 0.25")
    axes[1].set_xlabel("Fraction sp³ Carbons (Fsp3)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Fsp3 Distribution", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig11_SA_Fsp3_distribution", log)

    # ── FIG 12: Violin plots by activity class ────────────────────────────
    props_vio = ["MW", "LogP", "TPSA", "QED", "pIC50"]
    prop_labs = ["MW (Da)", "LogP", "TPSA (Ų)", "QED", "pIC$_{50}$"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 5.2))
    for ax, prop, plab in zip(axes, props_vio, prop_labs):
        data   = [df_filt[df_filt["ActivityClass"] == cls][prop].dropna().values
                  for cls in present_cls]
        colors = [PALETTE[c] for c in present_cls]
        parts  = ax.violinplot(data, positions=list(range(len(present_cls))),
                               showmedians=True, showextrema=True)
        for pc, clr in zip(parts["bodies"], colors):
            pc.set_facecolor(clr)
            pc.set_alpha(0.72)
            pc.set_edgecolor("white")
        parts["cmedians"].set_colors("black")
        parts["cmedians"].set_linewidth(2.2)
        parts["cbars"].set_linewidths(0.8)
        parts["cmaxes"].set_linewidths(0.8)
        parts["cmins"].set_linewidths(0.8)
        ax.set_xticks(list(range(len(present_cls))))
        ax.set_xticklabels([short_labels[c] for c in present_cls],
                           fontsize=8.5, rotation=20)
        ax.set_ylabel(plab, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Property Distributions by Activity Class — RET Kinase",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig12_violin_by_activity_class", log)

    # ── FIG 13: Chemical Space PCA (Morgan fingerprints) ──────────────────
    log.info("  Computing Morgan fingerprints for PCA chemical space...")
    fps  = []
    idxs = []
    sample_n = min(len(df_filt), 3000)
    df_sample = df_filt.sample(n=sample_n, random_state=42) if len(df_filt) > sample_n else df_filt
    for i, (_, row) in enumerate(df_sample.iterrows()):
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol:
            fps.append(morgan_fp(mol))
            idxs.append(i)

    fp_arr = np.array(fps, dtype=np.float32)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(fp_arr)
    df_pca = df_sample.iloc[idxs].copy().reset_index(drop=True)
    df_pca["PC1"] = coords[:, 0]
    df_pca["PC2"] = coords[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: coloured by activity class
    ax = axes[0]
    for cls, clr in PALETTE.items():
        sub = df_pca[df_pca["ActivityClass"] == cls]
        ax.scatter(sub["PC1"], sub["PC2"], s=8, c=clr, alpha=0.45,
                   label=cls, rasterized=True)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax.set_title("Chemical Space PCA — Activity Class", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, markerscale=2, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: coloured by pIC50 (continuous)
    ax = axes[1]
    sc = ax.scatter(df_pca["PC1"], df_pca["PC2"], s=8,
                    c=df_pca["pIC50"], cmap="RdYlGn",
                    alpha=0.55, rasterized=True,
                    vmin=df_pca["pIC50"].quantile(0.05),
                    vmax=df_pca["pIC50"].quantile(0.95))
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("pIC$_{50}$", fontsize=11)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax.set_title("Chemical Space PCA — pIC$_{50}$ (continuous)", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(f"Morgan Fingerprint Chemical Space (n = {len(fps):,}, "
                 f"PCA variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig13_chemical_space_PCA", log)

    log.info(f"All 13 figures saved to: {DIR_FIG}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log = setup_logging()
    t0  = datetime.now()
    log.info("╔" + "═" * 68 + "╗")
    log.info("║  RET KINASE GENERATIVE DRUG DESIGN PIPELINE                   ║")
    log.info("║  Step 01: Data Curation and Preprocessing                      ║")
    log.info(f"║  Target : RET kinase (UniProt P07949)                          ║")
    log.info(f"║  Started: {t0.strftime('%Y-%m-%d %H:%M:%S')}                          ║")
    log.info("╚" + "═" * 68 + "╝")

    # 1. Extract
    df_raw        = extract_bindingdb(log)
    n_raw         = len(df_raw)
    n_has_smiles  = int((df_raw.SMILES_raw != "").sum())

    # 2. IC50 filter + dedup
    df_dedup     = filter_and_dedup(df_raw, log)
    n_exact_ic50 = int(df_raw[df_raw.IC50_nM.notna() & (df_raw.IC50_censored == False)].shape[0])
    n_pchembl5   = int(df_raw[df_raw.IC50_nM.notna() &
                               (df_raw.IC50_censored == False) &
                               (df_raw.IC50_nM <= MAX_IC50_NM)].shape[0])
    n_dedup      = len(df_dedup)

    # 3. Standardize + properties
    df_std       = standardize_and_compute(df_dedup, log)
    n_standardized = len(df_std)

    # 4. Filters
    df_filt, fcounts = apply_filters(df_std, log)
    n_lipinski  = fcounts["n_lipinski"]
    n_type_i    = int(len(df_filt) + fcounts["n_inactive"])
    n_curated   = len(df_filt)
    n_training  = int((df_filt["pIC50"] >= MIN_TRAIN_PIC50).sum())

    # 5. Save processed CSVs
    curated_path = DIR_PROC / "RET_curated.csv"
    df_filt.to_csv(curated_path, index=False)

    df_train     = df_filt[df_filt["pIC50"] >= MIN_TRAIN_PIC50].copy()
    train_path   = DIR_PROC / "RET_training.csv"
    df_train.to_csv(train_path, index=False)

    log.info("=" * 70)
    log.info(f"CURATED DATASET  (3 classes): {curated_path}")
    for cls in ACT_LABELS:
        cnt = (df_filt["ActivityClass"] == cls).sum()
        log.info(f"  {cls:30s}: {cnt:,}")
    log.info(f"  TOTAL                         : {n_curated:,}")
    log.info(f"")
    log.info(f"TRAINING DATASET ({MIN_TRAIN_PIC50}+): {train_path}")
    for cls in ["Active (7–8)", "Highly Active (≥8)"]:
        cnt = (df_train["ActivityClass"] == cls).sum()
        log.info(f"  {cls:30s}: {cnt:,}")
    log.info(f"  TOTAL                         : {len(df_train):,}")

    log.info("\nProperty Summary (curated dataset):")
    for prop in ["MW", "LogP", "TPSA", "HBD", "HBA", "QED", "SA", "pIC50"]:
        s = df_filt[prop]
        log.info(f"  {prop:8s}: mean={s.mean():.2f}  std={s.std():.2f}  "
                 f"min={s.min():.2f}  max={s.max():.2f}")

    # 6. Stats JSON
    stats_dict = {
        "target":            "RET kinase (P07949)",
        "disease":           "RET-fusion+ NSCLC, Medullary Thyroid Carcinoma",
        "source":            str(BINDINGDB),
        "run_date":          t0.strftime("%Y-%m-%d %H:%M:%S"),
        "n_raw":             int(n_raw),
        "n_has_smiles":      int(n_has_smiles),
        "n_exact_ic50":      int(n_exact_ic50),
        "n_pchembl5":        int(n_pchembl5),
        "n_dedup":           int(n_dedup),
        "n_standardized":    int(n_standardized),
        "n_lipinski":        int(n_lipinski),
        "n_type_i":          int(n_type_i),
        "n_curated":         int(n_curated),
        "n_training":        int(n_training),
        "n_final":           int(n_curated),
        "pIC50_mean":        float(df_filt["pIC50"].mean()),
        "pIC50_std":         float(df_filt["pIC50"].std()),
        "pIC50_min":         float(df_filt["pIC50"].min()),
        "pIC50_max":         float(df_filt["pIC50"].max()),
        "MW_mean":           float(df_filt["MW"].mean()),
        "LogP_mean":         float(df_filt["LogP"].mean()),
        "TPSA_mean":         float(df_filt["TPSA"].mean()),
        "QED_mean":          float(df_filt["QED"].mean()),
        "SA_mean":           float(df_filt["SA"].mean()),
        "n_unique_scaffolds":int(df_filt["Scaffold"].nunique()),
        "scaffold_diversity":float(df_filt["Scaffold"].nunique() / n_curated),
        "activity_classes":  {
            cls: int((df_filt["ActivityClass"] == cls).sum())
            for cls in ACT_LABELS
        },
        "filter_summary": {
            "raw_rows":       int(n_raw),
            "exact_ic50":     int(n_exact_ic50),
            "pchembl_ge5":    int(n_pchembl5),
            "after_dedup":    int(n_dedup),
            "after_std":      int(n_standardized),
            "after_lipinski": int(n_lipinski),
            "final_type_i":   int(n_type_i),
            "curated":        int(n_curated),
            "training":       int(n_training),
        },
    }
    json_path = DIR_RES / "01_curation_stats.json"
    with open(json_path, "w") as jf:
        json.dump(stats_dict, jf, indent=2)
    log.info(f"Stats JSON saved: {json_path}")

    # 7. Figures
    make_figures(df_std, df_filt, stats_dict, log)

    elapsed = (datetime.now() - t0).total_seconds()
    log.info("=" * 70)
    log.info(f"Step 01 complete in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    log.info(f"Curated  : {n_curated:,} compounds (3 classes, pIC50 >= 6.5)")
    log.info(f"Training : {n_training:,} compounds (Active + Highly Active, pIC50 >= 7.0)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
