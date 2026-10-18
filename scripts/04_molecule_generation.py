"""
RET Kinase Generative Drug Design Pipeline
Step 04: Large-Scale Molecule Generation & Multi-Stage Filtering
Target: RET kinase (P07949) — RET-fusion+ NSCLC, Medullary Thyroid Carcinoma
Reference: ACS Omega 2026 (EGFR BiLSTM generative drug design)
"""

import os, sys, json, time, logging, warnings, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import (Descriptors, QED, AllChem, Draw,
                         rdMolDescriptors, rdDepictor, FilterCatalog)
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs, RDLogger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ─── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR  = "/home/stalin/Desktop/RET_Generative"
DATA_DIR     = os.path.join(PROJECT_DIR, "data/processed")
MODEL_DIR    = os.path.join(PROJECT_DIR, "models")
FIG_DIR      = os.path.join(PROJECT_DIR, "figures/04_generation")
LOG_DIR      = os.path.join(PROJECT_DIR, "logs")
RESULTS_DIR  = os.path.join(PROJECT_DIR, "results")

TRAIN_CSV    = os.path.join(DATA_DIR, "RET_training.csv")
PROTEIN_MAT  = os.path.join(DATA_DIR, "binding_site_matrix_norm.npy")
MODEL_PT     = os.path.join(MODEL_DIR, "best_model.pt")
TOKENIZER_J  = os.path.join(MODEL_DIR, "tokenizer.json")

OUT_CSV      = os.path.join(DATA_DIR, "RET_candidates.csv")
OUT_SDF      = os.path.join(DATA_DIR, "RET_candidates.sdf")

for d in [FIG_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("step04_ret")
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(LOG_DIR, "04_molecule_generation.log"), mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S"))
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(fh); log.addHandler(ch)

# ─── Config ────────────────────────────────────────────────────────────────────
N_GENERATE    = 10_000     # total raw attempts
N_CANDIDATES  = 100        # final candidates for docking
MAX_LEN       = 100
QED_MIN       = 0.40
MW_MAX        = 550.0      # slightly relaxed from 500 for RET macrocycles
LOGP_MAX      = 5.0
HBD_MAX       = 5
HBA_MAX       = 10
TC_DIVERSITY  = 0.70       # max Tanimoto within final set
FP_RADIUS     = 2
FP_NBITS      = 2048

TEMPERATURES  = [0.80, 0.90, 1.00, 1.10]   # diversity via temperature sampling
BUDGET        = [3000, 3000, 2500, 1500]    # raw attempts per temperature

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_TOKEN = "<PAD>"; SOS_TOKEN = "<SOS>"; EOS_TOKEN = "<EOS>"; UNK_TOKEN = "<UNK>"

# ─── Type I pharmacophore SMARTS ───────────────────────────────────────────────
TYPE_I_SMARTS = {
    "aminopyrimidine":    "[NH2,NH1;!$(NC=O)]c1ncccn1",
    "aminopyridine":      "[NH2,NH1;!$(NC=O)]c1ccncc1",
    "aminoquinoline":     "[NH2,NH1;!$(NC=O)]c1ccc2ncccc2c1",
    "aminoquinazoline":   "[NH2,NH1;!$(NC=O)]c1nc2ccccc2nc1",
    "anilinopyrimidine":  "Nc1ccnc(N)n1",
    "quinazoline":        "c1cnc2ccccc2n1",
    "pyrrolopyrimidine":  "c1cnc2[nH]ccc21",
    "pyrazolopyrimidine": "c1nn2ccncc2c1",
    "imidazopyridazine":  "c1cnc2ccnn21",
    "pyridopyrimidine":   "c1cnc2ncccc21",
    "naphthyridine":      "c1cncc2ncccc12",
    "phthalazine":        "c1cncc2ccccc12",
    "indazole_NH":        "c1ccc2[nH]ncc2c1",
    "benzimidazole_NH":   "c1cnc2ccccc2[nH]1",
    "pyrazolopyridine_NH":"c1cc2cn[nH]c2nc1",
    "aromatic_NH":        "a[nH]a",
    "pyrazole_NH":        "c1c[nH]nc1",
    "imidazole_NH":       "c1c[nH]cn1",
    "pyridinone":         "O=c1cc[nH]cc1",
    "pyrimidinone":       "O=c1[nH]ccnc1",
}
TYPE_I_MOLS = {k: Chem.MolFromSmarts(v) for k, v in TYPE_I_SMARTS.items()}

PALETTE = {
    "train":     "#2196F3",
    "generated": "#9E9E9E",
    "candidate": "#E53935",
    "valid":     "#4CAF50",
    "unique":    "#FF9800",
    "novel":     "#9C27B0",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.28, "grid.linestyle": "--",
    "figure.dpi": 100, "savefig.dpi": 300,
})


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CLASSES  (reproduced from Step 03 for standalone use)
# ══════════════════════════════════════════════════════════════════════════════
class SMILESTokenizer:
    def __init__(self):
        pass

    def decode(self, indices):
        out = []
        for i in indices:
            if i == self.eos_idx:
                break
            if i not in (self.pad_idx, self.sos_idx):
                out.append(self.idx2char.get(i, ""))
        return "".join(out)

    @classmethod
    def load(cls, path):
        tok = cls.__new__(cls)
        with open(path) as f:
            d = json.load(f)
        tok.char2idx   = d["char2idx"]
        tok.idx2char   = {int(k): v for k, v in d["idx2char"].items()}
        tok.vocab_size = len(tok.char2idx)
        tok.pad_idx    = tok.char2idx[PAD_TOKEN]
        tok.sos_idx    = tok.char2idx[SOS_TOKEN]
        tok.eos_idx    = tok.char2idx[EOS_TOKEN]
        tok.unk_idx    = tok.char2idx[UNK_TOKEN]
        return tok


class ProteinEncoder(nn.Module):
    def __init__(self, n_residues, residue_feat_dim, hidden_dim, n_heads, dropout, n_layers):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Linear(residue_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.self_attn  = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.to_h       = nn.Linear(hidden_dim, hidden_dim * n_layers)
        self.to_c       = nn.Linear(hidden_dim, hidden_dim * n_layers)
        self.n_layers   = n_layers
        self.hidden_dim = hidden_dim

    def forward(self, protein_matrix):
        x    = self.feat_proj(protein_matrix)
        x    = self.self_attn(x)
        pool = x.mean(dim=1)
        h0   = self.to_h(pool).view(-1, self.n_layers, self.hidden_dim)
        h0   = torch.tanh(h0).permute(1, 0, 2).contiguous()
        c0   = self.to_c(pool).view(-1, self.n_layers, self.hidden_dim)
        c0   = torch.tanh(c0).permute(1, 0, 2).contiguous()
        return x, (h0, c0)


class TargetAwareBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers,
                 n_residues, residue_feat_dim, n_heads, dropout):
        super().__init__()
        self.protein_enc = ProteinEncoder(
            n_residues, residue_feat_dim, hidden_dim, n_heads, dropout, n_layers)
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop    = nn.Dropout(dropout)
        self.embed_proj  = nn.Linear(embed_dim, hidden_dim)
        self.cross_attn  = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True)
        self.attn_norm   = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(
            input_size  = embed_dim + hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_drop = nn.Dropout(dropout)
        self.fc_out   = nn.Linear(hidden_dim, vocab_size)

    def generate(self, protein_matrix, tokenizer, max_len,
                 temperature=1.0, top_p=0.95, device=DEVICE):
        self.eval()
        with torch.no_grad():
            prot = protein_matrix.unsqueeze(0).to(device)
            prot_feats, hidden = self.protein_enc(prot)
            token = torch.tensor([[tokenizer.sos_idx]], dtype=torch.long, device=device)
            generated = []
            for _ in range(max_len):
                emb     = self.emb_drop(self.embedding(token))
                q       = self.embed_proj(emb)
                ctx, _  = self.cross_attn(q, prot_feats, prot_feats)
                ctx     = self.attn_norm(q + ctx)
                lstm_in = torch.cat([emb, ctx], dim=-1)
                lstm_out, hidden = self.lstm(lstm_in, hidden)
                logits  = self.fc_out(self.out_drop(self.out_norm(lstm_out)))
                logits  = logits[:, -1, :] / temperature
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove    = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                probs    = F.softmax(sorted_logits, dim=-1)
                next_tok = sorted_idx.gather(-1, torch.multinomial(probs, 1))
                if next_tok.item() == tokenizer.eos_idx:
                    break
                generated.append(next_tok.item())
                token = next_tok
        return tokenizer.decode(generated)


# ══════════════════════════════════════════════════════════════════════════════
# CHEMISTRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def canonical(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None

def mol_props(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return {
        "SMILES": Chem.MolToSmiles(mol),
        "MW":     Descriptors.MolWt(mol),
        "LogP":   Descriptors.MolLogP(mol),
        "TPSA":   CalcTPSA(mol),
        "QED":    QED.qed(mol),
        "HBD":    Descriptors.NumHDonors(mol),
        "HBA":    Descriptors.NumHAcceptors(mol),
        "RotB":   Descriptors.NumRotatableBonds(mol),
        "HeavyAt":mol.GetNumHeavyAtoms(),
        "Rings":  rdMolDescriptors.CalcNumRings(mol),
        "ArRings":rdMolDescriptors.CalcNumAromaticRings(mol),
    }

def get_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, FP_NBITS)

def passes_type1(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    for patt in TYPE_I_MOLS.values():
        if patt and mol.HasSubstructMatch(patt):
            return True
    return False

def get_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None

def diversity_select(df_in, max_tc=TC_DIVERSITY, n_max=N_CANDIDATES):
    """Greedy diversity selection: pick highest-QED first, then add only if max Tc < threshold."""
    df_sorted = df_in.sort_values("QED", ascending=False).reset_index(drop=True)
    fps       = [get_fp(s) for s in df_sorted["SMILES"]]
    selected  = []
    sel_fps   = []
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        fp = fps[i]
        if fp is None:
            continue
        if sel_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, sel_fps)
            if max(sims) >= max_tc:
                continue
        selected.append(row)
        sel_fps.append(fp)
        if len(selected) >= n_max:
            break
    return pd.DataFrame(selected).reset_index(drop=True)

def pairwise_tc(smiles_list):
    fps = [get_fp(s) for s in smiles_list]
    n   = len(fps)
    mat = np.zeros((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        mat[i] = sims
    return mat


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def generate_pool(model, tokenizer, protein_tensor, temperatures, budgets):
    all_smiles = []
    for temp, n in zip(temperatures, budgets):
        log.info(f"  Generating {n} SMILES at temperature={temp:.2f} ...")
        batch = []
        for i in range(n):
            smi = model.generate(protein_tensor, tokenizer, MAX_LEN,
                                  temperature=temp, top_p=0.95, device=DEVICE)
            batch.append(smi)
            if (i + 1) % 500 == 0:
                log.info(f"    [{temp:.2f}] {i+1}/{n} done")
        all_smiles.extend(batch)
        log.info(f"  → {n} generated at T={temp:.2f}")
    return all_smiles


# ══════════════════════════════════════════════════════════════════════════════
# FILTER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def filter_pipeline(raw_smiles, train_canon_set):
    funnel = {}

    funnel["Generated (raw)"] = len(raw_smiles)

    # Validity
    valid = [(s, canonical(s)) for s in raw_smiles]
    valid = [(s, c) for s, c in valid if c is not None]
    funnel["Valid SMILES"] = len(valid)

    # Unique canonical
    seen  = {}
    for s, c in valid:
        if c not in seen:
            seen[c] = s
    unique = list(seen.keys())
    funnel["Unique (canonical)"] = len(unique)

    # Novel (not in training set)
    novel = [s for s in unique if s not in train_canon_set]
    funnel["Novel (not in training)"] = len(novel)

    # Compute properties
    rows = [mol_props(s) for s in novel]
    rows = [r for r in rows if r]
    df   = pd.DataFrame(rows)

    # Lipinski Ro5 (relaxed MW for RET)
    ro5  = df[(df.MW <= MW_MAX) & (df.LogP <= LOGP_MAX) &
              (df.HBD <= HBD_MAX) & (df.HBA <= HBA_MAX)]
    funnel[f"Lipinski (MW≤{int(MW_MAX)}, Ro5)"] = len(ro5)

    # QED filter
    qed_ok = ro5[ro5.QED >= QED_MIN]
    funnel[f"QED ≥ {QED_MIN}"] = len(qed_ok)

    # PAINS
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    pains_free = []
    for smi in qed_ok["SMILES"].tolist():
        mol = Chem.MolFromSmiles(smi)
        if mol and not catalog.HasMatch(mol):
            pains_free.append(smi)
    funnel["PAINS-free"] = len(pains_free)

    # Type I pharmacophore
    type1_pass = [s for s in pains_free if passes_type1(s)]
    funnel["Type I pharmacophore"] = len(type1_pass)

    # Rebuild DataFrame for diversity selection
    rows2 = [mol_props(s) for s in type1_pass]
    rows2 = [r for r in rows2 if r]
    df2   = pd.DataFrame(rows2)

    # Diversity selection
    diverse = diversity_select(df2, max_tc=TC_DIVERSITY, n_max=N_CANDIDATES)
    funnel[f"Diverse (Tc<{TC_DIVERSITY})"] = len(diverse)
    funnel[f"Final candidates (top {N_CANDIDATES})"] = len(diverse)

    return funnel, diverse, df2


# ══════════════════════════════════════════════════════════════════════════════
# SDF OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
def write_sdf(df, path):
    from rdkit.Chem import SDWriter
    writer = SDWriter(path)
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue
        rdDepictor.Compute2DCoords(mol)
        mol.SetProp("_Name",    f"RET_cand_{row.name+1:03d}")
        mol.SetProp("SMILES",   row["SMILES"])
        mol.SetProp("MW",       f"{row['MW']:.2f}")
        mol.SetProp("LogP",     f"{row['LogP']:.3f}")
        mol.SetProp("TPSA",     f"{row['TPSA']:.2f}")
        mol.SetProp("QED",      f"{row['QED']:.4f}")
        mol.SetProp("HBD",      str(int(row["HBD"])))
        mol.SetProp("HBA",      str(int(row["HBA"])))
        writer.write(mol)
    writer.close()
    log.info(f"SDF written → {path}  ({len(df)} molecules)")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
def _save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {name}")


def fig_funnel(funnel):
    labels = list(funnel.keys())
    values = list(funnel.values())
    colors = plt.cm.Blues_r(np.linspace(0.2, 0.85, len(labels)))
    colors[-1] = [0.898, 0.224, 0.208, 1.0]   # highlight final candidates in red

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + values[0] * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Number of Molecules", fontsize=12)
    ax.set_title("RET Kinase Generative Pipeline — Molecule Filtering Funnel",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, values[0] * 1.12)
    fig.tight_layout()
    _save(fig, "fig01_filtering_funnel.png")


def fig_chemical_space(train_smiles, all_valid, candidates):
    log.info("  Building chemical space PCA ...")
    def batch_fps(smiles_list, label):
        fps = []
        for smi in smiles_list:
            fp = get_fp(smi)
            if fp:
                arr = np.zeros(FP_NBITS)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
        return np.array(fps) if fps else np.zeros((0, FP_NBITS))

    tr_arr  = batch_fps(train_smiles, "train")
    ge_arr  = batch_fps(all_valid,    "generated")
    ca_arr  = batch_fps(candidates,   "candidates")

    combined = np.vstack([tr_arr, ge_arr, ca_arr])
    pca  = PCA(n_components=2, random_state=42)
    emb  = pca.fit_transform(combined)

    n_tr = len(tr_arr)
    n_ge = len(ge_arr)
    n_ca = len(ca_arr)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(emb[:n_tr, 0], emb[:n_tr, 1],
               c=PALETTE["train"], s=18, alpha=0.55, label=f"Training ({n_tr})", zorder=2)
    ax.scatter(emb[n_tr:n_tr+n_ge, 0], emb[n_tr:n_tr+n_ge, 1],
               c=PALETTE["generated"], s=10, alpha=0.25, label=f"Generated valid ({n_ge})", zorder=1)
    ax.scatter(emb[n_tr+n_ge:, 0], emb[n_tr+n_ge:, 1],
               c=PALETTE["candidate"], s=60, alpha=0.90, label=f"Final candidates ({n_ca})",
               marker="*", zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)", fontsize=12)
    ax.set_title("Chemical Space — RET Kinase Generative Pipeline\n(Morgan FP, PCA)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig02_chemical_space_pca.png")


def fig_property_distributions(train_df, cand_df):
    props  = [("MW",   "Molecular Weight (Da)", [200, 600]),
              ("LogP", "LogP",                  [-2, 7]),
              ("TPSA", "TPSA (Å²)",             [40, 180]),
              ("QED",  "QED",                   [0, 1]),
              ("HBD",  "H-Bond Donors",         [0, 8]),
              ("HBA",  "H-Bond Acceptors",      [0, 12])]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for ax, (col, label, xlim) in zip(axes, props):
        tr_vals = train_df[col].dropna()
        ca_vals = cand_df[col].dropna()
        bins = np.linspace(xlim[0], xlim[1], 28)
        ax.hist(tr_vals, bins=bins, alpha=0.55, color=PALETTE["train"],
                label=f"Training (n={len(tr_vals)})", density=True, edgecolor="none")
        ax.hist(ca_vals, bins=bins, alpha=0.65, color=PALETTE["candidate"],
                label=f"Candidates (n={len(ca_vals)})", density=True, edgecolor="none")
        ax.axvline(tr_vals.mean(), color=PALETTE["train"], lw=1.8, ls="--", alpha=0.9)
        ax.axvline(ca_vals.mean(), color=PALETTE["candidate"], lw=1.8, ls="--", alpha=0.9)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{label}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
    fig.suptitle("Physicochemical Properties: Training vs Final Candidates — RET Kinase",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig03_property_distributions.png")


def fig_scaffold_analysis(cand_df, n_top=15):
    scaffolds = []
    for smi in cand_df["SMILES"].tolist():
        scaf = get_scaffold(smi)
        if scaf:
            scaffolds.append(scaf)

    cnt = Counter(scaffolds)
    top = cnt.most_common(n_top)
    labels  = [s[:35] + "…" if len(s) > 35 else s for s, _ in top]
    heights = [c for _, c in top]
    unique_scaf = len(cnt)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(heights)))
    bars = ax.barh(labels[::-1], heights[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, heights[::-1]):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", ha="left", fontsize=9, fontweight="bold")
    ax.set_xlabel("Frequency", fontsize=11)
    ax.set_title(f"Top {n_top} Bemis-Murcko Scaffolds\n(Total unique: {unique_scaf})",
                 fontsize=12, fontweight="bold")

    ax2 = axes[1]
    top5_scafs  = [s for s, _ in top[:5]]
    top5_labels = [f"#{i+1} (n={c})" for i, (_, c) in enumerate(top[:5])]
    mols = []
    for scaf_smi in top5_scafs:
        mol = Chem.MolFromSmiles(scaf_smi)
        if mol:
            mols.append(mol)
    ax2.axis("off")
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(320, 240),
                                   legends=top5_labels[:len(mols)], returnPNG=False)
        ax2.imshow(np.array(img))
        ax2.set_title("Top 5 Scaffolds (structures)", fontsize=12, fontweight="bold")

    fig.suptitle("Scaffold Diversity Analysis — Final RET Kinase Candidates",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig04_scaffold_analysis.png")


def fig_tanimoto_heatmap(cand_df):
    smiles = cand_df["SMILES"].tolist()
    log.info(f"  Computing {len(smiles)}×{len(smiles)} Tanimoto matrix ...")
    mat = pairwise_tc(smiles)

    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    from scipy.spatial.distance import squareform
    dist_mat = 1 - mat
    np.fill_diagonal(dist_mat, 0)
    condensed = squareform(dist_mat, checks=False)
    condensed = np.clip(condensed, 0, None)
    Z    = linkage(condensed, method="ward")
    order = leaves_list(Z)
    mat_ord = mat[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(mat_ord, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Tanimoto Similarity")
    ax.set_xlabel("Candidate index (clustered)", fontsize=11)
    ax.set_ylabel("Candidate index (clustered)", fontsize=11)
    ax.set_title(f"Pairwise Tanimoto Similarity — {len(smiles)} Final Candidates\n"
                 f"(Morgan FP, r=2, ward clustering; target diversity Tc<{TC_DIVERSITY})",
                 fontsize=12, fontweight="bold")
    np.fill_diagonal(mat, 0)
    mean_sim = mat.mean()
    ax.text(0.98, 0.02, f"Mean off-diagonal Tc: {mean_sim:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.tight_layout()
    _save(fig, "fig05_tanimoto_heatmap.png")


def fig_top_structures(cand_df, n=20):
    mols, legends = [], []
    for _, row in cand_df.head(n).iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol:
            mols.append(mol)
            legends.append(f"QED={row['QED']:.3f}\nMW={row['MW']:.0f}")
    if not mols:
        return
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(420, 300),
                                legends=legends, returnPNG=False)
    out = os.path.join(FIG_DIR, "fig06_top_candidates_grid.png")
    img.save(out, dpi=(300, 300))
    log.info("  Saved: fig06_top_candidates_grid.png")


def fig_summary_dashboard(funnel, cand_df, train_df):
    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.45)

    # Panel A: Funnel (compact)
    ax0 = fig.add_subplot(gs[:, 0])
    labels = list(funnel.keys())
    values = list(funnel.values())
    colors_f = plt.cm.Blues_r(np.linspace(0.2, 0.85, len(labels)))
    colors_f[-1] = [0.898, 0.224, 0.208, 1.0]
    bars = ax0.barh(labels, values, color=colors_f, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax0.text(bar.get_width() + values[0]*0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:,}", va="center", ha="left", fontsize=7.5)
    ax0.invert_yaxis()
    ax0.set_xlabel("Molecules", fontsize=9)
    ax0.set_title("Filtering\nFunnel", fontsize=11, fontweight="bold")
    ax0.tick_params(axis="y", labelsize=7)

    # Panel B: QED violin / boxplot — candidates vs training
    ax1 = fig.add_subplot(gs[0, 1])
    data_qed  = [train_df["QED"].dropna().tolist(), cand_df["QED"].dropna().tolist()]
    bp = ax1.boxplot(data_qed, patch_artist=True, widths=0.4,
                     medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], [PALETTE["train"], PALETTE["candidate"]]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax1.set_xticklabels(["Training", "Candidates"], fontsize=10)
    ax1.set_ylabel("QED", fontsize=10)
    ax1.set_title("QED Distribution", fontsize=11, fontweight="bold")

    # Panel C: MW boxplot
    ax2 = fig.add_subplot(gs[0, 2])
    data_mw  = [train_df["MW"].dropna().tolist(), cand_df["MW"].dropna().tolist()]
    bp2 = ax2.boxplot(data_mw, patch_artist=True, widths=0.4,
                      medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp2["boxes"], [PALETTE["train"], PALETTE["candidate"]]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax2.set_xticklabels(["Training", "Candidates"], fontsize=10)
    ax2.set_ylabel("MW (Da)", fontsize=10)
    ax2.set_title("MW Distribution", fontsize=11, fontweight="bold")

    # Panel D: LogP boxplot
    ax3 = fig.add_subplot(gs[0, 3])
    data_lp = [train_df["LogP"].dropna().tolist(), cand_df["LogP"].dropna().tolist()]
    bp3 = ax3.boxplot(data_lp, patch_artist=True, widths=0.4,
                      medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp3["boxes"], [PALETTE["train"], PALETTE["candidate"]]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax3.set_xticklabels(["Training", "Candidates"], fontsize=10)
    ax3.set_ylabel("LogP", fontsize=10)
    ax3.set_title("LogP Distribution", fontsize=11, fontweight="bold")

    # Panel E: Scaffold pie
    ax4 = fig.add_subplot(gs[1, 1])
    scaffolds = [get_scaffold(s) for s in cand_df["SMILES"].tolist()]
    scaffolds = [s for s in scaffolds if s]
    cnt       = Counter(scaffolds)
    n_singleton = sum(1 for v in cnt.values() if v == 1)
    n_multi     = len(cnt) - n_singleton
    sz_data     = [n_singleton, n_multi, len(cand_df) - len(scaffolds)]
    sz_labels   = [f"Singletons ({n_singleton})", f"Shared ({n_multi})", "No scaffold"]
    sz_colors   = ["#64B5F6", "#E53935", "#BDBDBD"]
    wedges, _ = ax4.pie([max(x,0) for x in sz_data], colors=sz_colors,
                         startangle=90, counterclock=False,
                         wedgeprops=dict(edgecolor="white", linewidth=1.2))
    ax4.legend(wedges, sz_labels, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, -0.15))
    ax4.set_title("Scaffold Distribution", fontsize=11, fontweight="bold")

    # Panel F: Stats table
    ax5 = fig.add_subplot(gs[1, 2:])
    ax5.axis("off")
    stats_rows = [
        ["Property", "Training set", "Final candidates"],
        ["n",          str(len(train_df)), str(len(cand_df))],
        ["MW (mean)",  f"{train_df['MW'].mean():.1f} Da",   f"{cand_df['MW'].mean():.1f} Da"],
        ["LogP (mean)",f"{train_df['LogP'].mean():.2f}",    f"{cand_df['LogP'].mean():.2f}"],
        ["TPSA (mean)",f"{train_df['TPSA'].mean():.1f} Å²", f"{cand_df['TPSA'].mean():.1f} Å²"],
        ["QED (mean)", f"{train_df['QED'].mean():.3f}",     f"{cand_df['QED'].mean():.3f}"],
        ["HBD (mean)", f"{train_df['HBD'].mean():.1f}",     f"{cand_df['HBD'].mean():.1f}"],
        ["HBA (mean)", f"{train_df['HBA'].mean():.1f}",     f"{cand_df['HBA'].mean():.1f}"],
        ["Unique scaffolds", "—", str(len(cnt))],
    ]
    tbl = ax5.table(cellText=stats_rows[1:], colLabels=stats_rows[0],
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1A237E"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#E8F5E9")
    ax5.set_title("Summary Statistics", fontsize=12, fontweight="bold")

    fig.suptitle("RET Kinase Generative Drug Design — Step 04 Summary Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)
    _save(fig, "fig07_summary_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  RET KINASE GENERATIVE DRUG DESIGN PIPELINE                 ║")
    log.info("║  Step 04: Molecule Generation & Filtering                   ║")
    log.info(f"║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                          ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 1: Loading Model & Tokenizer")
    log.info("=" * 70)
    tokenizer = SMILESTokenizer.load(TOKENIZER_J)
    log.info(f"Tokenizer: {tokenizer.vocab_size} tokens")

    ckpt = torch.load(MODEL_PT, map_location=DEVICE)
    cfg  = ckpt["cfg"]
    model = TargetAwareBiLSTM(
        vocab_size       = tokenizer.vocab_size,
        embed_dim        = cfg["embed_dim"],
        hidden_dim       = cfg["hidden_dim"],
        n_layers         = cfg["n_layers"],
        n_residues       = cfg["n_residues"],
        residue_feat_dim = cfg["protein_dim"],
        n_heads          = cfg["n_heads"],
        dropout          = 0.0,   # inference: no dropout
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Model loaded: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    protein_mat    = np.load(PROTEIN_MAT)
    protein_tensor = torch.tensor(protein_mat, dtype=torch.float32)

    # ── Load training set ─────────────────────────────────────────────────────
    train_df   = pd.read_csv(TRAIN_CSV)
    train_smi  = train_df["SMILES"].tolist()
    train_canon = set(filter(None, [canonical(s) for s in train_smi]))
    log.info(f"Training set: {len(train_smi)} SMILES  ({len(train_canon)} canonical)")

    # ── Generate ──────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info(f"SECTION 2: Generating {N_GENERATE:,} SMILES (4 temperatures)")
    log.info("=" * 70)
    raw_smiles = generate_pool(model, tokenizer, protein_tensor, TEMPERATURES, BUDGET)
    log.info(f"Total generated: {len(raw_smiles):,}")

    # ── Filter ────────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 3: Multi-Stage Filtering")
    log.info("=" * 70)
    funnel, cand_df, pre_diverse_df = filter_pipeline(raw_smiles, train_canon)
    log.info("Filtering funnel:")
    for stage, count in funnel.items():
        log.info(f"  {stage:<40s}: {count:,}")

    if len(cand_df) == 0:
        log.error("No candidates survived filtering — check thresholds.")
        sys.exit(1)

    # Add rank and scaffold
    cand_df = cand_df.sort_values("QED", ascending=False).reset_index(drop=True)
    cand_df["Rank"]     = cand_df.index + 1
    cand_df["Scaffold"] = cand_df["SMILES"].apply(get_scaffold)

    # ── Train DF for props ────────────────────────────────────────────────────
    from rdkit.Chem import Descriptors as D
    tr_rows = []
    for smi in train_smi:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            tr_rows.append({
                "SMILES": Chem.MolToSmiles(mol),
                "MW":     D.MolWt(mol), "LogP": D.MolLogP(mol),
                "TPSA":   CalcTPSA(mol), "QED":  QED.qed(mol),
                "HBD":    D.NumHDonors(mol), "HBA": D.NumHAcceptors(mol),
            })
    train_props_df = pd.DataFrame(tr_rows)

    # ── Save outputs ──────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 4: Saving Outputs")
    log.info("=" * 70)
    cols_out = ["Rank", "SMILES", "MW", "LogP", "TPSA", "QED",
                "HBD", "HBA", "RotB", "HeavyAt", "Rings", "ArRings", "Scaffold"]
    cand_df[cols_out].to_csv(OUT_CSV, index=False)
    log.info(f"CSV → {OUT_CSV}  ({len(cand_df)} rows)")
    write_sdf(cand_df, OUT_SDF)

    # ── Figures ───────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 5: Figures")
    log.info("=" * 70)
    fig_funnel(funnel)

    all_valid_smiles = []
    for smi in raw_smiles:
        c = canonical(smi)
        if c:
            all_valid_smiles.append(c)
    fig_chemical_space(list(train_canon), list(set(all_valid_smiles)), cand_df["SMILES"].tolist())
    fig_property_distributions(train_props_df, cand_df)
    fig_scaffold_analysis(cand_df)
    fig_tanimoto_heatmap(cand_df)
    fig_top_structures(cand_df, n=20)
    fig_summary_dashboard(funnel, cand_df, train_props_df)

    # ── Results JSON ──────────────────────────────────────────────────────────
    results = {
        "step":      "04_molecule_generation",
        "target":    "RET kinase (P07949)",
        "timestamp": datetime.now().isoformat(),
        "generation": {
            "n_raw":         len(raw_smiles),
            "temperatures":  TEMPERATURES,
            "budgets":       BUDGET,
        },
        "filtering_funnel": funnel,
        "candidates": {
            "n":          len(cand_df),
            "QED_mean":   round(float(cand_df["QED"].mean()), 4),
            "QED_max":    round(float(cand_df["QED"].max()), 4),
            "MW_mean":    round(float(cand_df["MW"].mean()), 2),
            "LogP_mean":  round(float(cand_df["LogP"].mean()), 3),
            "TPSA_mean":  round(float(cand_df["TPSA"].mean()), 2),
        },
        "files": {
            "csv": OUT_CSV,
            "sdf": OUT_SDF,
        }
    }
    results_path = os.path.join(RESULTS_DIR, "04_generation_stats.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results JSON → {results_path}")

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info(f"Step 04 complete in {elapsed/60:.1f} min")
    log.info(f"Final candidates  : {len(cand_df)}")
    log.info(f"CSV               : {OUT_CSV}")
    log.info(f"SDF               : {OUT_SDF}")
    log.info(f"QED (mean/max)    : {cand_df['QED'].mean():.3f} / {cand_df['QED'].max():.3f}")
    log.info(f"MW  (mean)        : {cand_df['MW'].mean():.1f} Da")
    log.info(f"LogP (mean)       : {cand_df['LogP'].mean():.2f}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
