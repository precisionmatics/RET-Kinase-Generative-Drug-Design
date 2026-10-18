"""
RET Kinase Generative Drug Design Pipeline
Step 03: Target-Aware BiLSTM Training
Architecture: 3-layer LSTM + multihead cross-attention conditioned on RET 20×13 feature matrix
Reference: ACS Omega 2026 (EGFR BiLSTM generative drug design)
Target: RET kinase (P07949) — RET-fusion+ NSCLC, Medullary Thyroid Carcinoma
"""

import os, sys, json, time, logging, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Draw, AllChem
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ─── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = "/home/stalin/Desktop/RET_Generative"
DATA_DIR    = os.path.join(PROJECT_DIR, "data/processed")
TRAIN_CSV   = os.path.join(DATA_DIR, "RET_training.csv")
PROTEIN_MAT = os.path.join(DATA_DIR, "binding_site_matrix_norm.npy")
MODEL_DIR   = os.path.join(PROJECT_DIR, "models")
CKPT_DIR    = os.path.join(PROJECT_DIR, "checkpoints")
FIG_DIR     = os.path.join(PROJECT_DIR, "figures/03_model_training")
LOG_DIR     = os.path.join(PROJECT_DIR, "logs")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
for d in [MODEL_DIR, CKPT_DIR, FIG_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("step03_ret")
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(LOG_DIR, "03_bilstm_training.log"), mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S"))
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(fh); log.addHandler(ch)

# ─── Hyperparameters ───────────────────────────────────────────────────────────
CFG = {
    # Architecture
    "embed_dim":    128,
    "hidden_dim":   512,
    "n_layers":     3,
    "n_heads":      8,
    "dropout":      0.30,
    "protein_dim":  13,
    "n_residues":   20,
    # Sequence
    "max_len":      100,     # slightly larger than FGFR1 (RET inhibitors can be longer)
    "augment_n":    12,
    # Training
    "epochs":       200,
    "batch_size":   64,
    "lr":           1e-3,
    "weight_decay": 1e-5,
    "grad_clip":    1.0,
    "tf_start":     1.0,
    "tf_end":       0.5,
    "val_split":    0.15,
    "patience":     30,
    "eval_every":   10,
    "ckpt_every":   20,
    "n_gen_eval":   200,
    "temperature":  1.0,
    "top_p":        0.95,
    "seed":         42,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

# ══════════════════════════════════════════════════════════════════════════════
# 1. SMILES TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════
class SMILESTokenizer:
    def __init__(self, smiles_list):
        chars    = sorted(set("".join(smiles_list)))
        specials = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        vocab    = specials + chars
        self.char2idx  = {c: i for i, c in enumerate(vocab)}
        self.idx2char  = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(vocab)
        self.pad_idx   = self.char2idx[PAD_TOKEN]
        self.sos_idx   = self.char2idx[SOS_TOKEN]
        self.eos_idx   = self.char2idx[EOS_TOKEN]
        self.unk_idx   = self.char2idx[UNK_TOKEN]

    def encode(self, smi):
        return ([self.sos_idx] +
                [self.char2idx.get(c, self.unk_idx) for c in smi] +
                [self.eos_idx])

    def decode(self, indices):
        out = []
        for i in indices:
            if i == self.eos_idx:
                break
            if i not in (self.pad_idx, self.sos_idx):
                out.append(self.idx2char.get(i, ""))
        return "".join(out)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"char2idx": self.char2idx,
                       "idx2char": {str(k): v for k, v in self.idx2char.items()}},
                      f, indent=2)
        log.info(f"Tokenizer saved → {path}")

    @classmethod
    def load(cls, path):
        tok = cls.__new__(cls)
        with open(path) as f:
            d = json.load(f)
        tok.char2idx  = d["char2idx"]
        tok.idx2char  = {int(k): v for k, v in d["idx2char"].items()}
        tok.vocab_size = len(tok.char2idx)
        tok.pad_idx   = tok.char2idx[PAD_TOKEN]
        tok.sos_idx   = tok.char2idx[SOS_TOKEN]
        tok.eos_idx   = tok.char2idx[EOS_TOKEN]
        tok.unk_idx   = tok.char2idx[UNK_TOKEN]
        return tok


# ══════════════════════════════════════════════════════════════════════════════
# 2. SMILES AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
def augment_smiles(smi, n=12, max_len=100):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi]
    results = set([smi])
    for _ in range(n * 6):
        try:
            s = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            if len(s) <= max_len:
                results.add(s)
        except Exception:
            pass
        if len(results) >= n:
            break
    return list(results)[:n]


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASET
# ══════════════════════════════════════════════════════════════════════════════
class RETDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, protein_matrix, max_len):
        self.tokenizer      = tokenizer
        self.protein_matrix = torch.tensor(protein_matrix, dtype=torch.float32)
        self.max_len        = max_len
        self.encoded        = []
        skipped = 0
        for smi in smiles_list:
            enc = tokenizer.encode(smi)
            if len(enc) <= max_len + 1:
                self.encoded.append(enc)
            else:
                skipped += 1
        if skipped:
            log.warning(f"  Skipped {skipped} sequences exceeding max_len={max_len}")

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        enc = self.encoded[idx]
        src = enc[:-1]
        tgt = enc[1:]
        pad = self.max_len - len(src)
        src = src + [self.tokenizer.pad_idx] * pad
        tgt = tgt + [self.tokenizer.pad_idx] * pad
        return (torch.tensor(src, dtype=torch.long),
                torch.tensor(tgt, dtype=torch.long),
                self.protein_matrix)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
class ProteinEncoder(nn.Module):
    """Encodes 20×13 binding-site matrix → residue keys/values + initial hidden state."""
    def __init__(self, n_residues, residue_feat_dim, hidden_dim, n_heads, dropout, n_layers):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Linear(residue_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.self_attn  = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.to_h       = nn.Linear(hidden_dim, hidden_dim * n_layers)
        self.to_c       = nn.Linear(hidden_dim, hidden_dim * n_layers)
        self.n_layers   = n_layers
        self.hidden_dim = hidden_dim

    def forward(self, protein_matrix):
        x    = self.feat_proj(protein_matrix)           # (B, 20, H)
        x    = self.self_attn(x)                        # (B, 20, H)
        pool = x.mean(dim=1)                            # (B, H)
        h0   = self.to_h(pool).view(-1, self.n_layers, self.hidden_dim)
        h0   = torch.tanh(h0).permute(1, 0, 2).contiguous()
        c0   = self.to_c(pool).view(-1, self.n_layers, self.hidden_dim)
        c0   = torch.tanh(c0).permute(1, 0, 2).contiguous()
        return x, (h0, c0)


class TargetAwareBiLSTM(nn.Module):
    """
    Target-conditioned SMILES generator.
    Protein features initialize the LSTM hidden state and supply keys/values
    for multihead cross-attention at every decoding step.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers,
                 n_residues, residue_feat_dim, n_heads, dropout):
        super().__init__()
        self.protein_enc = ProteinEncoder(
            n_residues, residue_feat_dim, hidden_dim, n_heads, dropout, n_layers)
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop   = nn.Dropout(dropout)
        self.embed_proj = nn.Linear(embed_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm  = nn.LayerNorm(hidden_dim)
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
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                n = p.size(0)
                p.data[n // 4: n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, src_tokens, protein_matrix, hidden=None):
        prot_feats, init_hidden = self.protein_enc(protein_matrix)
        if hidden is None:
            hidden = init_hidden
        emb      = self.emb_drop(self.embedding(src_tokens))   # (B, T, E)
        q        = self.embed_proj(emb)                         # (B, T, H)
        ctx, _   = self.cross_attn(q, prot_feats, prot_feats)  # (B, T, H)
        ctx      = self.attn_norm(q + ctx)
        lstm_in  = torch.cat([emb, ctx], dim=-1)               # (B, T, E+H)
        lstm_out, hidden = self.lstm(lstm_in, hidden)
        out      = self.out_drop(self.out_norm(lstm_out))
        logits   = self.fc_out(out)                            # (B, T, vocab)
        return logits, hidden

    def generate(self, protein_matrix, tokenizer, max_len,
                 temperature=1.0, top_p=0.95, device=DEVICE):
        self.eval()
        with torch.no_grad():
            prot  = protein_matrix.unsqueeze(0).to(device)
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
                probs     = F.softmax(sorted_logits, dim=-1)
                next_tok  = sorted_idx.gather(-1, torch.multinomial(probs, 1))
                if next_tok.item() == tokenizer.eos_idx:
                    break
                generated.append(next_tok.item())
                token = next_tok
        return tokenizer.decode(generated)


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def is_valid(smi):
    if not smi:
        return False
    return Chem.MolFromSmiles(smi) is not None

def evaluate_generated(gen_smiles, train_smiles_set):
    total  = len(gen_smiles)
    valid  = [s for s in gen_smiles if is_valid(s)]
    unique = list(set(valid))
    novel  = [s for s in unique if s not in train_smiles_set]
    v = len(valid)  / total      * 100 if total  > 0 else 0
    u = len(unique) / len(valid) * 100 if valid  else 0
    n = len(novel)  / len(unique)* 100 if unique else 0
    return {"total": total, "valid": len(valid), "unique": len(unique),
            "novel": len(novel), "validity": v, "uniqueness": u, "novelty": n}

def compute_properties(smiles_list):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            rows.append({
                "SMILES": smi,
                "MW":     Descriptors.MolWt(mol),
                "LogP":   Descriptors.MolLogP(mol),
                "TPSA":   CalcTPSA(mol),
                "QED":    QED.qed(mol),
                "HBD":    Descriptors.NumHDonors(mol),
                "HBA":    Descriptors.NumHAcceptors(mol),
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion, tokenizer, tf_ratio, device):
    model.train()
    total_loss, total_tokens = 0.0, 0
    for src, tgt, prot in loader:
        src, tgt, prot = src.to(device), tgt.to(device), prot.to(device)
        optimizer.zero_grad()
        if random.random() < tf_ratio:
            logits, _ = model(src, prot)
        else:
            B, T   = src.shape
            token  = src[:, :1]
            outputs = []
            hidden  = None
            for t in range(T):
                logit, hidden = model(token, prot, hidden)
                outputs.append(logit)
                token = logit[:, -1, :].argmax(dim=-1, keepdim=True)
            logits = torch.cat(outputs, dim=1)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        optimizer.step()
        mask          = (tgt != tokenizer.pad_idx)
        total_loss   += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()
    return total_loss / max(total_tokens, 1)

@torch.no_grad()
def val_epoch(model, loader, criterion, tokenizer, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for src, tgt, prot in loader:
        src, tgt, prot = src.to(device), tgt.to(device), prot.to(device)
        logits, _ = model(src, prot)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        mask          = (tgt != tokenizer.pad_idx)
        total_loss   += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()
    return total_loss / max(total_tokens, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = {"train": "#2196F3", "val": "#E53935",
           "valid": "#4CAF50", "unique": "#FF9800", "novel": "#9C27B0"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.30, "grid.linestyle": "--",
    "figure.dpi": 100, "savefig.dpi": 300,
})

def fig_loss_curves(train_losses, val_losses):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax = axes[0]
    ax.plot(epochs, train_losses, color=PALETTE["train"], lw=2, label="Train")
    ax.plot(epochs, val_losses,   color=PALETTE["val"],   lw=2, ls="--", label="Validation")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    ax2 = axes[1]
    w = 10
    if len(train_losses) >= w:
        ep2 = range(w, len(train_losses) + 1)
        ax2.plot(ep2, np.convolve(train_losses, np.ones(w)/w, "valid"),
                 color=PALETTE["train"], lw=2, label="Train (smoothed)")
        ax2.plot(ep2, np.convolve(val_losses,   np.ones(w)/w, "valid"),
                 color=PALETTE["val"],   lw=2, ls="--", label="Val (smoothed)")
    else:
        ax2.plot(epochs, train_losses, color=PALETTE["train"], lw=2)
        ax2.plot(epochs, val_losses,   color=PALETTE["val"],   lw=2, ls="--")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss (10-epoch avg)", fontsize=12)
    ax2.set_title("Smoothed Loss Curves", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    fig.suptitle("RET Kinase Target-Aware BiLSTM — Training Progress",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig01_loss_curves.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info("  Saved: fig01_loss_curves.png")

def fig_validity_curve(eval_records):
    if not eval_records:
        return
    ep   = [r["epoch"]      for r in eval_records]
    val  = [r["validity"]   for r in eval_records]
    uniq = [r["uniqueness"] for r in eval_records]
    nov  = [r["novelty"]    for r in eval_records]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep, val,  color=PALETTE["valid"],  lw=2.5, marker="o", ms=5, label="Validity (%)")
    ax.plot(ep, uniq, color=PALETTE["unique"], lw=2.5, marker="s", ms=5, label="Uniqueness (%)")
    ax.plot(ep, nov,  color=PALETTE["novel"],  lw=2.5, marker="^", ms=5, label="Novelty (%)")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Generative Quality Metrics Over Training — RET Kinase BiLSTM",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig02_validity_curve.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info("  Saved: fig02_validity_curve.png")

def fig_property_comparison(train_df, gen_df):
    props  = [("MW", "Molecular Weight (Da)"), ("LogP", "LogP"),
              ("TPSA", "TPSA (Å²)"), ("QED", "QED")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    for ax, (prop, label) in zip(axes, props):
        for name, df, c in [("Training", train_df, PALETTE["train"]),
                             ("Generated", gen_df, PALETTE["valid"])]:
            if prop in df.columns and len(df[prop].dropna()) > 0:
                vals = df[prop].dropna()
                ax.hist(vals, bins=30, alpha=0.55, color=c, label=name,
                        density=True, edgecolor="none")
                ax.axvline(vals.mean(), color=c, lw=1.5, ls="--", alpha=0.8)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{label} Distribution", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
    fig.suptitle("Property Distribution: Training vs Generated RET Inhibitors",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig03_property_comparison.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info("  Saved: fig03_property_comparison.png")

def fig_generated_grid(valid_smiles, n=20):
    mols, labels = [], []
    for smi in valid_smiles[:n]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            labels.append(smi[:28] + "…" if len(smi) > 28 else smi)
    if not mols:
        return
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(400, 280),
                                legends=labels, returnPNG=False)
    out = os.path.join(FIG_DIR, "fig04_generated_molecules_grid.png")
    img.save(out, dpi=(300, 300))
    log.info(f"  Saved: fig04_generated_molecules_grid.png  ({len(mols)} structures)")

def fig_training_summary(cfg, final_stats, eval_records):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40)

    # Architecture table
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    rows_data = [
        ["Parameter", "Value"],
        ["Target",           "RET kinase (P07949)"],
        ["Vocabulary size",  str(final_stats.get("vocab_size", "—"))],
        ["Embedding dim",    str(cfg["embed_dim"])],
        ["LSTM hidden dim",  str(cfg["hidden_dim"])],
        ["LSTM layers",      str(cfg["n_layers"])],
        ["Attn heads",       str(cfg["n_heads"])],
        ["Dropout",          str(cfg["dropout"])],
        ["Max seq len",      str(cfg["max_len"])],
        ["Batch size",       str(cfg["batch_size"])],
        ["Epochs trained",   str(final_stats.get("epochs_trained", "—"))],
        ["Best val loss",    f"{final_stats.get('best_val_loss', 0):.4f}"],
        ["Best epoch",       str(final_stats.get("best_epoch", "—"))],
        ["Train SMILES",     str(final_stats.get("n_train", "—"))],
        ["Aug SMILES",       str(final_stats.get("n_augmented", "—"))],
        ["Total params",     f"{final_stats.get('n_params', 0):,}"],
    ]
    tbl = ax0.table(cellText=rows_data[1:], colLabels=rows_data[0],
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1A237E"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F0F8FF")
    ax0.set_title("Model Architecture", fontsize=12, fontweight="bold", pad=8)

    # Generative quality bar
    ax1 = fig.add_subplot(gs[0, 1])
    if eval_records:
        last    = eval_records[-1]
        metrics = ["Validity", "Uniqueness", "Novelty"]
        values  = [last["validity"], last["uniqueness"], last["novelty"]]
        colors  = [PALETTE["valid"], PALETTE["unique"], PALETTE["novel"]]
        bars    = ax1.bar(metrics, values, color=colors, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax1.set_ylim(0, 115)
        ax1.set_ylabel("Percentage (%)", fontsize=11)
        ax1.set_title("Final Generative Quality", fontsize=12, fontweight="bold")

    # Loss curve
    ax2 = fig.add_subplot(gs[0, 2])
    tl, vl = final_stats.get("train_losses", []), final_stats.get("val_losses", [])
    if tl:
        ep = range(1, len(tl)+1)
        ax2.plot(ep, tl, color=PALETTE["train"], lw=1.5, label="Train")
        ax2.plot(ep, vl, color=PALETTE["val"],   lw=1.5, ls="--", label="Val")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
        ax2.set_title("Loss Curves", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)

    # Validity over epochs
    ax3 = fig.add_subplot(gs[1, :2])
    if eval_records:
        ep_v = [r["epoch"] for r in eval_records]
        ax3.plot(ep_v, [r["validity"]   for r in eval_records],
                 color=PALETTE["valid"],  lw=2, marker="o", ms=4, label="Validity")
        ax3.plot(ep_v, [r["uniqueness"] for r in eval_records],
                 color=PALETTE["unique"], lw=2, marker="s", ms=4, label="Uniqueness")
        ax3.plot(ep_v, [r["novelty"]    for r in eval_records],
                 color=PALETTE["novel"],  lw=2, marker="^", ms=4, label="Novelty")
        ax3.set_ylim(0, 105); ax3.set_xlabel("Epoch"); ax3.set_ylabel("(%)")
        ax3.set_title("Generative Quality Over Training", fontsize=12, fontweight="bold")
        ax3.legend(fontsize=10)

    # Property summary text
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    gen_props = final_stats.get("gen_properties", {})
    txt = "Generated RET Inhibitors\nProperty Summary\n\n"
    for k, v in gen_props.items():
        txt += f"{k}: {v}\n"
    ax4.text(0.08, 0.92, txt, transform=ax4.transAxes,
             fontsize=10, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F8F8", edgecolor="#CCCCCC"))
    ax4.set_title("Generated Properties", fontsize=12, fontweight="bold")

    fig.suptitle("RET Kinase Target-Aware BiLSTM — Training Summary",
                 fontsize=15, fontweight="bold", y=1.01)
    out = os.path.join(FIG_DIR, "fig05_training_summary.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info("  Saved: fig05_training_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  RET KINASE GENERATIVE DRUG DESIGN PIPELINE                 ║")
    log.info("║  Step 03: Target-Aware BiLSTM Training                      ║")
    log.info(f"║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                          ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")

    random.seed(CFG["seed"]); np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG["seed"])
        torch.backends.cudnn.deterministic = True

    log.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}  "
                 f"({torch.cuda.get_device_properties(0).total_memory//1024**2} MB VRAM)")

    # ── Data loading & augmentation ───────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 1: Data Loading & Augmentation")
    log.info("=" * 70)
    df = pd.read_csv(TRAIN_CSV)
    log.info(f"Training set      : {len(df)} compounds")
    log.info(f"  Active (7–8)    : {(df.ActivityClass=='Active (7–8)').sum()}")
    log.info(f"  Highly Active   : {(df.ActivityClass=='Highly Active (≥8)').sum()}")
    log.info(f"  pIC50 mean±std  : {df['pIC50'].mean():.2f} ± {df['pIC50'].std():.2f}")

    # Check max SMILES length
    smi_lens = [len(s) for s in df["SMILES"].tolist()]
    log.info(f"  SMILES len range: {min(smi_lens)}–{max(smi_lens)}  (max_len={CFG['max_len']})")

    raw_smiles  = df["SMILES"].tolist()
    train_set   = set(raw_smiles)
    protein_mat = np.load(PROTEIN_MAT)
    log.info(f"Protein matrix    : {protein_mat.shape}")

    log.info(f"Augmenting SMILES: {CFG['augment_n']}× per molecule ...")
    augmented = []
    for smi in raw_smiles:
        augmented.extend(augment_smiles(smi, n=CFG["augment_n"], max_len=CFG["max_len"]))
    random.shuffle(augmented)
    log.info(f"  Raw      : {len(raw_smiles)}")
    log.info(f"  Augmented: {len(augmented)}")

    tokenizer = SMILESTokenizer(augmented)
    log.info(f"Vocabulary size   : {tokenizer.vocab_size}")
    tokenizer.save(os.path.join(MODEL_DIR, "tokenizer.json"))

    # Train / val split
    random.seed(CFG["seed"])
    val_n   = max(int(len(raw_smiles) * CFG["val_split"]), 10)
    val_src = set(random.sample(raw_smiles, val_n))
    tr_src  = [s for s in raw_smiles if s not in val_src]

    tr_aug = []
    for smi in tr_src:
        tr_aug.extend(augment_smiles(smi, n=CFG["augment_n"], max_len=CFG["max_len"]))
    val_aug = list(val_src)

    log.info(f"Train: {len(tr_src)} molecules → {len(tr_aug)} augmented sequences")
    log.info(f"Val  : {len(val_src)} molecules")

    ds_train = RETDataset(tr_aug,  tokenizer, protein_mat, CFG["max_len"])
    ds_val   = RETDataset(val_aug, tokenizer, protein_mat, CFG["max_len"])
    dl_train = DataLoader(ds_train, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=2, pin_memory=True)
    log.info(f"Train batches: {len(dl_train)}  |  Val batches: {len(dl_val)}")

    # ── Model initialization ──────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 2: Model Initialization")
    log.info("=" * 70)
    model = TargetAwareBiLSTM(
        vocab_size       = tokenizer.vocab_size,
        embed_dim        = CFG["embed_dim"],
        hidden_dim       = CFG["hidden_dim"],
        n_layers         = CFG["n_layers"],
        n_residues       = CFG["n_residues"],
        residue_feat_dim = CFG["protein_dim"],
        n_heads          = CFG["n_heads"],
        dropout          = CFG["dropout"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total parameters  : {n_params:,}")
    log.info(f"  ProteinEncoder  : Linear(13→{CFG['hidden_dim']}) + TransformerEncoder(2L,{CFG['n_heads']}H)")
    log.info(f"  CrossAttention  : {CFG['n_heads']} heads")
    log.info(f"  LSTM            : {CFG['n_layers']}L × {CFG['hidden_dim']}H, in={CFG['embed_dim']+CFG['hidden_dim']}")
    log.info(f"  OutputFC        : {CFG['hidden_dim']} → {tokenizer.vocab_size}")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG["epochs"], eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    protein_tensor = torch.tensor(protein_mat, dtype=torch.float32)

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 3: Training")
    log.info("=" * 70)

    train_losses, val_losses = [], []
    eval_records             = []
    lr_history               = []
    best_val_loss            = float("inf")
    no_improve               = 0
    best_epoch               = 0

    for epoch in range(1, CFG["epochs"] + 1):
        tf_ratio = (CFG["tf_start"] -
                    (CFG["tf_start"] - CFG["tf_end"]) * (epoch / CFG["epochs"]))

        t_loss = train_epoch(model, dl_train, optimizer, criterion,
                             tokenizer, tf_ratio, DEVICE)
        v_loss = val_epoch(model, dl_val, criterion, tokenizer, DEVICE)
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        lr_history.append(optimizer.param_groups[0]["lr"])

        improved = v_loss < best_val_loss
        if improved:
            best_val_loss = v_loss
            best_epoch    = epoch
            no_improve    = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": v_loss,
                "cfg": CFG,
                "vocab_size": tokenizer.vocab_size,
            }, os.path.join(MODEL_DIR, "best_model.pt"))
        else:
            no_improve += 1

        if epoch % CFG["ckpt_every"] == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": v_loss, "cfg": CFG},
                       os.path.join(CKPT_DIR, f"ckpt_epoch{epoch:04d}.pt"))

        star = "★" if improved else " "
        log.info(f"Epoch {epoch:4d}/{CFG['epochs']} | "
                 f"Train: {t_loss:.4f} | Val: {v_loss:.4f} | "
                 f"LR: {lr_history[-1]:.6f} | TF: {tf_ratio:.2f} {star}")

        if epoch % CFG["eval_every"] == 0 or epoch == CFG["epochs"]:
            model.eval()
            gen_smiles = [
                model.generate(protein_tensor, tokenizer, CFG["max_len"],
                               CFG["temperature"], CFG["top_p"], DEVICE)
                for _ in range(CFG["n_gen_eval"])
            ]
            stats = evaluate_generated(gen_smiles, train_set)
            stats["epoch"] = epoch
            eval_records.append(stats)
            log.info(f"  → Generated {stats['total']} | "
                     f"Valid: {stats['validity']:.1f}% | "
                     f"Unique: {stats['uniqueness']:.1f}% | "
                     f"Novel: {stats['novelty']:.1f}%")

        if no_improve >= CFG["patience"]:
            log.info(f"Early stopping at epoch {epoch} "
                     f"(no improvement for {CFG['patience']} epochs)")
            break

    # ── Post-training ─────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 4: Final Evaluation")
    log.info("=" * 70)

    ckpt = torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(f"Best model: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    log.info("Generating 1,000 SMILES for final evaluation ...")
    final_gen = [
        model.generate(protein_tensor, tokenizer, CFG["max_len"],
                       CFG["temperature"], CFG["top_p"], DEVICE)
        for _ in range(1000)
    ]
    final_stats_gen = evaluate_generated(final_gen, train_set)
    log.info("Final evaluation (n=1000):")
    for k, v in final_stats_gen.items():
        if k != "epoch":
            log.info(f"  {k}: {v}")

    valid_gen = [s for s in final_gen if is_valid(s)]
    gen_df    = compute_properties(list(set(valid_gen)))
    train_df  = compute_properties(raw_smiles)

    gen_props_summary = {}
    if not gen_df.empty:
        for col in ["MW", "LogP", "TPSA", "QED"]:
            gen_props_summary[col] = f"{gen_df[col].mean():.2f} ± {gen_df[col].std():.2f}"

    gen_out = os.path.join(DATA_DIR, "RET_generated_raw.csv")
    pd.DataFrame({"SMILES": list(set(valid_gen))}).to_csv(gen_out, index=False)
    log.info(f"Generated SMILES saved → {gen_out}  ({len(set(valid_gen))} rows)")

    # ── Figures ───────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SECTION 5: Figures")
    log.info("=" * 70)
    fig_loss_curves(train_losses, val_losses)
    fig_validity_curve(eval_records)
    fig_property_comparison(train_df, gen_df)
    fig_generated_grid(list(set(valid_gen))[:20])

    final_summary_stats = {
        "vocab_size":    tokenizer.vocab_size,
        "n_train":       len(tr_src),
        "n_augmented":   len(tr_aug),
        "n_params":      n_params,
        "epochs_trained":len(train_losses),
        "best_val_loss": best_val_loss,
        "best_epoch":    best_epoch,
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "gen_properties":gen_props_summary,
    }
    fig_training_summary(CFG, final_summary_stats, eval_records)

    # ── Results JSON ──────────────────────────────────────────────────────────
    results = {
        "step":      "03_bilstm_training",
        "target":    "RET kinase (P07949)",
        "timestamp": datetime.now().isoformat(),
        "config":    CFG,
        "model": {
            "n_params":       n_params,
            "vocab_size":     tokenizer.vocab_size,
            "best_epoch":     best_epoch,
            "best_val_loss":  round(best_val_loss, 6),
            "epochs_trained": len(train_losses),
        },
        "data": {
            "n_raw_train":  len(tr_src),
            "n_augmented":  len(tr_aug),
            "n_val":        len(val_aug),
        },
        "generation_1000": {
            k: (round(v, 2) if isinstance(v, float) else v)
            for k, v in final_stats_gen.items() if k != "epoch"
        },
        "gen_properties": gen_props_summary,
        "train_losses":   [round(x, 6) for x in train_losses],
        "val_losses":     [round(x, 6) for x in val_losses],
        "eval_records":   eval_records,
    }
    results_path = os.path.join(RESULTS_DIR, "03_bilstm_training_stats.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results JSON → {results_path}")

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info(f"Step 03 complete in {elapsed/60:.1f} min")
    log.info(f"Best model    : {MODEL_DIR}/best_model.pt")
    log.info(f"Best epoch    : {best_epoch}  |  Val loss: {best_val_loss:.4f}")
    log.info(f"Generated CSV : {gen_out}")
    log.info(f"Validity      : {final_stats_gen['validity']:.1f}%  |  "
             f"Uniqueness: {final_stats_gen['uniqueness']:.1f}%  |  "
             f"Novelty: {final_stats_gen['novelty']:.1f}%")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
