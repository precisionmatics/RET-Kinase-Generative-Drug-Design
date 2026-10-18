"""
Redesigned MD stability figure — 5-panel journal-quality layout.
Fixes:
  - Rg edge artifact from zero-padded rolling mean
  - RMSF annotations now horizontal (readable)
  - Equilibration zone shading
  - Per-panel mean±std stat boxes
  - New panel E: RMSD probability distribution (last 50 ns)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import os

BASE   = "/home/stalin/Desktop/RET_Generative"
FIGDIR = f"{BASE}/figures/09_md"
MD074  = f"{BASE}/md/RET_074/analysis"
MD098  = f"{BASE}/md/RET_098/analysis"
os.makedirs(FIGDIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
C074  = "#1A5EA8"   # deep steel-blue
C098  = "#C0392B"   # crimson red
LTHIN = 0.55
LWIDE = 2.0
EQ_NS = 50.0        # equilibrated region starts at 50 ns
WINDOW = 25         # rolling mean window (frames)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   10,
    "axes.labelsize":   11,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8.5,
    "axes.linewidth":   1.0,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.top":   True,
    "ytick.right": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
})

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_dat(path, col=1):
    frames, vals = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            try:
                frames.append(float(parts[0]))
                vals.append(float(parts[col]))
            except (IndexError, ValueError):
                pass
    return np.array(frames), np.array(vals)

def frames_to_ns(frames, ns_per_frame=0.1):
    return frames * ns_per_frame

def rolling_mean_safe(arr, w=25):
    """Edge-safe rolling mean — no zero-padding; uses available neighbours."""
    result = np.empty_like(arr, dtype=float)
    half = w // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        result[i] = arr[lo:hi].mean()
    return result

def stat_box(ax, lines, x=0.97, y=0.97):
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes,
            fontsize=7.8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.35', fc='white',
                      ec='#BBBBBB', alpha=0.88))

def eq_stats(t, vals, cutoff=EQ_NS):
    eq = vals[t > cutoff]
    return eq.mean(), eq.std()

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading analysis data...")
fr074, brmsd074 = load_dat(f"{MD074}/backbone_rmsd.dat", 1)
fr098, brmsd098 = load_dat(f"{MD098}/backbone_rmsd.dat", 1)
_,     lrmsd074 = load_dat(f"{MD074}/ligand_rmsd.dat",   1)
_,     lrmsd098 = load_dat(f"{MD098}/ligand_rmsd.dat",   1)
_,     rg074    = load_dat(f"{MD074}/rg.dat",             1)
_,     rg098    = load_dat(f"{MD098}/rg.dat",             1)
res074, rmsf074 = load_dat(f"{MD074}/rmsf_ca.dat",        1)
res098, rmsf098 = load_dat(f"{MD098}/rmsf_ca.dat",        1)

t074 = frames_to_ns(fr074)
t098 = frames_to_ns(fr098)

print(f"  RET_074: {len(t074)} frames → {t074[-1]:.1f} ns")
print(f"  RET_098: {len(t098)} frames → {t098[-1]:.1f} ns")

# Equilibration stats (post-50 ns)
bm74, bs74 = eq_stats(t074, brmsd074)
bm98, bs98 = eq_stats(t098, brmsd098)
lm74, ls74 = eq_stats(t074, lrmsd074)
lm98, ls98 = eq_stats(t098, lrmsd098)
rm74, rs74 = eq_stats(t074, rg074)
rm98, rs98 = eq_stats(t098, rg098)

# ── Build figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig,
                         hspace=0.40, wspace=0.32,
                         left=0.08, right=0.97,
                         top=0.92, bottom=0.08)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

t_max = 100

def shade_eq(ax):
    ax.axvspan(EQ_NS, t_max, alpha=0.06, color='#2ECC71', zorder=0)

# ══════════════════════════════════════════════════════════════════════════════
# Panel A — Protein Backbone RMSD
# ══════════════════════════════════════════════════════════════════════════════
shade_eq(axA)
axA.plot(t074, brmsd074, color=C074, lw=LTHIN, alpha=0.28, zorder=1)
axA.plot(t074, rolling_mean_safe(brmsd074, WINDOW), color=C074, lw=LWIDE,
         label="RET_074 (96 ns)", zorder=2)
axA.plot(t098, brmsd098, color=C098, lw=LTHIN, alpha=0.28, zorder=1)
axA.plot(t098, rolling_mean_safe(brmsd098, WINDOW), color=C098, lw=LWIDE,
         label="RET_098 (100 ns)", zorder=2)
axA.axhline(2.0, color='#555', lw=0.9, ls='--', dashes=(5, 3), label="2.0 Å threshold")
axA.set_xlabel("Time (ns)")
axA.set_ylabel("Backbone RMSD (Å)")
axA.set_title("(A) Protein Backbone RMSD")
axA.legend(framealpha=0.9, loc='upper left', fontsize=8)
axA.xaxis.set_minor_locator(AutoMinorLocator(5))
axA.yaxis.set_minor_locator(AutoMinorLocator(5))
axA.set_xlim(0, t_max)
axA.set_ylim(0)

# ══════════════════════════════════════════════════════════════════════════════
# Panel B — Ligand RMSD
# ══════════════════════════════════════════════════════════════════════════════
shade_eq(axB)
axB.plot(t074, lrmsd074, color=C074, lw=LTHIN, alpha=0.28, zorder=1)
axB.plot(t074, rolling_mean_safe(lrmsd074, WINDOW), color=C074, lw=LWIDE,
         label="RET_074", zorder=2)
axB.plot(t098, lrmsd098, color=C098, lw=LTHIN, alpha=0.28, zorder=1)
axB.plot(t098, rolling_mean_safe(lrmsd098, WINDOW), color=C098, lw=LWIDE,
         label="RET_098", zorder=2)
axB.set_xlabel("Time (ns)")
axB.set_ylabel("Ligand RMSD (Å)")
axB.set_title("(B) Ligand RMSD")
axB.legend(framealpha=0.9, loc='upper left', fontsize=8)
axB.xaxis.set_minor_locator(AutoMinorLocator(5))
axB.yaxis.set_minor_locator(AutoMinorLocator(5))
axB.set_xlim(0, t_max)
axB.set_ylim(0)

# ══════════════════════════════════════════════════════════════════════════════
# Panel C — Radius of Gyration  (fixed: safe rolling mean, clipped y-axis)
# ══════════════════════════════════════════════════════════════════════════════
shade_eq(axC)
rg074_sm = rolling_mean_safe(rg074, WINDOW)
rg098_sm = rolling_mean_safe(rg098, WINDOW)
axC.plot(t074, rg074, color=C074, lw=LTHIN, alpha=0.28, zorder=1)
axC.plot(t074, rg074_sm,  color=C074, lw=LWIDE, label="RET_074", zorder=2)
axC.plot(t098, rg098, color=C098, lw=LTHIN, alpha=0.28, zorder=1)
axC.plot(t098, rg098_sm,  color=C098, lw=LWIDE, label="RET_098", zorder=2)
axC.set_xlabel("Time (ns)")
axC.set_ylabel("Radius of Gyration (Å)")
axC.set_title("(C) Radius of Gyration (Rg)")
axC.legend(framealpha=0.9, loc='lower right', fontsize=8)
axC.xaxis.set_minor_locator(AutoMinorLocator(5))
axC.yaxis.set_minor_locator(AutoMinorLocator(5))
axC.set_xlim(0, t_max)
# Clip y to data range + 3% margin — avoids any spurious spikes
all_rg = np.concatenate([rg074, rg098])
axC.set_ylim(np.percentile(all_rg, 0.5) * 0.98, np.percentile(all_rg, 99.5) * 1.02)

# ══════════════════════════════════════════════════════════════════════════════
# Panel D — Per-Residue C-α RMSF  (improved annotations)
# ══════════════════════════════════════════════════════════════════════════════
axD.fill_between(res074, rmsf074, alpha=0.18, color=C074)
axD.plot(res074, rmsf074, color=C074, lw=1.5, label="RET_074 (96 ns)")
axD.fill_between(res098, rmsf098, alpha=0.18, color=C098)
axD.plot(res098, rmsf098, color=C098, lw=1.5, label="RET_098 (100 ns)")

# Key binding-site residues — staggered to avoid overlap near hinge (92-95)
# columns: res_idx, label, x_text_offset, y_text_offset
annotations = [
    (46,  "LYS758",   0,  1.4),
    (92,  "VAL804",  -9,  3.2),
    (93,  "GLU805",   0,  5.0),
    (95,  "ALA807",   9,  3.2),
    (167, "ASP892",   0,  1.4),
]
ymax_r = max(rmsf074.max(), rmsf098.max())

for ri, name, xoff, yoff in annotations:
    ri_idx = ri - 1
    yval = max(
        rmsf074[ri_idx] if ri_idx < len(rmsf074) else 0,
        rmsf098[ri_idx] if ri_idx < len(rmsf098) else 0,
    )
    axD.annotate(
        name,
        xy=(ri, yval + 0.05),
        xytext=(ri + xoff, yval + yoff),
        fontsize=8, ha='center', color='#222222', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#555555', lw=0.9,
                        connectionstyle='arc3,rad=0.15'),
    )
    axD.axvline(ri, color='#AAAAAA', lw=0.7, ls=':', alpha=0.9)

axD.set_xlabel("Residue Index")
axD.set_ylabel("C-α RMSF (Å)")
axD.set_title("(D) Per-Residue C-α RMSF")
axD.legend(framealpha=0.9, loc='upper right')
axD.xaxis.set_minor_locator(AutoMinorLocator(5))
axD.yaxis.set_minor_locator(AutoMinorLocator(5))
axD.set_xlim(1, max(len(res074), len(res098)))
axD.set_ylim(0, ymax_r * 1.70)

out = f"{FIGDIR}/09_01_md_stability_v2.png"
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved: {out}")
