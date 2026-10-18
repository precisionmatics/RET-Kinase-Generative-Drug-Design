"""
Generate journal-quality MD figures from cpptraj analysis outputs and OpenMM logs.
Outputs saved to figures/09_md/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import os, re

BASE    = "/home/stalin/Desktop/RET_Generative"
FIGDIR  = f"{BASE}/figures/09_md"
MD074   = f"{BASE}/md/RET_074/analysis"
MD098   = f"{BASE}/md/RET_098/analysis"
LOG074  = f"{BASE}/md/RET_074/RET_074_md.log"
LOG098  = f"{BASE}/md/RET_098/RET_098_md.log"
os.makedirs(FIGDIR, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────────
C074  = "#1F77B4"   # steel blue
C098  = "#D62728"   # crimson
CALPHA= 0.18
LWIDE = 1.8
LTHIN = 0.9

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
})

# ── loaders ────────────────────────────────────────────────────────────────────

def load_dat(path, col=1):
    """Load two-column cpptraj .dat file, return arrays (frames, values)."""
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

def load_log(logpath):
    """
    Parse OpenMM production log lines:
    percent,steps,time_ps,pot_kJ,kin_kJ,temp_K,vol_A3,dens,ns_per_day
    Returns dict of arrays (time_ns, pot_kJ, temp_K, dens)
    """
    time_ns, pot, temp, dens = [], [], [], []
    with open(logpath) as f:
        for line in f:
            line = line.strip()
            # production lines start with a percentage like "2.0%,"
            if not re.match(r'^\d+\.?\d*%,', line):
                continue
            parts = line.split(',')
            if len(parts) < 8:
                continue
            try:
                t_ps = float(parts[2])
                pe   = float(parts[3])
                te   = float(parts[4])
                T    = float(parts[5])
                d    = float(parts[7])
                time_ns.append(t_ps / 1000.0)
                pot.append(pe / 1000.0)   # kJ -> MJ for nicer axis
                temp.append(T)
                dens.append(d)
            except ValueError:
                pass
    return {
        "time_ns": np.array(time_ns),
        "pot_MJ":  np.array(pot),
        "temp_K":  np.array(temp),
        "dens":    np.array(dens),
    }

def running_mean(arr, w=20):
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode='same')

# ── load all data ──────────────────────────────────────────────────────────────
print("Loading trajectory analysis data...")

fr074, brmsd074 = load_dat(f"{MD074}/backbone_rmsd.dat", 1)
fr098, brmsd098 = load_dat(f"{MD098}/backbone_rmsd.dat", 1)
_,     lrmsd074 = load_dat(f"{MD074}/ligand_rmsd.dat",   1)
_,     lrmsd098 = load_dat(f"{MD098}/ligand_rmsd.dat",   1)
_,     rg074    = load_dat(f"{MD074}/rg.dat",             1)
_,     rg098    = load_dat(f"{MD098}/rg.dat",             1)
res074, rmsf074 = load_dat(f"{MD074}/rmsf_ca.dat",       1)
res098, rmsf098 = load_dat(f"{MD098}/rmsf_ca.dat",       1)

t074 = frames_to_ns(fr074)
t098 = frames_to_ns(fr098)

log074 = load_log(LOG074)
log098 = load_log(LOG098)

print(f"  RET_074: {len(t074)} frames, {t074[-1]:.1f} ns")
print(f"  RET_098: {len(t098)} frames, {t098[-1]:.1f} ns")
print(f"  RET_074 log: {len(log074['time_ns'])} energy points")
print(f"  RET_098 log: {len(log098['time_ns'])} energy points")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 4-panel: Backbone RMSD | Ligand RMSD | Rg | RMSF
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.subplots_adjust(hspace=0.38, wspace=0.32)

# ── Panel A: Backbone RMSD ────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(t074, brmsd074, color=C074, lw=LTHIN, alpha=0.35, zorder=1)
ax.plot(t074, running_mean(brmsd074, 30), color=C074, lw=LWIDE,
        label="RET_074 (96 ns)", zorder=2)
ax.plot(t098, brmsd098, color=C098, lw=LTHIN, alpha=0.35, zorder=1)
ax.plot(t098, running_mean(brmsd098, 30), color=C098, lw=LWIDE,
        label="RET_098 (100 ns)", zorder=2)
ax.axhline(2.0, color='gray', lw=0.8, ls='--', label="2.0 Å threshold")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Backbone RMSD (Å)")
ax.set_title("(A) Protein Backbone RMSD")
ax.legend(framealpha=0.85, loc='upper right')
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# ── Panel B: Ligand RMSD ─────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(t074, lrmsd074, color=C074, lw=LTHIN, alpha=0.35, zorder=1)
ax.plot(t074, running_mean(lrmsd074, 30), color=C074, lw=LWIDE,
        label="RET_074 (96 ns)", zorder=2)
ax.plot(t098, lrmsd098, color=C098, lw=LTHIN, alpha=0.35, zorder=1)
ax.plot(t098, running_mean(lrmsd098, 30), color=C098, lw=LWIDE,
        label="RET_098 (100 ns)", zorder=2)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Ligand RMSD (Å)")
ax.set_title("(B) Ligand RMSD")
ax.legend(framealpha=0.85, loc='upper right')
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# ── Panel C: Radius of Gyration ───────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(t074, rg074, color=C074, lw=LTHIN, alpha=0.35, zorder=1)
ax.plot(t074, running_mean(rg074, 30), color=C074, lw=LWIDE,
        label="RET_074", zorder=2)
ax.plot(t098, rg098, color=C098, lw=LTHIN, alpha=0.35, zorder=1)
ax.plot(t098, running_mean(rg098, 30), color=C098, lw=LWIDE,
        label="RET_098", zorder=2)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Radius of Gyration (Å)")
ax.set_title("(C) Radius of Gyration (Rg)")
ax.legend(framealpha=0.85, loc='upper right')
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.set_xlim(left=0)

# ── Panel D: Per-residue RMSF ─────────────────────────────────────────────────
ax = axes[1, 1]
ax.fill_between(res074, rmsf074, alpha=0.25, color=C074)
ax.plot(res074, rmsf074, color=C074, lw=1.2, label="RET_074")
ax.fill_between(res098, rmsf098, alpha=0.25, color=C098)
ax.plot(res098, rmsf098, color=C098, lw=1.2, label="RET_098")

# Annotate key residues (approximate residue indices from 6NEC)
key_res = {46: "LYS758", 92: "VAL804", 93: "GLU805", 95: "ALA807", 167: "ASP892"}
ymax_rmsf = max(rmsf074.max(), rmsf098.max())
for ri, name in key_res.items():
    if ri <= len(rmsf074):
        ax.axvline(ri, color='gray', lw=0.6, ls=':', alpha=0.7)
        ax.text(ri, ymax_rmsf * 0.92, name, fontsize=6.5, rotation=90,
                ha='right', va='top', color='#555555')

ax.set_xlabel("Residue Index")
ax.set_ylabel("C-alpha RMSF (Å)")
ax.set_title("(D) Per-Residue C-alpha RMSF")
ax.legend(framealpha=0.85, loc='upper right')
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.set_xlim(1, len(res074))
ax.set_ylim(bottom=0)

fig.suptitle("Molecular Dynamics Simulation — RET_074 and RET_098 Binding Stability",
             fontsize=13, fontweight='bold', y=1.01)

out1 = f"{FIGDIR}/09_01_md_stability.png"
fig.savefig(out1, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Potential Energy + Temperature + Density
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.subplots_adjust(wspace=0.35)

# Panel A: Potential Energy
ax = axes[0]
ax.plot(log074["time_ns"], log074["pot_MJ"], color=C074, lw=LTHIN, alpha=0.4, zorder=1)
ax.plot(log074["time_ns"], running_mean(log074["pot_MJ"], 5), color=C074,
        lw=LWIDE, label="RET_074", zorder=2)
ax.plot(log098["time_ns"], log098["pot_MJ"], color=C098, lw=LTHIN, alpha=0.4, zorder=1)
ax.plot(log098["time_ns"], running_mean(log098["pot_MJ"], 5), color=C098,
        lw=LWIDE, label="RET_098", zorder=2)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Potential Energy (MJ/mol)")
ax.set_title("(A) Potential Energy")
ax.legend(framealpha=0.85)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.set_xlim(left=0)

# Panel B: Temperature
ax = axes[1]
ax.plot(log074["time_ns"], log074["temp_K"], color=C074, lw=LTHIN, alpha=0.4, zorder=1)
ax.plot(log074["time_ns"], running_mean(log074["temp_K"], 5), color=C074,
        lw=LWIDE, label="RET_074", zorder=2)
ax.plot(log098["time_ns"], log098["temp_K"], color=C098, lw=LTHIN, alpha=0.4, zorder=1)
ax.plot(log098["time_ns"], running_mean(log098["temp_K"], 5), color=C098,
        lw=LWIDE, label="RET_098", zorder=2)
ax.axhline(300.0, color='black', lw=0.9, ls='--', label="Target 300 K")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Temperature (K)")
ax.set_title("(B) System Temperature")
ax.legend(framealpha=0.85)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.set_xlim(left=0)
ax.set_ylim(290, 310)

# Panel C: Density
ax = axes[2]
ax.plot(log074["time_ns"], log074["dens"], color=C074, lw=LTHIN, alpha=0.4, zorder=1)
ax.plot(log074["time_ns"], running_mean(log074["dens"], 5), color=C074,
        lw=LWIDE, label="RET_074", zorder=2)
ax.plot(log098["time_ns"], log098["dens"], color=C098, lw=LTHIN, alpha=0.4, zorder=1)
ax.plot(log098["time_ns"], running_mean(log098["dens"], 5), color=C098,
        lw=LWIDE, label="RET_098", zorder=2)
ax.axhline(1.0, color='black', lw=0.9, ls='--', label="Water density")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Density (g/cm³)")
ax.set_title("(C) System Density")
ax.legend(framealpha=0.85)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.set_xlim(left=0)

fig.suptitle("Molecular Dynamics Thermodynamic Properties — RET_074 and RET_098",
             fontsize=13, fontweight='bold', y=1.03)

out2 = f"{FIGDIR}/09_02_md_thermodynamics.png"
fig.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {out2}")

print("\nAll MD figures generated successfully.")
