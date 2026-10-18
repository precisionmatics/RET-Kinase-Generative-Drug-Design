#!/usr/bin/env python3
"""
Step 07: GNINA CNN Docking — Top 20 RET Kinase Candidates
GNINA v1.3.2 — independent docking with CNN scoring (cnn_scoring=all)

Input : best AutoDock-GPU docked poses (mk_export.py from DLG) as starting coords
Output: minimizedAffinity (Vina), CNNscore, CNNaffinity for each candidate
        5-method consensus: AutoDock + Boltz-2 + MM-PBSA + MM-GBSA + GNINA
"""

import os
import re
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path("/home/stalin/Desktop/RET_Generative")
GNINA_DIR   = ROOT / "docking" / "gnina"
FIG_DIR     = ROOT / "figures" / "07_gnina"
RESULTS_DIR = ROOT / "results"
DATA_DIR    = ROOT / "data" / "processed"

RECEPTOR    = ROOT / "docking" / "receptor" / "6NEC_chainA.pdb"
DLG_DIR     = ROOT / "docking" / "results"
MK_EXPORT   = Path("/home/stalin/miniforge3/bin/mk_export.py")

GNINA_BIN   = Path("/home/stalin/Downloads/Enamine/GIPR/gnina")
CUDNN_LIB   = "/home/stalin/miniforge3/lib/python3.13/site-packages/nvidia/cudnn/lib"

DOCK_CSV    = DATA_DIR / "RET_docking_results.csv"
BOLTZ_CSV   = DATA_DIR / "RET_boltz2_affinity.csv"
MMPBSA_CSV  = DATA_DIR / "RET_mmpbsa_results.csv"

# Docking box — same as AutoDock-GPU
CX, CY, CZ = 5.72, 4.62, -3.27
SX, SY, SZ = 22.5, 22.5, 22.5

TOP_N = 20

for d in [GNINA_DIR, FIG_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def gnina_env():
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{CUDNN_LIB}:{existing}" if existing else CUDNN_LIB
    return env


def run(cmd, log_path=None, check=True, env=None):
    result = subprocess.run(
        cmd, shell=True, env=env or os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if log_path:
        Path(log_path).write_text(result.stdout)
    if check and result.returncode != 0:
        tail = result.stdout[-3000:] if result.stdout else "(no output)"
        raise RuntimeError(f"Command failed (rc={result.returncode}):\n{cmd}\n\n{tail}")
    return result.stdout


def parse_gnina_sdf(sdf_path):
    """Extract minimizedAffinity, CNNscore, CNNaffinity from GNINA output SDF."""
    text = Path(sdf_path).read_text()
    affinity = cnn_score = cnn_affinity = None

    m = re.search(r'>.*?minimizedAffinity.*?\n([-\d.]+)', text)
    if m:
        affinity = float(m.group(1))

    m = re.search(r'>.*?CNNscore.*?\n([-\d.]+)', text)
    if m:
        cnn_score = float(m.group(1))

    m = re.search(r'>.*?CNNaffinity.*?\n([-\d.]+)', text)
    if m:
        cnn_affinity = float(m.group(1))

    return affinity, cnn_score, cnn_affinity


# ── Docking ────────────────────────────────────────────────────────────────────
def dock_one(cand_id, dlg_file):
    """
    1. Extract best AutoDock-GPU pose from DLG → SDF (docked coordinates)
    2. Run GNINA independent docking with CNN scoring from that starting pose
    Returns dict with vina_affinity, cnn_score, cnn_affinity.
    """
    wdir     = GNINA_DIR / cand_id
    wdir.mkdir(exist_ok=True)

    cache = wdir / "gnina_result.json"
    if cache.exists():
        print(f"  [skip] {cand_id} — cached")
        return json.loads(cache.read_text())

    dlg_path  = DLG_DIR / dlg_file
    input_sdf = wdir / "docked_pose.sdf"
    out_sdf   = wdir / "gnina_out.sdf"
    log_file  = wdir / "gnina.log"

    # Extract docked pose from DLG
    run(f"python {MK_EXPORT} {dlg_path} -s {input_sdf}",
        log_path=wdir / "mk_export.log")

    # GNINA docking with CNN scoring
    cmd = (
        f"{GNINA_BIN} "
        f"--receptor {RECEPTOR} "
        f"--ligand {input_sdf} "
        f"--center_x {CX} --center_y {CY} --center_z {CZ} "
        f"--size_x {SX} --size_y {SY} --size_z {SZ} "
        f"--out {out_sdf} "
        f"--log {log_file} "
        f"--num_modes 9 "
        f"--exhaustiveness 16 "
        f"--cnn_scoring rescore "
        f"--no_gpu"
    )
    print(f"  {cand_id}: GNINA docking...")
    run(cmd, log_path=wdir / "gnina_run.log", env=gnina_env())

    # Parse best pose (first in SDF)
    vina, cnn_s, cnn_a = parse_gnina_sdf(out_sdf)

    result = {
        "CandID":       cand_id,
        "vina_affinity": vina,
        "cnn_score":    cnn_s,
        "cnn_affinity": cnn_a,
    }
    cache.write_text(json.dumps(result, indent=2))
    print(f"  ✓ {cand_id}: Vina={vina:.2f}  CNNscore={cnn_s:.3f}  CNNaffinity={cnn_a:.2f}")
    return result


# ── Figures ────────────────────────────────────────────────────────────────────
def generate_figures(df):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.linewidth": 1.2, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 300,
    })
    pal = plt.cm.Set2.colors

    # Fig 1 — CNNscore bar (binding probability)
    df_s = df.sort_values("cnn_score", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [pal[0] if v > 0.7 else pal[1] if v > 0.5 else pal[3]
              for v in df_s["cnn_score"]]
    ax.bar(df_s["CandID"], df_s["cnn_score"], color=colors, edgecolor="k", lw=0.5)
    ax.axhline(0.7, color="red", ls="--", lw=1.0, label="CNN score 0.70 threshold")
    ax.axhline(0.5, color="orange", ls="--", lw=0.8, label="CNN score 0.50")
    ax.set_ylabel("GNINA CNN Score (binding probability)", fontsize=12)
    ax.set_title("GNINA CNN Score — Top 20 RET Candidates", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=9)
    patches = [mpatches.Patch(color=pal[0], label="> 0.70 (strong binder)"),
               mpatches.Patch(color=pal[1], label="0.50–0.70"),
               mpatches.Patch(color=pal[3], label="< 0.50")]
    ax.legend(handles=patches, fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_01_cnn_score_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: 07_01_cnn_score_bar.png")

    # Fig 2 — CNNaffinity bar
    df_a = df.sort_values("cnn_affinity")
    fig, ax = plt.subplots(figsize=(12, 6))
    colors2 = [pal[2] if v < -8 else pal[1] if v < -7 else pal[3]
               for v in df_a["cnn_affinity"]]
    ax.barh(df_a["CandID"][::-1], df_a["cnn_affinity"][::-1],
            color=colors2[::-1], edgecolor="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("GNINA CNN Affinity (kcal/mol)", fontsize=12)
    ax.set_title("GNINA CNN Affinity — Top 20 RET Candidates", fontsize=13, fontweight="bold")
    ax.invert_xaxis()
    patches2 = [mpatches.Patch(color=pal[2], label="< −8 kcal/mol"),
                mpatches.Patch(color=pal[1], label="−8 to −7"),
                mpatches.Patch(color=pal[3], label="> −7")]
    ax.legend(handles=patches2, fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_02_cnn_affinity_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: 07_02_cnn_affinity_bar.png")

    # Fig 3 — AutoDock vs GNINA Vina vs CNNaffinity (3-panel)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pairs = [
        ("DockingScore", "vina_affinity", "AutoDock-GPU", "GNINA Vina", "viridis"),
        ("DockingScore", "cnn_affinity",  "AutoDock-GPU", "CNN Affinity", "plasma"),
        ("vina_affinity","cnn_affinity",  "GNINA Vina",   "CNN Affinity", "cividis"),
    ]
    for ax, (xc, yc, xl, yl, cmap) in zip(axes, pairs):
        sc = ax.scatter(df[xc], df[yc], c=df["cnn_score"], cmap=cmap,
                        s=75, edgecolors="k", lw=0.5, vmin=0, vmax=1, zorder=3)
        for _, r in df.iterrows():
            ax.annotate(r["CandID"], (r[xc], r[yc]),
                        xytext=(3, 3), textcoords="offset points", fontsize=6, color="gray")
        try:
            r_val, p_val = pearsonr(df[xc].dropna(), df[yc].dropna())
            ax.text(0.05, 0.92, f"r = {r_val:.3f}", transform=ax.transAxes, fontsize=9)
        except Exception:
            pass
        plt.colorbar(sc, ax=ax, label="CNN Score")
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.set_title(f"{xl} vs {yl}", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_03_docking_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: 07_03_docking_comparison.png")

    # Fig 4 — 5-method consensus heatmap
    boltz  = pd.read_csv(BOLTZ_CSV)[["CandID", "pIC50_boltz"]]
    mmpbsa = pd.read_csv(MMPBSA_CSV)[["CandID", "DELTA_TOTAL_pb", "DELTA_TOTAL_gb"]]
    merged = df.merge(boltz, on="CandID", how="left")
    merged = merged.merge(mmpbsa, on="CandID", how="left")
    merged["pIC50_boltz"] = merged["pIC50_boltz"].fillna(merged["pIC50_boltz"].min())

    def norm_rank(s, ascending=True):
        r = s.rank(ascending=ascending, method="min")
        return (r - 1) / max(len(s) - 1, 1)

    heat = np.column_stack([
        norm_rank(merged["DockingScore"],     ascending=True),
        norm_rank(merged["DELTA_TOTAL_pb"],   ascending=True),
        norm_rank(merged["DELTA_TOTAL_gb"],   ascending=True),
        norm_rank(merged["pIC50_boltz"],      ascending=False),
        norm_rank(merged["cnn_affinity"],     ascending=True),
        norm_rank(merged["cnn_score"],        ascending=False),
    ])

    fig, ax = plt.subplots(figsize=(9, 11))
    im = ax.imshow(heat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels(
        ["AutoDock-GPU", "MM-PBSA", "MM-GBSA", "Boltz-2\npIC50", "GNINA\nAffinity", "GNINA\nCNNscore"],
        fontsize=9, fontweight="bold"
    )
    ax.set_yticks(range(len(merged)))
    ax.set_yticklabels(merged["CandID"].tolist(), fontsize=9)
    plt.colorbar(im, ax=ax, label="Normalized rank (0=best, 1=worst)", shrink=0.6)
    ax.set_title("6-Method Consensus Ranking\nTop 20 RET Kinase Candidates",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_04_consensus_heatmap_6method.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: 07_04_consensus_heatmap_6method.png")

    # Fig 5 — Final consensus bar (top 10)
    merged["rank_dock"]   = norm_rank(merged["DockingScore"],   ascending=True)
    merged["rank_pb"]     = norm_rank(merged["DELTA_TOTAL_pb"], ascending=True)
    merged["rank_gb"]     = norm_rank(merged["DELTA_TOTAL_gb"], ascending=True)
    merged["rank_boltz"]  = norm_rank(merged["pIC50_boltz"],    ascending=False)
    merged["rank_gnina_a"]= norm_rank(merged["cnn_affinity"],   ascending=True)
    merged["rank_gnina_s"]= norm_rank(merged["cnn_score"],      ascending=False)
    merged["consensus_6"] = (merged[["rank_dock","rank_pb","rank_gb",
                                      "rank_boltz","rank_gnina_a","rank_gnina_s"]].mean(axis=1))
    merged.sort_values("consensus_6", inplace=True, ignore_index=True)

    top10 = merged.head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_c = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top10)))[::-1]
    bars = ax.barh(top10["CandID"][::-1], top10["consensus_6"][::-1],
                   color=colors_c, edgecolor="k", lw=0.5)
    ax.set_xlabel("6-Method Consensus Score (lower = better)", fontsize=11)
    ax.set_title("Top 10 RET Lead Candidates — 6-Method Consensus",
                 fontsize=12, fontweight="bold")
    ax.invert_xaxis()
    for bar, val in zip(bars, top10["consensus_6"][::-1]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_05_final_consensus_top10.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: 07_05_final_consensus_top10.png")

    return merged


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Step 07: GNINA CNN Docking — Top 20 RET Kinase Candidates")
    print("=" * 65)

    dock_df = pd.read_csv(DOCK_CSV)
    top20   = dock_df[dock_df["DockRank"] <= TOP_N].copy().reset_index(drop=True)

    print(f"\nTop {TOP_N} candidates to dock with GNINA:")
    print(top20[["DockRank","CandID","DockingScore"]].to_string(index=False))

    print(f"\n[1/2] GNINA docking ({TOP_N} candidates, exhaustiveness=16, cnn_scoring=all)...")
    results = []
    failed  = []

    for _, row in top20.iterrows():
        try:
            res = dock_one(row["CandID"], row["DLGFile"])
            res.update({
                "DockRank":     int(row["DockRank"]),
                "DockingScore": float(row["DockingScore"]),
                "QED":          float(row["QED"]),
                "MW":           float(row["MW"]),
                "SMILES":       row["SMILES"],
            })
            results.append(res)
        except Exception as exc:
            print(f"  ✗ {row['CandID']}: {exc}")
            failed.append({"CandID": row["CandID"], "error": str(exc)[:300]})

    if not results:
        print("ERROR: all GNINA runs failed.")
        return

    res_df = pd.DataFrame(results).sort_values("cnn_affinity", ignore_index=True)
    res_df.to_csv(DATA_DIR / "RET_gnina_results.csv", index=False)
    res_df.to_csv(RESULTS_DIR / "07_gnina_results.csv", index=False)
    print(f"\nSaved → data/processed/RET_gnina_results.csv ({len(res_df)} rows)")

    if failed:
        pd.DataFrame(failed).to_csv(RESULTS_DIR / "07_gnina_failed.csv", index=False)

    # Summary table
    print("\n" + "─" * 70)
    print(f"{'#':<4} {'CandID':<10} {'Vina':>8} {'CNNscore':>10} {'CNNaff':>10} {'AutoDock':>10}")
    print("─" * 70)
    for i, r in res_df.iterrows():
        print(f"{i+1:<4} {r['CandID']:<10} {r['vina_affinity']:>8.2f} "
              f"{r['cnn_score']:>10.3f} {r['cnn_affinity']:>10.2f} "
              f"{r['DockingScore']:>10.2f}")

    print("\n[2/2] Generating figures...")
    merged = generate_figures(res_df.copy())

    top5 = merged.head(5)["CandID"].tolist()
    best_cnn_s = res_df.sort_values("cnn_score", ascending=False).iloc[0]
    best_cnn_a = res_df.sort_values("cnn_affinity").iloc[0]

    stats = {
        "n_docked": len(results), "n_failed": len(failed),
        "best_cnn_score_cand": best_cnn_s["CandID"],
        "best_cnn_score_val":  float(best_cnn_s["cnn_score"]),
        "best_cnn_affinity_cand": best_cnn_a["CandID"],
        "best_cnn_affinity_val":  float(best_cnn_a["cnn_affinity"]),
        "mean_cnn_score":    float(res_df["cnn_score"].mean()),
        "mean_cnn_affinity": float(res_df["cnn_affinity"].mean()),
        "top5_consensus_6method": top5,
    }
    (RESULTS_DIR / "07_gnina_stats.json").write_text(
        __import__("json").dumps(stats, indent=2)
    )

    print("\n" + "=" * 65)
    print("Step 07 complete.")
    print(f"  Best CNNscore    : {stats['best_cnn_score_cand']} ({stats['best_cnn_score_val']:.3f})")
    print(f"  Best CNNaffinity : {stats['best_cnn_affinity_cand']} ({stats['best_cnn_affinity_val']:.2f} kcal/mol)")
    print(f"  Top-5 (6-method consensus): {top5}")
    print("  Next → Step 08: Per-residue MM-GBSA decomposition (top 5)")
    print("=" * 65)


if __name__ == "__main__":
    main()
