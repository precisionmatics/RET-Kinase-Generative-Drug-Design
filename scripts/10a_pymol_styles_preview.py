"""
PyMOL style preview — 5 styles for RET_091 (top lead)
Run: conda run -n pymol pymol -cq scripts/10a_pymol_styles_preview.py
"""

import os
from pymol import cmd

RECEPTOR = "docking/receptor/6NEC_chainA.pdb"
LIGAND   = "docking/results/RET_cand_91-best.pdbqt"
OUTDIR   = "figures/10_leads/style_preview"
os.makedirs(OUTDIR, exist_ok=True)

W, H = 1600, 1200
KEY_RES = "resi 758+804+805+806+807+892"

def base_load():
    cmd.reinitialize()
    cmd.load(RECEPTOR, "receptor")
    cmd.load(LIGAND,   "ligand")
    cmd.remove("solvent")
    cmd.select("bsite", f"receptor and ({KEY_RES})")
    cmd.select("lig",   "ligand")
    cmd.zoom("lig", buffer=12)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)

# ── Style 1: Classic cartoon + sticks ─────────────────────────────────────────
def style1():
    base_load()
    cmd.bg_color("white")
    cmd.show("cartoon", "receptor")
    cmd.hide("everything", "ligand")
    cmd.show("sticks", "ligand")
    cmd.color("marine", "receptor")
    cmd.color("atomic", "ligand and not elem C")
    cmd.color("yellow", "ligand and elem C")
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("stick_radius", 0.15)
    cmd.show("sticks", "bsite")
    cmd.color("cyan", "bsite and elem C")
    cmd.util.cnc("bsite")
    cmd.ray(W, H)
    cmd.png(f"{OUTDIR}/style1_classic_cartoon.png", dpi=300)
    print("Style 1 done")

# ── Style 2: Transparent surface + cartoon + ligand sticks ────────────────────
def style2():
    base_load()
    cmd.bg_color("white")
    cmd.show("cartoon", "receptor")
    cmd.show("surface", "receptor")
    cmd.color("lightblue", "receptor")
    cmd.set("transparency", 0.65, "receptor")
    cmd.hide("everything", "ligand")
    cmd.show("sticks", "ligand")
    cmd.color("atomic", "ligand and not elem C")
    cmd.color("orange", "ligand and elem C")
    cmd.set("stick_radius", 0.18)
    cmd.set("two_sided_lighting", 1)
    cmd.ray(W, H)
    cmd.png(f"{OUTDIR}/style2_transparent_surface.png", dpi=300)
    print("Style 2 done")

# ── Style 3: White surface + ball-and-stick ligand ────────────────────────────
def style3():
    base_load()
    cmd.bg_color("white")
    cmd.show("surface", "receptor")
    cmd.color("wheat", "receptor")
    cmd.hide("everything", "ligand")
    cmd.show("sticks",  "ligand")
    cmd.show("spheres", "ligand")
    cmd.set("sphere_scale", 0.28)
    cmd.color("atomic", "ligand and not elem C")
    cmd.color("tv_red", "ligand and elem C")
    cmd.set("stick_radius", 0.13)
    cmd.ray(W, H)
    cmd.png(f"{OUTDIR}/style3_surface_ballstick.png", dpi=300)
    print("Style 3 done")

# ── Style 4: Dark background — neon ligand ────────────────────────────────────
def style4():
    base_load()
    cmd.bg_color("black")
    cmd.show("cartoon", "receptor")
    cmd.color("gray60", "receptor")
    cmd.set("cartoon_transparency", 0.3)
    cmd.hide("everything", "ligand")
    cmd.show("sticks", "ligand")
    cmd.color("atomic", "ligand and not elem C")
    cmd.color("chartreuse", "ligand and elem C")
    cmd.set("stick_radius", 0.20)
    cmd.show("sticks", "bsite")
    cmd.color("magenta", "bsite and elem C")
    cmd.set("depth_cue", 0)
    cmd.ray(W, H)
    cmd.png(f"{OUTDIR}/style4_dark_neon.png", dpi=300)
    print("Style 4 done")

# ── Style 5: B-factor spectrum surface + ligand + H-bonds ─────────────────────
def style5():
    base_load()
    cmd.bg_color("white")
    cmd.show("surface", "receptor")
    cmd.set("transparency", 0.25, "receptor")
    cmd.spectrum("b", "blue_white_red", "receptor")
    cmd.hide("everything", "ligand")
    cmd.show("sticks", "ligand")
    cmd.color("atomic", "ligand and not elem C")
    cmd.color("lime", "ligand and elem C")
    cmd.set("stick_radius", 0.18)
    # H-bonds
    cmd.select("hbond_don", "(receptor) within 3.5 of (ligand)")
    cmd.show("sticks", "hbond_don")
    cmd.color("salmon", "hbond_don and elem C")
    cmd.distance("hbonds", "(receptor) within 3.5 of (ligand)", "ligand", 3.5, mode=2)
    cmd.set("dash_color", "yellow")
    cmd.set("dash_width", 3.0)
    cmd.set("dash_gap", 0.3)
    cmd.hide("labels", "hbonds")
    cmd.ray(W, H)
    cmd.png(f"{OUTDIR}/style5_spectrum_hbonds.png", dpi=300)
    print("Style 5 done")


style1()
style2()
style3()
style4()
style5()

print(f"\nAll 5 previews saved to {OUTDIR}/")
