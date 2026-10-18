#!/usr/bin/env python3
"""
10b_2d_interaction_diagrams.py
2D protein-ligand interaction diagrams for RET top 5 candidates.
Residues coloured by interaction type; H-bond distances labelled.
"""

import os
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = "/home/stalin/Desktop/RET_Generative"
OUTDIR = f"{BASE}/figures/10_leads/2d_interactions"
os.makedirs(OUTDIR, exist_ok=True)

# ── Compounds ──────────────────────────────────────────────────────────────────
COMPOUNDS = {
    "RET_074": ("Cc1cc(CC(=O)c2cn(C3(C)CC3)c3ncnc(N)c23)n[nH]1",          -9.04),
    "RET_091": ("Cc1cc(NC(=O)c2cn(C3(C)CC3)c3c(C)cnc(N)c23)n[nH]1",       -9.40),
    "RET_092": ("Cc1cc(NC(=O)c2cn(C3CCCC3)c3ccnc(N)c23)n[nH]1",           -9.29),
    "RET_098": ("Cc1cc(NC(=O)c2cn(C3CCCC3)c3ncnc(N)c23)[nH]n1",           -9.05),
    "RET_099": ("Cc1cc(NC(=O)c2c(N)n(C3(C)CC3)c3ccc(C#N)cc23)n[nH]1",    -9.32),
}

# ── Interaction visual styles ──────────────────────────────────────────────────
ITYPE_STYLE = {
    'hbond':       {'color': '#1D6FA4', 'lw': 2.2, 'ls': (0,(6,2.5)),      'label': 'H-bond'},
    'saltbridge':  {'color': '#1A7A3C', 'lw': 2.2, 'ls': (0,(4,2,1,2)),   'label': 'Salt bridge'},
    'pistacking':  {'color': '#7B2D8B', 'lw': 2.0, 'ls': (0,(3,1.5,1,1.5)),'label': 'π–π stacking'},
    'hydrophobic': {'color': '#C75B00', 'lw': 1.8, 'ls': (0,(1,2)),        'label': 'Hydrophobic'},
}
ITYPE_ORDER = ['hbond', 'saltbridge', 'pistacking', 'hydrophobic']

# ── Residue chemistry sets ─────────────────────────────────────────────────────
HYDROPHOBIC_RES = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO', 'TYR'}
AROMATIC_RES    = {'PHE', 'TYR', 'TRP', 'HIS'}
POS_CHARGED     = {'LYS', 'ARG'}
NEG_CHARGED     = {'ASP', 'GLU'}

HBOND_ACC_ATOMS = {
    'ASP': {'OD1','OD2'}, 'GLU': {'OE1','OE2'},
    'SER': {'OG'},        'THR': {'OG1'},  'TYR': {'OH'},
    'HIS': {'ND1','NE2'}, 'ASN': {'OD1'}, 'GLN': {'OE1'}, 'MET': {'SD'},
}
HBOND_DON_ATOMS = {
    'SER': {'OG'}, 'THR': {'OG1'}, 'TYR': {'OH'},
    'LYS': {'NZ'}, 'ARG': {'NH1','NH2','NE'},
    'HIS': {'ND1','NE2'}, 'ASN': {'ND2'}, 'GLN': {'NE2'}, 'TRP': {'NE1'},
}

# ── PDB / topology helpers ─────────────────────────────────────────────────────

def build_resnum_map(pdb_path):
    """Return ordered list of original PDB residue numbers from chain A PDB.
    AMBER 1-based residue index i  →  returned_list[i-1].
    """
    seen, prev = [], None
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith('ATOM'):
                rn = int(line[22:26])
                if rn != prev:
                    seen.append(rn)
                    prev = rn
    return seen


def parse_complex(pdb_path, a2p):
    """Parse _MMPBSA_complex.pdb.
    Returns (rec_heavy, lig_heavy) – lists of atom dicts (no Hs).
    """
    rec, lig = [], []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith('ATOM'):
                continue
            an  = line[12:16].strip()
            rn  = line[17:20].strip()
            ri  = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            pdb_ri = a2p[ri - 1] if ri <= len(a2p) else ri
            atom = {'an': an, 'rn': rn, 'ri': ri, 'x': x, 'y': y, 'z': z, 'pdb_ri': pdb_ri}
            if rn == 'LIG':
                if not an.startswith('H'):
                    lig.append(atom)
            else:
                if not an.startswith('H'):
                    rec.append(atom)
    return rec, lig


def d3(a, b):
    return np.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)

# ── Interaction detection ──────────────────────────────────────────────────────

def detect_interactions(rec, lig, mol):
    """Return list of interaction dicts for each residue (one best type per residue).

    Each dict: {type, label, pdb_ri, li (lig heavy atom idx), dist}
    """
    # Ligand atom properties from mol (heavy atoms, SMILES order)
    lig_donors, lig_acceptors = set(), set()
    for atom in mol.GetAtoms():
        s = atom.GetSymbol()
        i = atom.GetIdx()
        if s in ('N', 'O', 'S'):
            lig_acceptors.add(i)
            if atom.GetTotalNumHs() > 0:
                lig_donors.add(i)

    n_mol = mol.GetNumAtoms()

    def mi(li):
        return li if li < n_mol else 0

    # Group receptor atoms by residue
    res_groups = defaultdict(list)
    for ra in rec:
        res_groups[(ra['ri'], ra['rn'], ra['pdb_ri'])].append(ra)

    interactions = []
    seen = set()

    for (ri, rn, pdb_ri), ratoms in res_groups.items():
        label = f"{rn}{pdb_ri}"
        hbond = hydro = pi = salt = None

        for ra in ratoms:
            an = ra['an']
            for li, la in enumerate(lig):
                dist = d3(ra, la)
                mol_i = mi(li)

                # ── H-bond (< 3.5 Å) ──────────────────────────────────────────
                if dist < 3.5:
                    res_is_don = (an == 'N' or an in HBOND_DON_ATOMS.get(rn, set()))
                    res_is_acc = (an == 'O' or an in HBOND_ACC_ATOMS.get(rn, set()))
                    lig_is_don = mol_i in lig_donors
                    lig_is_acc = mol_i in lig_acceptors
                    if (res_is_don and lig_is_acc) or (res_is_acc and lig_is_don):
                        if hbond is None or dist < hbond[0]:
                            hbond = (dist, li)

                # ── Hydrophobic (< 4.0 Å, C–C) ────────────────────────────────
                if dist < 4.0 and rn in HYDROPHOBIC_RES:
                    if an.startswith('C') and la['an'].startswith('C'):
                        if hydro is None or dist < hydro[0]:
                            hydro = (dist, li)

                # ── π–π stacking (< 5.5 Å, aromatic residue) ─────────────────
                if dist < 5.5 and rn in AROMATIC_RES:
                    if pi is None or dist < pi[0]:
                        pi = (dist, li)

                # ── Salt bridge (< 5.0 Å) ──────────────────────────────────────
                if dist < 5.0:
                    if rn in POS_CHARGED and an in {'NZ','NH1','NH2','NE'}:
                        if la['an'].startswith('O') or la['an'].startswith('N'):
                            if salt is None or dist < salt[0]:
                                salt = (dist, li)
                    if rn in NEG_CHARGED and an in {'OD1','OD2','OE1','OE2'}:
                        if la['an'].startswith('N'):
                            if salt is None or dist < salt[0]:
                                salt = (dist, li)

        for itype, best in [('hbond', hbond), ('saltbridge', salt),
                             ('pistacking', pi), ('hydrophobic', hydro)]:
            if best is not None:
                key = (label, itype)
                if key not in seen:
                    seen.add(key)
                    interactions.append({'type': itype, 'label': label,
                                         'pdb_ri': pdb_ri, 'li': best[1], 'dist': best[0]})
                break  # one type per residue (highest priority wins)

    return interactions

# ── 2-D ligand rendering ───────────────────────────────────────────────────────

def render_ligand(smiles, img_size=550, lig_pdb=None):
    """Return (PIL Image, atom_pixel_coords dict, RDKit mol).

    If lig_pdb is supplied the mol is built from the MMPBSA heavy-atom PDB so
    that atom index i in atom_px matches lig_heavy[i] from parse_complex.
    Pixel coords: y=0 at top (standard image convention).
    """
    if lig_pdb is not None:
        # Build heavy-atom-only PDB block preserving AMBER atom order
        heavy_lines = []
        with open(lig_pdb) as fh:
            for line in fh:
                if line.startswith(('ATOM', 'HETATM')):
                    an = line[12:16].strip()
                    if not an.startswith('H'):
                        heavy_lines.append(line.rstrip())
        block = '\n'.join(heavy_lines) + '\nEND\n'
        mol3d = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
        tmpl  = Chem.MolFromSmiles(smiles)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            mol = AllChem.AssignBondOrdersFromTemplate(tmpl, mol3d)
    else:
        mol = Chem.MolFromSmiles(smiles)

    AllChem.Compute2DCoords(mol)

    def _draw(drawer_cls, size):
        d = drawer_cls(size, size)
        d.drawOptions().addStereoAnnotation = False
        d.drawOptions().padding = 0.12
        d.drawOptions().bondLineWidth = 2.6
        d.DrawMolecule(mol)
        d.FinishDrawing()
        return d

    img = Image.open(io.BytesIO(_draw(rdMolDraw2D.MolDraw2DCairo, img_size).GetDrawingText()))

    svg_d = _draw(rdMolDraw2D.MolDraw2DSVG, img_size)
    atom_px = {i: np.array([svg_d.GetDrawCoords(i).x, svg_d.GetDrawCoords(i).y])
               for i in range(mol.GetNumAtoms())}

    return img, atom_px, mol

# ── Single compound diagram ───────────────────────────────────────────────────

def draw_diagram(cmpd_id, smiles, score, interactions,
                 fig_size=10, img_size=550, R=265,
                 font_label=9.5, font_dist=7.5,
                 out_path=None, ax_in=None, panel_font=None):
    """Draw one 2D interaction diagram.  If ax_in is given, draw into that axis
    (for collage); otherwise create a standalone figure.
    """
    img, atom_px, mol = render_ligand(smiles, img_size)
    all_xy = np.array(list(atom_px.values()))
    cx, cy = all_xy.mean(axis=0)

    standalone = ax_in is None
    if standalone:
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    else:
        ax = ax_in

    border = R + 60
    ax.set_xlim(-border, img_size + border)
    ax.set_ylim(img_size + border, -border)   # y-axis inverted: 0 at top
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_facecolor('white')

    # Ligand image (y=0 at top, matching inverted axis)
    ax.imshow(img, extent=[0, img_size, img_size, 0], zorder=1)

    if not interactions:
        if standalone:
            ax.set_title(f'{cmpd_id}  |  ΔG = {score:.2f} kcal/mol',
                         fontsize=12, fontweight='bold')
            plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        return

    n = len(interactions)
    # Sort: priority type first, then alphabetical label
    ints_sorted = sorted(interactions,
        key=lambda x: (ITYPE_ORDER.index(x['type'])
                       if x['type'] in ITYPE_ORDER else 9, x['label']))

    # Evenly-spaced angles starting slightly above horizontal
    angles = np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi, n, endpoint=False)

    for i, inter in enumerate(ints_sorted):
        angle = angles[i]
        lx = cx + R * np.cos(angle)
        ly = cy + R * np.sin(angle)
        style = ITYPE_STYLE[inter['type']]

        # Nearest ligand 2D atom to this label position
        best_j = min(atom_px, key=lambda j: np.linalg.norm(atom_px[j] - np.array([lx, ly])))
        ax_px, ay_px = atom_px[best_j]

        dvec = np.array([lx - ax_px, ly - ay_px])
        norm = np.linalg.norm(dvec)
        uv   = dvec / norm if norm > 1e-6 else np.array([1.0, 0.0])

        # Line endpoints: start just outside atom circle, end before label box
        sx, sy = ax_px + uv[0] * 13, ay_px + uv[1] * 13
        ex, ey = lx   - uv[0] * 40, ly   - uv[1] * 40

        ax.plot([sx, ex], [sy, ey],
                color=style['color'], lw=style['lw'], ls=style['ls'],
                zorder=2, alpha=0.88, solid_capstyle='round')

        # Distance label on H-bonds
        if inter['type'] == 'hbond':
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            ax.text(mx, my, f"{inter['dist']:.1f} Å",
                    fontsize=font_dist, color=style['color'],
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.88),
                    zorder=4)

        # Residue label box
        fs = panel_font if panel_font else font_label
        ax.text(lx, ly, inter['label'],
                fontsize=fs, ha='center', va='center',
                fontweight='bold', color='white', zorder=5,
                bbox=dict(boxstyle='round,pad=0.45',
                          facecolor=style['color'],
                          edgecolor='white', linewidth=0.8, alpha=0.93))

    # Legend (standalone only)
    if standalone:
        seen_types = {x['type'] for x in interactions}
        handles = [
            Line2D([0], [0],
                   color=ITYPE_STYLE[t]['color'],
                   lw=ITYPE_STYLE[t]['lw'] + 0.3,
                   ls=ITYPE_STYLE[t]['ls'],
                   label=ITYPE_STYLE[t]['label'])
            for t in ITYPE_ORDER if t in seen_types
        ]
        ax.legend(handles=handles, loc='lower right',
                  fontsize=10, title='Interaction type', title_fontsize=10,
                  frameon=True, framealpha=0.93, edgecolor='#888888',
                  bbox_to_anchor=(1.04, -0.02))

        ax.set_title(f'{cmpd_id}  |  ΔG = {score:.2f} kcal/mol  |  RET Kinase (PDB: 6NEC)',
                     fontsize=12, fontweight='bold', pad=14, color='#222222')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  → {out_path}")
    else:
        # Panel title for collage
        n_hb  = sum(1 for x in interactions if x['type'] == 'hbond')
        n_hy  = sum(1 for x in interactions if x['type'] == 'hydrophobic')
        n_pi  = sum(1 for x in interactions if x['type'] == 'pistacking')
        n_sb  = sum(1 for x in interactions if x['type'] == 'saltbridge')
        info  = f"HB:{n_hb}  HP:{n_hy}  π:{n_pi}  SB:{n_sb}"
        ax.set_title(f"{cmpd_id}  (ΔG = {score:.2f} kcal/mol)\n{info}",
                     fontsize=9.5, fontweight='bold', pad=5, color='#222222')

# ── Collage ───────────────────────────────────────────────────────────────────

def draw_collage(all_results):
    # Match 3D PyMOL collage order
    ORDER = ["RET_091", "RET_098", "RET_092", "RET_074", "RET_099"]

    IMG    = 520          # ligand canvas size in pixels
    R      = 225          # label ring — just outside molecule edge (~198px half-width)
    BORDER = 45           # small extra margin outside canvas so edge labels aren't clipped

    fig, axes = plt.subplots(2, 3, figsize=(32, 22))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.005,
                        hspace=0.05, wspace=0.02)

    for idx, cid in enumerate(ORDER):
        smiles, score, ints = all_results[cid]
        r, c = divmod(idx, 3)
        ax = axes[r, c]

        lig_pdb = f"{BASE}/mmpbsa/ligands/{cid}/_MMPBSA_ligand.pdb"
        img, atom_px, mol = render_ligand(smiles, IMG, lig_pdb=lig_pdb)
        all_xy = np.array(list(atom_px.values()))
        cx, cy = all_xy.mean(axis=0)

        ax.set_xlim(-BORDER, IMG + BORDER)
        ax.set_ylim(IMG + BORDER, -BORDER)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_facecolor('white')
        ax.imshow(img, extent=[0, IMG, IMG, 0], zorder=1)

        n = len(ints)
        if n == 0:
            continue

        ints_s = sorted(ints,
            key=lambda x: (ITYPE_ORDER.index(x['type'])
                           if x['type'] in ITYPE_ORDER else 9, x['label']))

        # ── Compute label angle from actual interacting atom direction ─────────
        ITYPE_PRIORITY = {'hbond': 0, 'saltbridge': 1, 'pistacking': 2, 'hydrophobic': 3}

        def atom_angle(inter):
            li = inter['li']
            ax2d = atom_px[li] if li < len(atom_px) else np.array([cx, cy])
            d = ax2d - np.array([cx, cy])
            return np.arctan2(d[1], d[0])

        def spread_and_filter(ints_list):
            """Drop lowest-priority labels until remaining ones fit without overlap.

            Minimum acceptable gap = drop_gap (~31 deg).
            Maximum labels shown  = floor(2π / drop_gap) = 11.
            """
            SPREAD = 0.60   # target gap during spreading (slightly larger than drop_gap)
            DROP   = 0.55   # minimum gap — any pair below this → drop lower priority

            keep = list(range(len(ints_list)))

            while True:
                m = len(keep)
                if m == 0:
                    return [], []

                angs = [atom_angle(ints_list[k]) for k in keep]

                # Spread to convergence
                for _ in range(500):
                    order = sorted(range(m), key=lambda k: angs[k])
                    moved = False
                    for j in range(m):
                        a, b = order[j], order[(j + 1) % m]
                        gap = (angs[b] - angs[a]) % (2 * np.pi)
                        if gap < SPREAD:
                            push = (SPREAD - gap) / 2
                            angs[a] -= push
                            angs[b] += push
                            moved = True
                    if not moved:
                        break

                # Find tightest pair
                order = sorted(range(m), key=lambda k: angs[k])
                worst = min(range(m),
                            key=lambda j: (angs[order[(j+1)%m]] - angs[order[j]]) % (2*np.pi))
                a, b   = order[worst], order[(worst + 1) % m]
                min_g  = (angs[b] - angs[a]) % (2 * np.pi)

                if min_g >= DROP:
                    break  # all labels fit cleanly

                # Drop the lower-priority label of the colliding pair
                pa = ITYPE_PRIORITY.get(ints_list[keep[a]]['type'], 9)
                pb = ITYPE_PRIORITY.get(ints_list[keep[b]]['type'], 9)
                keep.pop(a if pa >= pb else b)

            return keep, angs

        keep_idx, angles = spread_and_filter(ints_s)

        for idx2, i in enumerate(keep_idx):
            inter = ints_s[i]
            angle = angles[idx2]
            lx = cx + R * np.cos(angle)
            ly = cy + R * np.sin(angle)
            style = ITYPE_STYLE[inter['type']]

            # Line starts from the ACTUAL interacting atom
            li = inter['li']
            if li < len(atom_px):
                ax_px, ay_px = atom_px[li]
            else:
                ax_px, ay_px = cx, cy

            dvec = np.array([lx - ax_px, ly - ay_px])
            nrm  = np.linalg.norm(dvec)
            uv   = dvec / nrm if nrm > 1e-6 else np.array([1.0, 0.0])

            sx, sy = ax_px + uv[0] * 10, ay_px + uv[1] * 10
            ex, ey = lx   - uv[0] * 28, ly   - uv[1] * 28

            ax.plot([sx, ex], [sy, ey],
                    color=style['color'], lw=1.8, ls=style['ls'],
                    zorder=2, alpha=0.88, solid_capstyle='round')

            if inter['type'] == 'hbond':
                mx, my = (sx + ex) / 2, (sy + ey) / 2
                ax.text(mx, my, f"{inter['dist']:.1f} Å",
                        fontsize=9.5, color=style['color'],
                        ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85),
                        zorder=4)

            ax.text(lx, ly, inter['label'],
                    fontsize=13, ha='center', va='center',
                    fontweight='bold', color='white', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.45',
                              facecolor=style['color'],
                              edgecolor='white', linewidth=1.0, alpha=0.93))

        # Simple label below panel (matching 3D collage style)
        ax.text(0.5, 1.0, cid,
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=15, fontweight='bold', color='#222222')

    # ── Legend panel (bottom-right) ───────────────────────────────────────────
    ax_leg = axes[1, 2]
    ax_leg.set_facecolor('white')
    ax_leg.axis('off')

    handles = [
        Line2D([0], [0],
               color=ITYPE_STYLE[t]['color'],
               lw=ITYPE_STYLE[t]['lw'] + 0.8,
               ls=ITYPE_STYLE[t]['ls'],
               label=ITYPE_STYLE[t]['label'])
        for t in ITYPE_ORDER
    ]
    ax_leg.legend(handles=handles, loc='center',
                  fontsize=15, title='Interaction Type', title_fontsize=14,
                  frameon=True, framealpha=0.95, edgecolor='#aaaaaa',
                  handlelength=4.5, labelspacing=1.8,
                  bbox_to_anchor=(0.5, 0.5))

    out = f"{OUTDIR}/TOP5_2d_interaction_collage.png"
    plt.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {out}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    chainA_pdb = f"{BASE}/docking/receptor/6NEC_chainA.pdb"
    a2p = build_resnum_map(chainA_pdb)
    print(f"Residue map: {len(a2p)} entries  ({a2p[0]} → {a2p[-1]})")

    all_results = {}

    for cmpd_id, (smiles, score) in COMPOUNDS.items():
        print(f"\n{'─'*54}")
        print(f"  {cmpd_id}")
        complex_pdb = f"{BASE}/mmpbsa/ligands/{cmpd_id}/_MMPBSA_complex.pdb"
        rec, lig = parse_complex(complex_pdb, a2p)
        print(f"  Receptor heavy atoms : {len(rec)}")
        print(f"  Ligand heavy atoms   : {len(lig)}")

        mol  = Chem.MolFromSmiles(smiles)
        ints = detect_interactions(rec, lig, mol)

        print(f"  Interactions         : {len(ints)}")
        for it in sorted(ints, key=lambda x: (ITYPE_ORDER.index(x['type']), x['dist'])):
            print(f"    {it['type']:12s}  {it['label']:<10s}  {it['dist']:.2f} Å")

        out_path = f"{OUTDIR}/{cmpd_id}_2d_interaction.png"
        draw_diagram(cmpd_id, smiles, score, ints, out_path=out_path)
        all_results[cmpd_id] = (smiles, score, ints)

    print(f"\n{'─'*54}")
    print("Creating collage...")
    draw_collage(all_results)

    print("\nAll done!")
    print(f"Output : {OUTDIR}/")


if __name__ == '__main__':
    main()
