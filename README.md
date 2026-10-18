# Generative AI Pipeline for RET Kinase Type I Inhibitor Candidate Identification

**Target:** RET kinase (UniProt P07949) — Medullary Thyroid Carcinoma / RET-rearranged NSCLC  
**Method:** Target-conditioned BiLSTM generative model with orthogonal consensus validation  
**Status:** Manuscript under revision at *Scientific Reports*

---

## Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 01 | `01_data_curation.py` | BindingDB data retrieval, IC50 filtering, SMILES standardization, PAINS/Type-I pharmacophore filtering |
| 02 | `02_protein_features.py` | PDB 6NEC binding site analysis, 20×13 feature matrix generation for target conditioning |
| 03 | `03_bilstm_training.py` | Target-conditioned BiLSTM with 8-head attention training (546 curated RET inhibitors) |
| 04 | `04_molecule_generation.py` | 10,000 SMILES generation + 9-stage filtering funnel → 100 candidates |
| 05 | `05_docking_analysis.py` | AutoDock-GPU v1.6 (OpenCL) docking of 100 candidates against PDB 6NEC |
| 05b | `05b_boltz2_affinity.py` | Boltz-2 binding affinity prediction for top 20 candidates |
| 06 | `06_mmpbsa_mmgbsa.py` | AmberTools22 MM-PBSA/MM-GBSA free energy calculations (top 20) |
| 06b | `06b_reparse_mmpbsa.py` | MM-PBSA output re-parsing utility |
| 07 | `07_gnina_docking.py` | GNINA v1.3.2 CNN rescoring + 6-method consensus scoring heatmap |
| 08 | `08_residue_decomp.py` | Per-residue MM-GBSA energy decomposition (top 5 consensus leads) |
| 09 | `09_md_simulation.py` | MD simulation orchestrator (cpptraj analysis wrapper) |
| 09b | `09b_openmm_run.py` | OpenMM 8.0 production NPT MD (ff14SB + GAFF2 + TIP3P, CUDA mixed precision) |
| 09c | `09c_md_figures.py` | MD trajectory figure generation |
| 09d | `09d_rerun_cpptraj.py` | cpptraj RMSD/RMSF/Rg/H-bond re-analysis |
| 09e | `09e_md_stability_v2.py` | Final 4-panel MD stability figure (backbone RMSD, ligand RMSD, Rg, RMSF) |
| 10a | `10a_pymol_styles_preview.py` | PyMOL 3D pose visualization |
| 10b | `10b_2d_interaction_diagrams.py` | 2D protein-ligand interaction diagrams (RDKit + matplotlib) |

### Revision Analysis Scripts (`revisions/`)

| Script | Purpose |
|--------|---------|
| `R1_novelty_check.py` | Exact SMILES + Tanimoto novelty verification of all 100 candidates vs training set |
| `R2_sa_scores.py` | Synthetic accessibility (SA) scores for all candidates and top 5 leads |
| `R3_reference_docking.py` | AutoDock-GPU docking of selpercatinib, pralsetinib, nintedanib as benchmarks |
| `R3b_reference_docking_fixed.py` | Fixed version with correct SMILES from PubChem CID lookup |
| `R4_correlation_matrix.py` | Spearman rank correlation between 6 scoring methods |
| `R5_multi_ic50_analysis.py` | Analysis of multiple IC50 measurements per compound in BindingDB |

---

## Key Results

| Metric | Value |
|--------|-------|
| Training compounds | 546 (pIC50 ≥ 7.0) |
| Generated candidates | 10,000 → 100 after 9-stage filtering |
| Best docking (generated) | −9.65 kcal/mol (RET_016) |
| Best docking (clinical ref.) | Selpercatinib −9.01, Pralsetinib −8.71, Nintedanib −10.19 kcal/mol |
| Best Boltz-2 pIC50 | 9.641 (RET_091) |
| Best MM-GBSA | −56.89 kcal/mol (all 20); −48.56 (top 5) |
| MD stability (RET_074) | Backbone RMSD 1.74 ± 0.29 Å, Ligand RMSD 0.644 ± 0.156 Å (100 ns) |
| MD stability (RET_098) | Backbone RMSD ≤ 2.0 Å, Ligand RMSD 0.717 ± 0.151 Å (100 ns) |
| All candidates novel | Max Tanimoto = 0.734 vs full curated set; zero exact SMILES matches |
| SA scores (top 5) | 2.66–3.26 (all < 4.0, moderate–easy synthesizability) |

---

## Requirements

```
Python 3.10+
RDKit >= 2023.09
PyTorch >= 2.0
OpenMM 8.0
AmberTools 22 (MMPBSA.py, cpptraj, antechamber, parmchk2, tleap)
meeko 0.7.1
AutoDock-GPU v1.6 (OpenCL)
GNINA v1.3.2
Boltz-2 v2.2.1
```

## Target Structure

- **PDB:** 6NEC (RET kinase domain, Chain A, 1.87 Å, co-crystal with nintedanib analog XIN)
- **Docking box:** 22.5 × 22.5 × 22.5 Å³ centered at (5.72, 4.62, −3.27 Å)
- **Binding site residues (key):** GLU805, ALA807, LYS758, VAL804 (gatekeeper), ASP892

## Citation

If you use this pipeline, please cite the associated *Scientific Reports* manuscript (DOI to be added upon acceptance).
