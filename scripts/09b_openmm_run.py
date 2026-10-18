#!/usr/bin/env python3
"""
OpenMM MD simulation for one RET kinase lead compound.
Called by 09_md_simulation.py via: conda run -n md-suite python 09b_openmm_run.py <cand_id>

Protocol:
  - AMBER ff14SB + GAFF2 (topology from tleap)
  - TIP3P explicit solvent, 12 Å buffer, 150 mM NaCl
  - Minimization → NVT 100 ps (T=300K, restrained) → NPT 1ns → Production 100 ns
  - Trajectory: DCD every 10 ps, restart every 1 ns
"""

import sys
import time
import json
from pathlib import Path

import numpy as np

try:
    from openmm.app import *
    from openmm import *
    from openmm import unit
except ImportError:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk import unit

# ── Args ───────────────────────────────────────────────────────────────────────
if len(sys.argv) < 3:
    print("Usage: python 09b_openmm_run.py <cand_id> <md_dir>")
    sys.exit(1)

CAND_ID = sys.argv[1]
MD_DIR  = Path(sys.argv[2])
MD_DIR.mkdir(parents=True, exist_ok=True)

PRMTOP  = MD_DIR / "solvated.prmtop"
INPCRD  = MD_DIR / "solvated.inpcrd"

# MD parameters
T           = 300  * unit.kelvin
P           = 1.0  * unit.bar
DT          = 0.002 * unit.picoseconds      # 2 fs timestep
PROD_NS     = 100                            # 100 ns production
PROD_STEPS  = int(PROD_NS * 1_000_000 / 2)  # steps = ns × 1e6 fs/ns / 2 fs/step = 50,000,000
EQUIL_STEPS_NVT = 50_000                    # 100 ps NVT
EQUIL_STEPS_NPT = 500_000                   # 1 ns NPT
MIN_STEPS   = 10_000
TRAJ_EVERY  = 50_000                        # save frame every 100 ps → 1,000 frames for 100 ns
REPORT_EVERY = 500_000                       # stdout report every 1 ns → 100 lines for 100 ns

TRAJ_FILE        = MD_DIR / f"{CAND_ID}_prod.dcd"
LOG_FILE         = MD_DIR / f"{CAND_ID}_md.log"
RST_FILE         = MD_DIR / f"{CAND_ID}_restart.xml"
STATS_FILE       = MD_DIR / f"{CAND_ID}_md_stats.json"
CHECKPOINT_FILE  = MD_DIR / f"{CAND_ID}_checkpoint.chk"
PROGRESS_FILE    = MD_DIR / f"{CAND_ID}_progress.json"
CHECKPOINT_EVERY = 500_000   # save checkpoint every 1 ns (500k steps × 2 fs)


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def build_system():
    log(f"Loading topology: {PRMTOP}")
    prmtop = AmberPrmtopFile(str(PRMTOP))
    inpcrd = AmberInpcrdFile(str(INPCRD))

    log("Building OpenMM system (PME, 10 Å cutoff, HBonds constraints)...")
    system = prmtop.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=10.0 * unit.angstroms,
        constraints=HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
    )
    return prmtop, inpcrd, system


def get_heavy_atom_restraints(system, prmtop, k=10.0):
    """Add flat-bottom position restraints on receptor + ligand heavy atoms."""
    positions_only = []
    force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", k * unit.kilocalories_per_mole / unit.angstrom**2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    return force


def run_md():
    t0 = time.time()
    log("=" * 60)
    log(f"OpenMM MD — {CAND_ID}")
    log(f"Production: {PROD_NS} ns | DT: 2 fs | T: 300 K | P: 1 bar")
    log("=" * 60)

    # Check for existing checkpoint to enable crash-restart
    steps_done = 0
    is_restart = False
    if CHECKPOINT_FILE.exists() and PROGRESS_FILE.exists() and TRAJ_FILE.exists():
        try:
            prog = json.loads(PROGRESS_FILE.read_text())
            steps_done = int(prog.get("steps_done", 0))
            if 0 < steps_done < PROD_STEPS:
                is_restart = True
                log(f"Checkpoint found: {steps_done:,}/{PROD_STEPS:,} steps done "
                    f"({steps_done * 2 / 1_000_000:.1f}/{PROD_NS} ns) — resuming.")
        except Exception:
            steps_done = 0

    prmtop, inpcrd, system = build_system()

    # Barostat for NPT
    system.addForce(MonteCarloBarostat(P, T, 25))

    # Langevin integrator
    integrator = LangevinMiddleIntegrator(T, 1.0/unit.picoseconds, DT)
    integrator.setRandomNumberSeed(42)

    # Platform — prefer CUDA, fallback to CPU
    try:
        platform = Platform.getPlatformByName("CUDA")
        prop = {"CudaPrecision": "mixed"}
        log("Platform: CUDA (mixed precision)")
    except Exception:
        platform = Platform.getPlatformByName("CPU")
        prop = {}
        log("Platform: CPU")

    simulation = Simulation(prmtop.topology, system, integrator, platform, prop)

    if not is_restart:
        simulation.context.setPositions(inpcrd.positions)
        if inpcrd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

        # ── Minimization ──────────────────────────────────────────────────────
        log(f"\n[1/4] Energy minimization ({MIN_STEPS} max steps)...")
        simulation.minimizeEnergy(maxIterations=MIN_STEPS, tolerance=10.0)
        state = simulation.context.getState(getEnergy=True)
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        log(f"  Potential energy after min: {pe:.2f} kJ/mol")

        # ── NVT equilibration ─────────────────────────────────────────────────
        log(f"\n[2/4] NVT equilibration (100 ps, T=300K, restrained)...")
        simulation.context.setVelocitiesToTemperature(T)
        simulation.reporters.append(
            StateDataReporter(str(LOG_FILE), REPORT_EVERY, step=True,
                              time=True, potentialEnergy=True, temperature=True,
                              progress=True, totalSteps=EQUIL_STEPS_NVT, speed=True, append=True)
        )
        simulation.step(EQUIL_STEPS_NVT)
        log("  NVT equilibration complete.")

        # ── NPT equilibration ─────────────────────────────────────────────────
        log(f"\n[3/4] NPT equilibration (1 ns, T=300K, P=1bar)...")
        simulation.reporters.clear()
        simulation.reporters.append(
            StateDataReporter(str(LOG_FILE), REPORT_EVERY, step=True,
                              time=True, potentialEnergy=True, temperature=True,
                              volume=True, progress=True, totalSteps=EQUIL_STEPS_NPT,
                              speed=True, append=True)
        )
        simulation.step(EQUIL_STEPS_NPT)
        log("  NPT equilibration complete.")

    else:
        log(f"\n[Restart] Skipping min/NVT/NPT — loading production checkpoint...")
        simulation.loadCheckpoint(str(CHECKPOINT_FILE))

    # ── Production MD ─────────────────────────────────────────────────────────
    steps_remaining = PROD_STEPS - steps_done
    log(f"\n[4/4] Production MD ({PROD_NS} ns = {PROD_STEPS:,} steps, "
        f"{steps_remaining:,} remaining)...")
    simulation.reporters.clear()
    simulation.reporters.append(DCDReporter(str(TRAJ_FILE), TRAJ_EVERY, append=is_restart))
    simulation.reporters.append(
        StateDataReporter(str(LOG_FILE), REPORT_EVERY, step=True,
                          time=True, potentialEnergy=True, kineticEnergy=True,
                          temperature=True, volume=True, density=True,
                          progress=True, totalSteps=PROD_STEPS, speed=True, append=True)
    )

    # Run in chunks; save checkpoint every 1 ns so a crash can be resumed
    steps_run = 0
    while steps_run < steps_remaining:
        this_chunk = min(CHECKPOINT_EVERY, steps_remaining - steps_run)
        simulation.step(this_chunk)
        steps_run += this_chunk
        simulation.saveCheckpoint(str(CHECKPOINT_FILE))
        PROGRESS_FILE.write_text(json.dumps(
            {"steps_done": steps_done + steps_run, "steps_total": PROD_STEPS}, indent=2))

    # Save final state
    simulation.saveState(str(RST_FILE))
    log(f"\nFinal state saved: {RST_FILE}")

    elapsed = (time.time() - t0) / 3600
    log(f"Total wall time: {elapsed:.2f} h")

    # Save stats
    state = simulation.context.getState(getEnergy=True)
    stats = {
        "cand_id": CAND_ID,
        "prod_ns": PROD_NS,
        "n_frames": PROD_STEPS // TRAJ_EVERY,
        "final_potential_energy_kJ": state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
        "wall_time_h": elapsed,
        "traj_file": str(TRAJ_FILE),
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2))
    log(f"Stats saved: {STATS_FILE}")
    log("MD complete.")


if __name__ == "__main__":
    run_md()
