"""Shared ASE relaxation support for PhonoKiller workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE, LBFGS
import numpy as np

from .config import OptimizerName, RelaxationConfig, RelaxationMode
from .exceptions import CalculatorValidationError, RelaxationError
from .models import CalculationContext, ProgressCallback


_OPTIMIZERS: dict[OptimizerName, type] = {
    OptimizerName.BFGS: BFGS,
    OptimizerName.LBFGS: LBFGS,
    OptimizerName.FIRE: FIRE,
}


@dataclass(slots=True)
class RelaxationOutcome:
    """A detached relaxed structure and its final calculator metrics."""

    atoms: Atoms
    metrics: dict[str, Any]


def calculator_for(
    provider: Calculator | Callable[..., Calculator], context: CalculationContext
) -> Calculator:
    """Resolve and validate a calculator instance for one calculation context."""

    if isinstance(provider, Calculator):
        calculator = provider
    elif callable(provider):
        try:
            calculator = provider(context=context)
        except Exception as exc:
            raise CalculatorValidationError(
                f"calculator factory failed for {context.stage}: {exc}"
            ) from exc
    else:
        raise CalculatorValidationError(
            "calculator must be an ASE Calculator or a callable factory"
        )
    if not isinstance(calculator, Calculator):
        raise CalculatorValidationError(
            f"calculator provider returned {type(calculator).__name__}, "
            "not an ASE Calculator"
        )
    return calculator


def relax_atoms(
    input_atoms: Atoms,
    provider: Calculator | Callable[..., Calculator],
    config: RelaxationConfig,
    *,
    context: CalculationContext,
    relaxed_structure: Path,
    trajectory_path: Path,
    progress: ProgressCallback | None = None,
) -> RelaxationOutcome:
    """Relax one structure and write the standard relaxation artifacts."""

    relaxation_dir = context.workdir
    relaxation_dir.mkdir(parents=True, exist_ok=True)
    _archive_failed_relaxation(relaxation_dir)
    atoms = input_atoms.copy()
    atoms.calc = calculator_for(provider, context)
    _validate_initial_calculator(atoms, config.mode)

    target: Any = atoms
    if config.mode is RelaxationMode.FULL_CELL:
        target = FrechetCellFilter(atoms)
    elif config.mode is RelaxationMode.FIXED_SHAPE:
        target = FrechetCellFilter(atoms, hydrostatic_strain=True)

    logfile = relaxation_dir / "optimizer.log"
    restart = relaxation_dir / "optimizer.restart.json"
    trajectory = Trajectory(str(trajectory_path), "w", atoms)
    optimizer = _OPTIMIZERS[config.optimizer](
        target,
        logfile=str(logfile),
        restart=str(restart),
    )
    optimizer.attach(trajectory.write, interval=1)
    if progress is not None:
        optimizer.attach(
            lambda: _report_progress(atoms, optimizer, config, context, progress),
            interval=1,
        )
    try:
        converged = bool(
            optimizer.run(fmax=config.force_tolerance, steps=config.max_steps)
        )
    except BaseException:
        failed = atoms.copy()
        failed.calc = None
        write(relaxation_dir / "failed.extxyz", failed, format="extxyz")
        raise
    finally:
        trajectory.close()

    metrics = _relaxation_metrics(atoms, optimizer.get_number_of_steps())
    metrics["converged"] = converged
    _atomic_json(relaxation_dir / "metrics.json", metrics)
    relaxed = atoms.copy()
    relaxed.calc = None
    write(relaxed_structure, relaxed, format="extxyz")
    if not converged:
        raise RelaxationError(
            f"{config.optimizer.value} did not reach {config.force_tolerance:g} "
            f"eV/Angstrom within {config.max_steps} steps"
        )
    return RelaxationOutcome(atoms=relaxed, metrics=metrics)


def _report_progress(
    atoms: Atoms,
    optimizer: Any,
    config: RelaxationConfig,
    context: CalculationContext,
    progress: ProgressCallback,
) -> None:
    """Report one cached optimizer state without changing optimizer behavior."""

    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    max_force = float(np.linalg.norm(forces, axis=1).max())
    if context.candidate_index is None:
        label = "Initial relaxation"
    else:
        candidate = context.candidate_id or str(context.candidate_index)
        label = f"Candidate {context.candidate_index + 1} ({candidate})"
    progress(
        f"{label}: step {optimizer.get_number_of_steps()}/{config.max_steps}; "
        f"energy {energy:.8f} eV; max force {max_force:.6f} eV/Angstrom."
    )


def _archive_failed_relaxation(relaxation_dir: Path) -> None:
    candidates = [
        relaxation_dir / "trajectory.traj",
        relaxation_dir / "optimizer.log",
        relaxation_dir / "optimizer.restart.json",
        relaxation_dir / "failed.extxyz",
        relaxation_dir / "metrics.json",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return
    attempt = 1
    while (relaxation_dir / f"failed-attempt-{attempt:03d}").exists():
        attempt += 1
    archive = relaxation_dir / f"failed-attempt-{attempt:03d}"
    archive.mkdir()
    for path in existing:
        os.replace(path, archive / path.name)


def _validate_initial_calculator(atoms: Atoms, mode: RelaxationMode) -> None:
    try:
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
    except Exception as exc:
        raise CalculatorValidationError(
            f"calculator must provide energy and forces for relaxation: {exc}"
        ) from exc
    _validate_scalar_and_forces(energy, forces, len(atoms), "initial structure")
    if mode is not RelaxationMode.POSITIONS:
        try:
            stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
        except Exception as exc:
            raise CalculatorValidationError(
                f"{mode.value} relaxation requires calculator stress: {exc}"
            ) from exc
        if stress.shape != (3, 3) or not np.all(np.isfinite(stress)):
            raise CalculatorValidationError("calculator returned invalid 3x3 stress")


def _validate_scalar_and_forces(
    energy: float, forces: np.ndarray, atom_count: int, label: str
) -> None:
    if not np.isfinite(energy):
        raise CalculatorValidationError(
            f"calculator returned non-finite energy for {label}"
        )
    if forces.shape != (atom_count, 3):
        raise CalculatorValidationError(
            f"calculator returned force shape {forces.shape} for {label}; "
            f"expected {(atom_count, 3)}"
        )
    if not np.all(np.isfinite(forces)):
        raise CalculatorValidationError(
            f"calculator returned non-finite forces for {label}"
        )


def _relaxation_metrics(atoms: Atoms, steps: int) -> dict[str, Any]:
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    metrics: dict[str, Any] = {
        "steps": int(steps),
        "energy_eV": energy,
        "max_force_eV_per_A": float(np.linalg.norm(forces, axis=1).max()),
        "forces_eV_per_A": forces.tolist(),
    }
    try:
        stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
    except Exception:
        stress = None
    if stress is not None and stress.shape == (3, 3) and np.all(np.isfinite(stress)):
        metrics["stress_eV_per_A3"] = stress.tolist()
    return metrics


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
