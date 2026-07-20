"""Structure input validation and ASE/Phonopy conversion helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from phonopy.structure.atoms import PhonopyAtoms

from .exceptions import StructureValidationError


def load_structure(
    structure: Atoms | str | Path,
    *,
    format: str | None = None,
    index: int | str = -1,
) -> Atoms:
    """Return a defensive, validated copy of a 3D periodic crystal."""

    if isinstance(structure, Atoms):
        atoms = structure.copy()
    else:
        try:
            loaded = read(str(structure), format=format, index=index)
        except Exception as exc:
            raise StructureValidationError(
                f"failed to read structure {structure!s}: {exc}"
            ) from exc
        if not isinstance(loaded, Atoms):
            raise StructureValidationError(
                "structure input resolved to multiple frames; select one integer index"
            )
        atoms = loaded.copy()
    validate_structure(atoms)
    return atoms


def validate_structure(atoms: Atoms) -> None:
    """Validate the physical and numeric invariants needed by Phonopy."""

    if len(atoms) == 0:
        raise StructureValidationError("structure contains no atoms")
    if not np.all(atoms.pbc):
        raise StructureValidationError(
            "v1 requires periodic boundary conditions along all three axes"
        )
    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise StructureValidationError("cell must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(cell))) <= 1.0e-12:
        raise StructureValidationError("cell is singular or has zero volume")
    if not np.all(np.isfinite(positions)):
        raise StructureValidationError("atomic positions contain non-finite values")
    numbers = np.asarray(atoms.numbers)
    if numbers.shape != (len(atoms),) or np.any(numbers <= 0):
        raise StructureValidationError("all atoms must have valid chemical species")
    masses = np.asarray(atoms.get_masses(), dtype=float)
    if masses.shape != (len(atoms),) or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise StructureValidationError("all atomic masses must be finite and positive")


def ase_to_phonopy(atoms: Atoms) -> PhonopyAtoms:
    """Convert ASE Atoms without carrying ASE constraints or calculator state."""

    kwargs: dict[str, object] = {
        "symbols": atoms.get_chemical_symbols(),
        "cell": np.asarray(atoms.cell.array, dtype=float),
        "scaled_positions": np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float),
        "masses": np.asarray(atoms.get_masses(), dtype=float),
    }
    if atoms.has("initial_magmoms"):
        kwargs["magnetic_moments"] = np.asarray(
            atoms.get_initial_magnetic_moments(), dtype=float
        )
    return PhonopyAtoms(**kwargs)


def phonopy_to_ase(atoms: PhonopyAtoms) -> Atoms:
    """Convert a Phonopy structure to unconstrained 3D-periodic ASE Atoms."""

    converted = Atoms(
        symbols=list(atoms.symbols),
        cell=np.asarray(atoms.cell, dtype=float),
        scaled_positions=np.asarray(atoms.scaled_positions, dtype=float),
        pbc=True,
    )
    if atoms.masses is not None:
        converted.set_masses(np.asarray(atoms.masses, dtype=float))
    magnetic_moments = getattr(atoms, "magnetic_moments", None)
    if magnetic_moments is not None:
        converted.set_initial_magnetic_moments(np.asarray(magnetic_moments, dtype=float))
    return converted
