"""Public and internal data types for the unified workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, TYPE_CHECKING, runtime_checkable

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

if TYPE_CHECKING:
    from phonopy import Phonopy


RunStatus = Literal["stable", "cycle_detected", "max_evaluations"]
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class CalculationContext:
    stage: str
    workdir: Path
    iteration_index: int = 0
    displacement_index: int | None = None
    candidate_index: int | None = None
    candidate_id: str | None = None


@runtime_checkable
class CalculatorFactory(Protocol):
    def __call__(
        self, *, context: CalculationContext, **kwargs: Any
    ) -> Calculator:
        """Return a new ASE calculator for *context*."""


@dataclass(frozen=True, slots=True)
class MeshData:
    qpoints: np.ndarray
    weights: np.ndarray
    frequencies: np.ndarray
    eigenvectors: np.ndarray
    mesh_numbers: np.ndarray
    mesh_length: float


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    output_dir: Path
    manifest: Path
    resolved_config: Path
    fingerprint: Path
    input_structure: Path
    iterations_dir: Path
    history: Path
    summary: Path
    final_dir: Path
    final_structure: Path
    final_phonopy_parameters: Path
    final_force_constants: Path
    final_mesh_yaml: Path
    final_mesh_arrays: Path


@dataclass(frozen=True, slots=True)
class IterationSummary:
    index: int
    status: Literal["stable", "selected", "cycle_detected", "max_evaluations"]
    evaluated_structure: Path
    minimum_frequency_thz: float
    number_of_soft_mode_groups: int
    supercell_matrix: tuple[tuple[int, int, int], ...]
    mesh_numbers: tuple[int, int, int]
    selected_candidate_id: str | None = None
    selected_structure: Path | None = None
    selected_energy_per_atom_eV: float | None = None
    energy_change_per_atom_eV: float | None = None
    candidate_failures: int = 0


@dataclass(slots=True)
class RunResult:
    status: RunStatus
    relaxed_atoms: Atoms
    phonon: Phonopy
    mesh: MeshData
    iterations: tuple[IterationSummary, ...]
    artifacts: ArtifactPaths


@dataclass(frozen=True, slots=True)
class SoftModeGroup:
    rank: int
    qpoint_index: int
    qpoint: tuple[float, float, float]
    weight: int
    band_indices: tuple[int, ...]
    frequencies_thz: tuple[float, ...]
    minimum_frequency_thz: float

    @property
    def degeneracy(self) -> int:
        return len(self.band_indices)


@dataclass(frozen=True, slots=True)
class CommensurateSupercell:
    group_rank: int
    qpoint_fractions: tuple[str, str, str]
    matrix: tuple[tuple[int, int, int], ...]
    determinant: int
    atom_count: int
    reference_structure: Path


@dataclass(frozen=True, slots=True)
class DisplacementStatistics:
    mean_angstrom: float
    rms_angstrom: float
    maximum_angstrom: float


@dataclass(frozen=True, slots=True)
class DistortionCandidate:
    candidate_id: str
    group_rank: int
    band_indices: tuple[int, ...]
    coefficients: tuple[int, ...]
    frequencies_thz: tuple[float, ...]
    sign: int
    phase_degrees: float
    target_mean_displacement_angstrom: float
    displacement_statistics: DisplacementStatistics
    structure_path: Path


@dataclass(frozen=True, slots=True)
class SoftModeResult:
    soft_mode_groups: tuple[SoftModeGroup, ...]
    selected_mode_groups: tuple[SoftModeGroup, ...]
    supercells: tuple[CommensurateSupercell, ...]
    candidates: tuple[DistortionCandidate, ...]
    output_dir: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class CandidateReductionArtifactPaths:
    output_dir: Path
    manifest: Path
    resolved_config: Path
    fingerprint: Path
    candidates_dir: Path
    unique_dir: Path
    deduplication: Path
    summary: Path


@dataclass(slots=True)
class CandidateResult:
    index: int
    candidate_id: str
    source: str
    status: Literal["success", "failed"]
    relaxed_atoms: Atoms | None = None
    primitive_atoms: Atoms | None = None
    energy_eV: float | None = None
    energy_per_atom_eV: float | None = None
    max_force_eV_per_A: float | None = None
    spacegroup_number: int | None = None
    spacegroup_symbol: str | None = None
    duplicate_group: int | None = None
    is_representative: bool = False
    error: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    index: int
    representative_index: int
    member_indices: tuple[int, ...]
    structure_path: Path


@dataclass(slots=True)
class CandidateReductionResult:
    status: Literal["complete", "partial"]
    candidates: list[CandidateResult]
    duplicate_groups: list[DuplicateGroup]
    unique_structures: list[Atoms]
    artifacts: CandidateReductionArtifactPaths
