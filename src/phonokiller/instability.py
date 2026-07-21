"""Internal soft-mode ranking and exhaustive distortion generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

from ase.geometry import find_mic
from ase.io import write
import numpy as np
from phonopy import Phonopy

from .config import (
    CandidateRelaxationOverrides,
    RelaxationConfig,
    SoftModeConfig,
)
from .exceptions import CandidateLimitError, SoftModeError
from .models import (
    CommensurateSupercell,
    DisplacementStatistics,
    DistortionCandidate,
    MeshData,
    SoftModeGroup,
    SoftModeResult,
)
from .structure import phonopy_to_ase, validate_structure
from ._resources import candidate_resource_violations, optimizer_state_estimate


_OUTPUT_SCHEMA = 3
_REPORT_NAME = "soft_modes.json"
_PREFLIGHT_NAME = "preflight.json"


@dataclass(frozen=True, slots=True)
class _GroupGenerationPlan:
    group: SoftModeGroup
    rational_qpoint: tuple[Fraction, Fraction, Fraction]
    matrix: np.ndarray
    determinant: int
    atom_count: int
    direction_count: int
    candidate_count: int


def rank_soft_modes(
    mesh: MeshData,
    config: SoftModeConfig,
) -> tuple[SoftModeGroup, ...]:
    """Return frequency-degenerate soft groups from most to least unstable."""

    qpoints, weights, frequencies, _ = _validated_mesh_arrays(mesh)
    groups: list[SoftModeGroup] = []
    for qpoint_index, (qpoint, weight, q_frequencies) in enumerate(
        zip(qpoints, weights, frequencies, strict=True)
    ):
        canonical_qpoint = tuple(
            float(value)
            for value in _canonical_qpoint(qpoint, config.qpoint_tolerance)
        )
        ordered_bands = sorted(
            range(len(q_frequencies)),
            key=lambda band_index: (float(q_frequencies[band_index]), band_index),
        )
        for cluster in _frequency_clusters(
            ordered_bands,
            q_frequencies,
            config.degeneracy_tolerance_thz,
        ):
            cluster_frequencies = tuple(
                float(q_frequencies[band_index]) for band_index in cluster
            )
            minimum = min(cluster_frequencies)
            if minimum < config.frequency_threshold_thz:
                groups.append(
                    SoftModeGroup(
                        rank=0,
                        qpoint_index=qpoint_index,
                        qpoint=canonical_qpoint,
                        weight=int(weight),
                        band_indices=tuple(sorted(cluster)),
                        frequencies_thz=tuple(
                            float(q_frequencies[band_index])
                            for band_index in sorted(cluster)
                        ),
                        minimum_frequency_thz=minimum,
                    )
                )
    groups.sort(
        key=lambda group: (
            group.minimum_frequency_thz,
            group.qpoint,
            group.band_indices[0],
        )
    )
    return tuple(replace(group, rank=rank) for rank, group in enumerate(groups, 1))


def ternary_directions(degeneracy: int) -> tuple[tuple[int, ...], ...]:
    """Return unique nonzero {-1,0,1} directions modulo overall sign."""

    if degeneracy <= 0:
        raise SoftModeError("mode degeneracy must be positive")
    directions = [
        tuple(int(value) for value in coefficients)
        for coefficients in itertools.product((-1, 0, 1), repeat=degeneracy)
        if any(coefficients)
        and next(value for value in coefficients if value != 0) > 0
    ]
    directions.sort(key=lambda value: (sum(item != 0 for item in value), value))
    return tuple(directions)


def candidate_count(groups: Iterable[SoftModeGroup]) -> int:
    """Return the exhaustive signed ternary candidate count."""

    return sum(3**group.degeneracy - 1 for group in groups)


def generate_soft_mode_candidates(
    phonon: Phonopy,
    mesh: MeshData,
    config: SoftModeConfig,
    output_dir: str | Path,
    *,
    max_candidates: int,
    candidate_relaxation: RelaxationConfig | None = None,
    max_candidate_atoms: int = 3500,
    max_dense_hessian_memory_mib: float = 256.0,
    source_fingerprint: str | None = None,
    selected_group_rank: int = 1,
    candidates_already_generated: int = 0,
) -> SoftModeResult:
    """Preflight and generate candidates for one explicitly ranked mode group."""

    destination = Path(output_dir).resolve()
    report_path = destination / _REPORT_NAME
    preflight_path = destination / _PREFLIGHT_NAME
    if report_path.exists():
        return load_soft_mode_result(destination)

    _validate_phonon(phonon, mesh)
    groups = rank_soft_modes(mesh, config)
    selected_groups = tuple(
        group for group in groups if group.rank == selected_group_rank
    )
    if not selected_groups:
        raise SoftModeError(
            f"unstable mode group rank {selected_group_rank} is not available"
        )
    effective_relaxation = candidate_relaxation or CandidateRelaxationOverrides().resolve(
        RelaxationConfig()
    )
    plans = tuple(
        _plan_group_generation(phonon, mesh, group, config)
        for group in selected_groups
    )
    preflight = _candidate_preflight_payload(
        plans,
        effective_relaxation,
        max_candidates=max_candidates,
        candidates_already_generated=candidates_already_generated,
        max_candidate_atoms=max_candidate_atoms,
        max_dense_hessian_memory_mib=max_dense_hessian_memory_mib,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_dir():
            raise SoftModeError(f"instability output is not a directory: {destination}")
        unexpected = [
            item for item in destination.iterdir() if item.name != _PREFLIGHT_NAME
        ]
        if unexpected:
            raise SoftModeError(
                f"nonempty instability output has no complete report: {destination}"
            )
    destination.mkdir(exist_ok=True)
    _write_json(preflight_path, preflight)
    violations = tuple(str(value) for value in preflight["violations"])
    if violations:
        raise CandidateLimitError(
            "candidate resource preflight refused generation: " + "; ".join(violations)
        )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-tmp-", dir=destination.parent)
    )
    supercells: list[CommensurateSupercell] = []
    candidates: list[DistortionCandidate] = []
    try:
        _write_json(temporary / _PREFLIGHT_NAME, preflight)
        for plan in plans:
            supercell, generated = _generate_group_candidates(
                phonon,
                plan,
                config,
                temporary,
                destination,
            )
            supercells.append(supercell)
            candidates.extend(generated)
        payload = {
            "schema_version": _OUTPUT_SCHEMA,
            "status": "complete",
            "source_fingerprint": source_fingerprint,
            "settings": config.model_dump(mode="json"),
            "resource_preflight": str(preflight_path),
            "counts": {
                "soft_mode_groups": len(groups),
                "selected_mode_groups": len(selected_groups),
                "generated_candidates": len(candidates),
            },
            "soft_mode_groups": [_group_payload(group) for group in groups],
            "selected_ranks": [group.rank for group in selected_groups],
            "supercells": [_jsonable(asdict(item)) for item in supercells],
            "candidates": [_jsonable(asdict(item)) for item in candidates],
        }
        _write_json(temporary / _REPORT_NAME, payload)
        if destination.exists():
            preflight_path.unlink(missing_ok=True)
            destination.rmdir()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return SoftModeResult(
        soft_mode_groups=groups,
        selected_mode_groups=selected_groups,
        supercells=tuple(supercells),
        candidates=tuple(candidates),
        output_dir=destination,
        report_path=report_path,
        preflight_path=preflight_path,
    )


def load_soft_mode_result(output_dir: str | Path) -> SoftModeResult:
    destination = Path(output_dir).resolve()
    report_path = destination / _REPORT_NAME
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != _OUTPUT_SCHEMA:
            raise ValueError("unsupported schema")
        groups = tuple(_group_from_payload(item) for item in payload["soft_mode_groups"])
        selected_ranks = set(int(value) for value in payload["selected_ranks"])
        supercells = tuple(
            CommensurateSupercell(
                group_rank=int(item["group_rank"]),
                qpoint_fractions=tuple(str(value) for value in item["qpoint_fractions"]),
                matrix=tuple(tuple(int(value) for value in row) for row in item["matrix"]),
                determinant=int(item["determinant"]),
                atom_count=int(item["atom_count"]),
                reference_structure=Path(item["reference_structure"]),
            )
            for item in payload["supercells"]
        )
        candidates = tuple(
            DistortionCandidate(
                candidate_id=str(item["candidate_id"]),
                group_rank=int(item["group_rank"]),
                band_indices=tuple(int(value) for value in item["band_indices"]),
                coefficients=tuple(int(value) for value in item["coefficients"]),
                frequencies_thz=tuple(float(value) for value in item["frequencies_thz"]),
                sign=int(item["sign"]),
                phase_degrees=float(item["phase_degrees"]),
                target_mean_displacement_angstrom=float(
                    item["target_mean_displacement_angstrom"]
                ),
                displacement_statistics=DisplacementStatistics(
                    **{
                        key: float(value)
                        for key, value in item["displacement_statistics"].items()
                    }
                ),
                structure_path=Path(item["structure_path"]),
            )
            for item in payload["candidates"]
        )
    except Exception as exc:
        raise SoftModeError(f"cannot load instability report {report_path}: {exc}") from exc
    return SoftModeResult(
        soft_mode_groups=groups,
        selected_mode_groups=tuple(group for group in groups if group.rank in selected_ranks),
        supercells=supercells,
        candidates=candidates,
        output_dir=destination,
        report_path=report_path,
        preflight_path=destination / _PREFLIGHT_NAME,
    )


def _validated_mesh_arrays(
    mesh: MeshData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qpoints = np.asarray(mesh.qpoints, dtype=float)
    weights = np.asarray(mesh.weights)
    frequencies = np.asarray(mesh.frequencies, dtype=float)
    eigenvectors = np.asarray(mesh.eigenvectors, dtype=complex)
    if qpoints.ndim != 2 or qpoints.shape[1:] != (3,):
        raise SoftModeError(f"mesh q-points have invalid shape {qpoints.shape}")
    nqpoint = len(qpoints)
    if weights.shape != (nqpoint,) or not np.all(np.asarray(weights, dtype=float) > 0):
        raise SoftModeError("mesh weights must be positive and match q-points")
    if frequencies.ndim != 2 or frequencies.shape[0] != nqpoint:
        raise SoftModeError(f"mesh frequencies have invalid shape {frequencies.shape}")
    nband = frequencies.shape[1]
    if eigenvectors.shape != (nqpoint, nband, nband):
        raise SoftModeError(f"mesh eigenvectors have invalid shape {eigenvectors.shape}")
    if np.asarray(mesh.mesh_numbers).shape != (3,):
        raise SoftModeError("mesh numbers must contain three values")
    if not (
        np.all(np.isfinite(qpoints))
        and np.all(np.isfinite(frequencies))
        and np.all(np.isfinite(eigenvectors.real))
        and np.all(np.isfinite(eigenvectors.imag))
    ):
        raise SoftModeError("mesh contains non-finite values")
    return qpoints, np.asarray(weights, dtype=int), frequencies, eigenvectors


def _frequency_clusters(
    ordered_bands: list[int], frequencies: np.ndarray, tolerance: float
) -> Iterable[tuple[int, ...]]:
    cluster: list[int] = []
    cluster_minimum = 0.0
    for band_index in ordered_bands:
        frequency = float(frequencies[band_index])
        if not cluster:
            cluster = [band_index]
            cluster_minimum = frequency
        elif frequency - cluster_minimum <= tolerance:
            cluster.append(band_index)
        else:
            yield tuple(cluster)
            cluster = [band_index]
            cluster_minimum = frequency
    if cluster:
        yield tuple(cluster)


def _canonical_qpoint(qpoint: np.ndarray, tolerance: float) -> np.ndarray:
    canonical = np.asarray(qpoint, dtype=float) - np.floor(
        np.asarray(qpoint, dtype=float) + 0.5
    )
    canonical[np.abs(canonical) <= tolerance] = 0.0
    return canonical


def _rational_qpoint(
    qpoint: tuple[float, float, float], mesh_numbers: np.ndarray, tolerance: float
) -> tuple[Fraction, Fraction, Fraction]:
    mesh = np.asarray(mesh_numbers, dtype=int)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise SoftModeError("mesh numbers must be three positive integers")
    # Phonopy GR-grid q-points are integer grid addresses transformed by an
    # integer unimodular matrix and D_diag; their component denominators still
    # divide twice the LCM of D_diag (the factor two covers half-grid shifts).
    denominator_bound = 2 * math.lcm(*(int(value) for value in mesh))
    fractions: list[Fraction] = []
    for value in qpoint:
        fraction = Fraction(float(value)).limit_denominator(denominator_bound)
        if abs(float(fraction) - float(value)) > tolerance:
            raise SoftModeError(
                f"q-point component {value:.16g} cannot be reconstructed from "
                f"returned mesh metadata {mesh.tolist()} within {tolerance:g}"
            )
        fractions.append(fraction)
    return fractions[0], fractions[1], fractions[2]


def _minimum_commensurate_supercell(
    qpoint: tuple[Fraction, Fraction, Fraction], primitive_lattice: np.ndarray
) -> np.ndarray:
    order = math.lcm(*(component.denominator for component in qpoint))
    lattice = np.asarray(primitive_lattice, dtype=float)
    kernel = _commensurate_kernel_basis(qpoint, order)
    reduced = _lll_reduce_columns(kernel, lattice)
    candidates = list(_oriented_bases(reduced))
    matrix = min(candidates, key=lambda value: _cell_shape_score(value, lattice))
    if _integer_determinant(matrix) != order or not _is_exactly_commensurate(
        matrix, qpoint
    ):
        raise SoftModeError("internal commensurate-supercell validation failed")
    return np.asarray(matrix, dtype=int)


def _commensurate_kernel_basis(
    qpoint: tuple[Fraction, Fraction, Fraction], order: int
) -> np.ndarray:
    """Return a column basis for integer vectors with integral q phase."""

    if order <= 0:
        raise SoftModeError("q-point order must be positive")
    phase = np.asarray(
        [int(component * order) for component in qpoint], dtype=object
    )
    if all(value == 0 for value in phase):
        return np.eye(3, dtype=int)

    transform = np.eye(3, dtype=object)
    reduced_phase = phase.copy()
    for column in (1, 2):
        divisor, left, right = _extended_gcd(
            int(reduced_phase[0]), int(reduced_phase[column])
        )
        if divisor == 0:
            continue
        operation = np.eye(3, dtype=object)
        operation[0, 0] = left
        operation[0, column] = -int(reduced_phase[column]) // divisor
        operation[column, 0] = right
        operation[column, column] = int(reduced_phase[0]) // divisor
        transform = transform @ operation
        reduced_phase = reduced_phase @ operation

    phase_gcd = abs(int(reduced_phase[0]))
    if math.gcd(phase_gcd, order) != 1:
        raise SoftModeError("q-point fractions do not have the expected order")
    kernel = transform @ np.diag([order, 1, 1]).astype(object)
    try:
        return np.asarray(kernel, dtype=int)
    except OverflowError as exc:
        raise SoftModeError("commensurate supercell exceeds integer range") from exc


def _extended_gcd(left: int, right: int) -> tuple[int, int, int]:
    """Return positive gcd and Bezout coefficients for two integers."""

    old_remainder, remainder = abs(left), abs(right)
    old_left, current_left = 1, 0
    old_right, current_right = 0, 1
    while remainder:
        quotient = old_remainder // remainder
        old_remainder, remainder = remainder, old_remainder - quotient * remainder
        old_left, current_left = current_left, old_left - quotient * current_left
        old_right, current_right = current_right, old_right - quotient * current_right
    if left < 0:
        old_left = -old_left
    if right < 0:
        old_right = -old_right
    return old_remainder, old_left, old_right


def _is_exactly_commensurate(
    matrix: np.ndarray, qpoint: tuple[Fraction, Fraction, Fraction]
) -> bool:
    return all(
        sum(
            (qpoint[row] * int(matrix[row, column]) for row in range(3)),
            Fraction(0, 1),
        ).denominator
        == 1
        for column in range(3)
    )


def _lll_reduce_columns(matrix: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    basis = np.asarray(matrix, dtype=int).T @ lattice
    transform = np.eye(3, dtype=int)
    k = 1
    iterations = 0
    while k < 3:
        iterations += 1
        if iterations > 1000:
            raise SoftModeError("lattice-basis reduction did not converge")
        _, coefficients, norms = _gram_schmidt_rows(basis)
        for j in range(k - 1, -1, -1):
            multiplier = int(np.rint(coefficients[k, j]))
            if multiplier:
                basis[k] -= multiplier * basis[j]
                transform[k] -= multiplier * transform[j]
                _, coefficients, norms = _gram_schmidt_rows(basis)
        if norms[k] >= (0.75 - coefficients[k, k - 1] ** 2) * norms[k - 1]:
            k += 1
        else:
            basis[[k, k - 1]] = basis[[k - 1, k]]
            transform[[k, k - 1]] = transform[[k - 1, k]]
            k = max(k - 1, 1)
    reduced = np.asarray(matrix, dtype=int) @ transform.T
    if _integer_determinant(reduced) < 0:
        reduced[:, 0] *= -1
    return reduced


def _gram_schmidt_rows(
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orthogonal = np.zeros_like(basis, dtype=float)
    coefficients = np.zeros((3, 3), dtype=float)
    norms = np.zeros(3, dtype=float)
    for i in range(3):
        orthogonal[i] = basis[i]
        for j in range(i):
            if norms[j] <= 1.0e-30:
                raise SoftModeError("commensurate supercell basis is singular")
            coefficients[i, j] = float(np.dot(basis[i], orthogonal[j]) / norms[j])
            orthogonal[i] -= coefficients[i, j] * orthogonal[j]
        norms[i] = float(np.dot(orthogonal[i], orthogonal[i]))
    return orthogonal, coefficients, norms


def _oriented_bases(matrix: np.ndarray) -> Iterable[np.ndarray]:
    for permutation in itertools.permutations(range(3)):
        permuted = np.asarray(matrix, dtype=int)[:, permutation]
        for signs in itertools.product((-1, 1), repeat=3):
            candidate = permuted @ np.diag(signs)
            if _integer_determinant(candidate) > 0:
                yield candidate


def _cell_shape_score(
    matrix: np.ndarray, primitive_lattice: np.ndarray
) -> tuple[float, float, float, tuple[int, ...]]:
    physical_vectors = np.asarray(matrix, dtype=float).T @ primitive_lattice
    lengths = np.linalg.norm(physical_vectors, axis=1)
    return (
        round(float(np.max(lengths)), 12),
        round(float(np.linalg.cond(physical_vectors)), 12),
        round(float(np.sum(lengths**2)), 12),
        tuple(int(value) for value in np.asarray(matrix).flat),
    )


def _integer_determinant(matrix: np.ndarray) -> int:
    a, b, c = (tuple(int(value) for value in row) for row in matrix)
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _plan_group_generation(
    phonon: Phonopy,
    mesh: MeshData,
    group: SoftModeGroup,
    settings: SoftModeConfig,
) -> _GroupGenerationPlan:
    rational_qpoint = _rational_qpoint(
        group.qpoint, mesh.mesh_numbers, settings.qpoint_tolerance
    )
    matrix = _minimum_commensurate_supercell(
        rational_qpoint, np.asarray(phonon.primitive.cell, dtype=float)
    )
    determinant = _integer_determinant(matrix)
    direction_count = (3**group.degeneracy - 1) // 2
    return _GroupGenerationPlan(
        group=group,
        rational_qpoint=rational_qpoint,
        matrix=matrix,
        determinant=determinant,
        atom_count=len(phonon.primitive) * determinant,
        direction_count=direction_count,
        candidate_count=2 * direction_count,
    )


def _candidate_preflight_payload(
    plans: tuple[_GroupGenerationPlan, ...],
    relaxation: RelaxationConfig,
    *,
    max_candidates: int,
    candidates_already_generated: int = 0,
    max_candidate_atoms: int,
    max_dense_hessian_memory_mib: float,
) -> dict[str, Any]:
    total_candidates = sum(plan.candidate_count for plan in plans)
    violations: list[str] = []
    cumulative_candidates = candidates_already_generated + total_candidates
    if cumulative_candidates > max_candidates:
        violations.append(
            f"exhaustive soft-mode expansion requires {total_candidates} additional "
            f"candidates after {candidates_already_generated} already generated, "
            f"exceeding search.max_candidates_per_iteration={max_candidates}"
        )

    group_payloads: list[dict[str, Any]] = []
    total_candidate_atoms = 0
    total_atom_steps = 0
    summed_optimizer_state_bytes = 0
    peak_optimizer_state_bytes = 0
    for plan in plans:
        estimate = optimizer_state_estimate(plan.atom_count, relaxation.optimizer)
        group_violations = candidate_resource_violations(
            plan.atom_count,
            relaxation.optimizer,
            max_candidate_atoms=max_candidate_atoms,
            max_dense_hessian_memory_mib=max_dense_hessian_memory_mib,
            group_rank=plan.group.rank,
        )
        violations.extend(group_violations)
        candidate_atoms = plan.candidate_count * plan.atom_count
        atom_steps = candidate_atoms * relaxation.max_steps
        optimizer_state_bytes = plan.candidate_count * int(estimate["bytes"])
        total_candidate_atoms += candidate_atoms
        total_atom_steps += atom_steps
        summed_optimizer_state_bytes += optimizer_state_bytes
        peak_optimizer_state_bytes = max(
            peak_optimizer_state_bytes, int(estimate["bytes"])
        )
        group_payloads.append(
            {
                "group_rank": plan.group.rank,
                "qpoint": list(plan.group.qpoint),
                "qpoint_fractions": [str(value) for value in plan.rational_qpoint],
                "band_indices": list(plan.group.band_indices),
                "minimum_frequency_thz": plan.group.minimum_frequency_thz,
                "degeneracy": plan.group.degeneracy,
                "supercell_matrix": plan.matrix.tolist(),
                "supercell_determinant": plan.determinant,
                "atoms_per_candidate": plan.atom_count,
                "direction_count_modulo_sign": plan.direction_count,
                "physical_candidate_count": plan.candidate_count,
                "total_candidate_atoms": candidate_atoms,
                "maximum_atom_steps": atom_steps,
                "optimizer_state_per_candidate": estimate,
                "summed_optimizer_state_bytes": optimizer_state_bytes,
                "violations": list(group_violations),
            }
        )

    return {
        "schema_version": 1,
        "status": "refused" if violations else "accepted",
        "selection_policy": "sequential_ranked_group_fallback",
        "selected_basin_count": len(plans),
        "candidate_relaxation": relaxation.model_dump(mode="json"),
        "limits": {
            "max_candidates_per_iteration": max_candidates,
            "max_candidate_atoms": max_candidate_atoms,
            "max_dense_hessian_memory_mib": max_dense_hessian_memory_mib,
        },
        "groups": group_payloads,
        "totals": {
            "candidate_count": total_candidates,
            "candidates_already_generated": candidates_already_generated,
            "cumulative_candidate_count": cumulative_candidates,
            "candidate_atoms": total_candidate_atoms,
            "maximum_atom_steps": total_atom_steps,
            "peak_optimizer_state_bytes": peak_optimizer_state_bytes,
            "peak_optimizer_state_mib": peak_optimizer_state_bytes / 1024**2,
            "summed_optimizer_state_bytes": summed_optimizer_state_bytes,
            "summed_optimizer_state_mib": summed_optimizer_state_bytes / 1024**2,
        },
        "violations": violations,
        "notes": [
            "Atom-step totals assume every candidate reaches the configured max_steps.",
            "Optimizer estimates exclude calculator model, graph, and activation memory.",
            "Optimizer byte estimates are raw arrays; serialized checkpoints may be larger.",
            "Candidates run sequentially, so peak optimizer state is per candidate.",
        ],
    }


def _validate_phonon(phonon: Phonopy, mesh: MeshData) -> None:
    _validated_mesh_arrays(mesh)
    force_constants = phonon.force_constants
    if force_constants is None or not np.all(np.isfinite(force_constants)):
        raise SoftModeError("completed Phonopy model has invalid force constants")


def _generate_group_candidates(
    phonon: Phonopy,
    plan: _GroupGenerationPlan,
    settings: SoftModeConfig,
    temporary_root: Path,
    public_root: Path,
) -> tuple[CommensurateSupercell, tuple[DistortionCandidate, ...]]:
    group = plan.group
    rational_qpoint = plan.rational_qpoint
    matrix = plan.matrix
    determinant = plan.determinant
    qpoint = np.asarray([float(value) for value in rational_qpoint], dtype=float)
    group_name = f"group_{group.rank:03d}_{_qpoint_slug(rational_qpoint)}"
    working_dir = temporary_root / group_name
    public_dir = public_root / group_name
    working_dir.mkdir(parents=True)
    expected_atoms = plan.atom_count

    phonon_modes = [
        [qpoint.tolist(), int(band_index), 1.0, settings.phase_degrees]
        for band_index in group.band_indices
    ]
    try:
        modulation = phonon.run_modulations(
            dimension=matrix,
            phonon_modes=phonon_modes,
        )
    except Exception as exc:
        raise SoftModeError(
            f"Phonopy modulation failed for group {group.rank}: {exc}"
        ) from exc
    base = phonopy_to_ase(modulation.supercell)
    validate_structure(base)
    if len(base) != expected_atoms:
        raise SoftModeError("Phonopy modulation returned an unexpected atom count")
    write(working_dir / "reference.extxyz", base, format="extxyz")
    raw = np.asarray(modulation.modulations, dtype=complex)
    if raw.shape != (group.degeneracy, len(base), 3):
        raise SoftModeError(f"Phonopy returned invalid modulation shape {raw.shape}")
    fields = [np.asarray(field.real, dtype=float) for field in raw]
    returned_frequencies = np.asarray(modulation.frequencies, dtype=float)
    if returned_frequencies.shape != (group.degeneracy,):
        raise SoftModeError(
            f"Phonopy returned invalid frequency shape {returned_frequencies.shape}"
        )
    for returned_frequency, expected_frequency in zip(
        returned_frequencies, group.frequencies_thz, strict=True
    ):
        if not math.isclose(
            float(returned_frequency),
            expected_frequency,
            rel_tol=0,
            abs_tol=max(1.0e-7, settings.degeneracy_tolerance_thz),
        ):
            raise SoftModeError("mesh and modulation frequencies disagree")

    generated: list[DistortionCandidate] = []
    for direction_index, coefficients in enumerate(
        ternary_directions(group.degeneracy)
    ):
        combined = np.sum(
            np.asarray(coefficients, dtype=float)[:, None, None]
            * np.asarray(fields, dtype=float),
            axis=0,
        )
        mean = float(np.mean(np.linalg.norm(combined, axis=1)))
        if not np.isfinite(mean) or mean <= 1.0e-15:
            raise SoftModeError(
                f"group {group.rank} direction {coefficients} has a zero real field"
            )
        normalized = combined * (settings.mean_displacement_angstrom / mean)
        for sign, label in ((1, "plus"), (-1, "minus")):
            candidate = base.copy()
            candidate.positions = base.positions + sign * normalized
            candidate.wrap(eps=settings.qpoint_tolerance)
            validate_structure(candidate)
            actual, _ = find_mic(
                candidate.positions - base.positions, cell=base.cell, pbc=True
            )
            statistics = _displacement_statistics(actual)
            if not math.isclose(
                statistics.mean_angstrom,
                settings.mean_displacement_angstrom,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            ):
                raise SoftModeError("periodic wrapping changed the target displacement")
            candidate_id = f"g{group.rank:03d}_d{direction_index:04d}_{label}"
            filename = f"{candidate_id}.extxyz"
            write(working_dir / filename, candidate, format="extxyz")
            generated.append(
                DistortionCandidate(
                    candidate_id=candidate_id,
                    group_rank=group.rank,
                    band_indices=group.band_indices,
                    coefficients=coefficients,
                    frequencies_thz=group.frequencies_thz,
                    sign=sign,
                    phase_degrees=settings.phase_degrees,
                    target_mean_displacement_angstrom=(
                        settings.mean_displacement_angstrom
                    ),
                    displacement_statistics=statistics,
                    structure_path=public_dir / filename,
                )
            )
    supercell = CommensurateSupercell(
        group_rank=group.rank,
        qpoint_fractions=tuple(str(value) for value in rational_qpoint),
        matrix=tuple(tuple(int(value) for value in row) for row in matrix),
        determinant=determinant,
        atom_count=expected_atoms,
        reference_structure=public_dir / "reference.extxyz",
    )
    return supercell, tuple(generated)


def _displacement_statistics(displacements: np.ndarray) -> DisplacementStatistics:
    magnitudes = np.linalg.norm(np.asarray(displacements, dtype=float), axis=1)
    if len(magnitudes) == 0 or not np.all(np.isfinite(magnitudes)):
        raise SoftModeError("generated displacement magnitudes are invalid")
    return DisplacementStatistics(
        mean_angstrom=float(np.mean(magnitudes)),
        rms_angstrom=float(np.sqrt(np.mean(magnitudes**2))),
        maximum_angstrom=float(np.max(magnitudes)),
    )


def _qpoint_slug(qpoint: tuple[Fraction, Fraction, Fraction]) -> str:
    return "q_" + "_".join(
        f"{'m' if value < 0 else 'p'}{abs(value).numerator}d{abs(value).denominator}"
        for value in qpoint
    )


def _group_payload(group: SoftModeGroup) -> dict[str, Any]:
    return {**asdict(group), "degeneracy": group.degeneracy}


def _group_from_payload(item: dict[str, Any]) -> SoftModeGroup:
    return SoftModeGroup(
        rank=int(item["rank"]),
        qpoint_index=int(item["qpoint_index"]),
        qpoint=tuple(float(value) for value in item["qpoint"]),
        weight=int(item["weight"]),
        band_indices=tuple(int(value) for value in item["band_indices"]),
        frequencies_thz=tuple(float(value) for value in item["frequencies_thz"]),
        minimum_frequency_thz=float(item["minimum_frequency_thz"]),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
