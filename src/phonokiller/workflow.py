"""Unified resumable ASE-to-Phonopy structural-stability search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable

import ase
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write
import numpy as np
from phonopy import Phonopy, load as load_phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS
import phonopy
import yaml

from .candidates import (
    make_nonideal_primitive,
    reduce_candidates,
    structures_equivalent,
)
from .config import RunConfig
from .exceptions import (
    CandidateReductionError,
    DisplacementError,
    OutputDirectoryError,
    ResumeMismatchError,
)
from .instability import generate_soft_mode_candidates, rank_soft_modes
from .models import (
    ArtifactPaths,
    CalculationContext,
    CalculatorFactory,
    IterationSummary,
    MeshData,
    RunResult,
    RunStatus,
)
from .relaxation import calculator_for, relax_atoms
from .structure import ase_to_phonopy, load_structure, phonopy_to_ase, validate_structure


_MANIFEST_SCHEMA = 2
_HISTORY_SCHEMA = 1
_SECRET_KEY = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|credential)", re.I
)
_TERMINAL_STATUSES = {"stable", "cycle_detected", "max_evaluations"}


@dataclass(frozen=True, slots=True)
class _IterationPaths:
    directory: Path
    evaluated_structure: Path
    accepted_primitive: Path
    relaxation_dir: Path
    relaxation_structure: Path
    relaxation_trajectory: Path
    phonopy_dir: Path
    phonopy_manifest: Path
    phonopy_parameters: Path
    force_constants: Path
    mesh_yaml: Path
    mesh_arrays: Path
    phonopy_settings: Path
    instabilities_dir: Path
    candidates_dir: Path
    selection: Path
    selected_structure: Path


def run_workflow(
    structure: Atoms | str | Path,
    calculator: Calculator | CalculatorFactory | Callable[..., Calculator],
    config: RunConfig | dict[str, Any],
    output_dir: str | Path,
    resume: bool = True,
    *,
    format: str | None = None,
    index: int | str = -1,
) -> RunResult:
    """Run the complete iterative search until stable or safely terminated."""

    run_config = config if isinstance(config, RunConfig) else RunConfig.model_validate(config)
    input_atoms = load_structure(structure, format=format, index=index)
    artifacts = _artifact_paths(Path(output_dir).resolve())
    fingerprint_payload = _fingerprint_payload(input_atoms, run_config, calculator)
    fingerprint = _hash_payload(fingerprint_payload)
    manifest = _prepare_output(
        artifacts,
        input_atoms,
        run_config,
        fingerprint_payload,
        fingerprint,
        resume,
    )
    history_payload = _load_history(artifacts)
    history: list[dict[str, Any]] = history_payload["iterations"]
    terminal_status = (
        manifest.get("status")
        if manifest.get("status") in _TERMINAL_STATUSES
        else (
            history[-1].get("status")
            if history and history[-1].get("status") in _TERMINAL_STATUSES
            else None
        )
    )
    if terminal_status is not None:
        _repair_terminal_artifacts(
            manifest,
            artifacts,
            run_config,
            history,
            terminal_status,
        )
        return load_workflow_result(artifacts.output_dir)
    try:
        manifest.update(status="running", error=None)
        _atomic_json(artifacts.manifest, manifest)
        while True:
            iteration_index = len(history)
            paths = _iteration_paths(artifacts, iteration_index)
            paths.directory.mkdir(parents=True, exist_ok=True)
            manifest["stage"] = f"iteration:{iteration_index}:structure"
            _atomic_json(artifacts.manifest, manifest)

            if iteration_index == 0:
                evaluated, parent_energy_per_atom = _initial_relaxation(
                    input_atoms,
                    calculator,
                    run_config,
                    paths,
                )
            else:
                previous = history[-1]
                selected_path = Path(previous["selected_structure"])
                evaluated = load_structure(selected_path)
                parent_energy_per_atom = float(previous["selected_energy_per_atom_eV"])
                if not paths.evaluated_structure.exists():
                    write(paths.evaluated_structure, evaluated, format="extxyz")

            if not paths.accepted_primitive.exists():
                accepted = make_nonideal_primitive(evaluated, run_config)
                write(paths.accepted_primitive, accepted, format="extxyz")

            manifest["stage"] = f"iteration:{iteration_index}:phonopy"
            _atomic_json(artifacts.manifest, manifest)
            phonon, mesh, phonopy_metadata = _run_phonopy_evaluation(
                evaluated,
                calculator,
                run_config,
                paths,
                iteration_index=iteration_index,
                resume=resume,
            )
            groups = rank_soft_modes(mesh, run_config.soft_modes)
            minimum_frequency = float(np.min(mesh.frequencies))
            base_entry = {
                "index": iteration_index,
                "evaluated_structure": str(paths.evaluated_structure),
                "accepted_primitive": str(paths.accepted_primitive),
                "minimum_frequency_thz": minimum_frequency,
                "number_of_soft_mode_groups": len(groups),
                "supercell_matrix": phonopy_metadata["supercell_matrix"],
                "supercell_face_heights_angstrom": phonopy_metadata[
                    "unitcell_face_heights_angstrom"
                ],
                "supercell_spans_angstrom": phonopy_metadata[
                    "supercell_spans_angstrom"
                ],
                "supercell_atom_count": phonopy_metadata["supercell_atom_count"],
                "mesh_length": run_config.phonopy.mesh_length,
                "mesh_numbers": mesh.mesh_numbers.tolist(),
                "phonopy_directory": str(paths.phonopy_dir),
                "instability_report": str(
                    paths.instabilities_dir / "soft_modes.json"
                ),
                "soft_mode_groups": [_jsonable(asdict(group)) for group in groups],
                "parent_energy_per_atom_eV": parent_energy_per_atom,
            }

            if not groups:
                _write_ranking_only_report(paths, run_config, groups, mesh)
                entry = {**base_entry, "status": "stable"}
                history.append(entry)
                _write_history(artifacts, history)
                _export_final(artifacts, paths, evaluated, run_config)
                _finish_manifest(manifest, artifacts, "stable", history)
                return _result_from_components(
                    "stable", evaluated, phonon, mesh, history, artifacts
                )

            if iteration_index + 1 >= run_config.search.max_evaluations:
                _write_ranking_only_report(paths, run_config, groups, mesh)
                entry = {**base_entry, "status": "max_evaluations"}
                history.append(entry)
                _write_history(artifacts, history)
                _finish_manifest(manifest, artifacts, "max_evaluations", history)
                return _result_from_components(
                    "max_evaluations", evaluated, phonon, mesh, history, artifacts
                )

            manifest["stage"] = f"iteration:{iteration_index}:instabilities"
            _atomic_json(artifacts.manifest, manifest)
            soft_result = generate_soft_mode_candidates(
                phonon,
                mesh,
                run_config.soft_modes,
                paths.instabilities_dir,
                max_candidates=run_config.search.max_candidates_per_iteration,
                source_fingerprint=manifest["fingerprint"],
            )

            manifest["stage"] = f"iteration:{iteration_index}:candidates"
            _atomic_json(artifacts.manifest, manifest)
            reduced = reduce_candidates(
                soft_result.candidates,
                calculator,
                run_config,
                paths.candidates_dir,
                iteration_index=iteration_index,
                resume=resume,
            )
            successful = [item for item in reduced.candidates if item.status == "success"]
            if not successful:
                raise CandidateReductionError(
                    f"all {len(reduced.candidates)} candidate relaxations failed"
                )
            representatives = [
                reduced.candidates[group.representative_index]
                for group in reduced.duplicate_groups
            ]
            winner = min(
                representatives,
                key=lambda item: (
                    float(item.energy_per_atom_eV),
                    float(item.max_force_eV_per_A),
                    item.candidate_id,
                ),
            )
            assert winner.primitive_atoms is not None
            write(paths.selected_structure, winner.primitive_atoms, format="extxyz")
            energy_change = float(winner.energy_per_atom_eV) - parent_energy_per_atom
            ranking = sorted(
                representatives,
                key=lambda item: (
                    float(item.energy_per_atom_eV),
                    float(item.max_force_eV_per_A),
                    item.candidate_id,
                ),
            )
            selection_payload = {
                "selected_candidate_id": winner.candidate_id,
                "selected_candidate_index": winner.index,
                "selected_structure": str(paths.selected_structure),
                "selected_energy_per_atom_eV": winner.energy_per_atom_eV,
                "selected_max_force_eV_per_A": winner.max_force_eV_per_A,
                "energy_change_per_atom_eV": energy_change,
                "ranking": [
                    {
                        "rank": rank,
                        "candidate_id": item.candidate_id,
                        "candidate_index": item.index,
                        "energy_per_atom_eV": item.energy_per_atom_eV,
                        "max_force_eV_per_A": item.max_force_eV_per_A,
                        "duplicate_group": item.duplicate_group,
                    }
                    for rank, item in enumerate(ranking, 1)
                ],
            }
            _atomic_json(paths.selection, selection_payload)

            previous_primitives = [
                load_structure(Path(item["accepted_primitive"])) for item in history
            ]
            previous_primitives.append(load_structure(paths.accepted_primitive))
            cycle_index = next(
                (
                    previous_index
                    for previous_index, previous_atoms in enumerate(previous_primitives)
                    if structures_equivalent(
                        winner.primitive_atoms, previous_atoms, run_config
                    )
                ),
                None,
            )
            failed_count = len(reduced.candidates) - len(successful)
            selection_fields = {
                "selected_candidate_id": winner.candidate_id,
                "selected_structure": str(paths.selected_structure),
                "selected_energy_per_atom_eV": winner.energy_per_atom_eV,
                "selected_max_force_eV_per_A": winner.max_force_eV_per_A,
                "energy_change_per_atom_eV": energy_change,
                "number_of_generated_candidates": len(soft_result.candidates),
                "number_of_successful_candidates": len(successful),
                "number_of_failed_candidates": failed_count,
                "number_of_unique_candidates": len(reduced.duplicate_groups),
                "selection_report": str(paths.selection),
            }
            if cycle_index is not None:
                entry = {
                    **base_entry,
                    **selection_fields,
                    "status": "cycle_detected",
                    "cycle_matches_iteration": cycle_index,
                }
                history.append(entry)
                _write_history(artifacts, history)
                _finish_manifest(manifest, artifacts, "cycle_detected", history)
                return _result_from_components(
                    "cycle_detected", evaluated, phonon, mesh, history, artifacts
                )

            history.append({**base_entry, **selection_fields, "status": "selected"})
            _write_history(artifacts, history)
            manifest["completed_evaluations"] = len(history)
            manifest["stage"] = f"iteration:{iteration_index}:complete"
            _atomic_json(artifacts.manifest, manifest)
    except BaseException as exc:
        manifest.update(
            status="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _atomic_json(artifacts.manifest, manifest)
        raise


def load_workflow_result(output_dir: str | Path) -> RunResult:
    """Load a terminal stable or unresolved workflow result."""

    artifacts = _artifact_paths(Path(output_dir).resolve())
    if not artifacts.manifest.exists():
        raise OutputDirectoryError(f"workflow manifest does not exist: {artifacts.manifest}")
    manifest = _read_json(artifacts.manifest)
    status = manifest.get("status")
    if status not in _TERMINAL_STATUSES:
        raise OutputDirectoryError(f"workflow is not terminal (status={status!r})")
    history = _load_history(artifacts)["iterations"]
    if not history:
        raise OutputDirectoryError("terminal workflow has no iteration history")
    last = history[-1]
    iteration_paths = _iteration_paths(artifacts, int(last["index"]))
    atoms = load_structure(Path(last["evaluated_structure"]))
    phonon, mesh = _load_phonopy_result(iteration_paths)
    return _result_from_components(status, atoms, phonon, mesh, history, artifacts)


def automatic_supercell_matrix(
    atoms: Atoms, minimum_span_angstrom: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return diagonal repeats whose face-to-face spans reach the target."""

    if not np.isfinite(minimum_span_angstrom) or minimum_span_angstrom <= 0:
        raise ValueError("minimum supercell span must be finite and positive")
    validate_structure(atoms)
    cell = np.asarray(atoms.cell.array, dtype=float)
    volume = abs(float(np.linalg.det(cell)))
    face_areas = np.asarray(
        [
            np.linalg.norm(np.cross(cell[1], cell[2])),
            np.linalg.norm(np.cross(cell[2], cell[0])),
            np.linalg.norm(np.cross(cell[0], cell[1])),
        ],
        dtype=float,
    )
    heights = volume / face_areas
    repeats = np.maximum(1, np.ceil(minimum_span_angstrom / heights).astype(int))
    matrix = np.diag(repeats)
    spans = heights * repeats
    atom_count = int(len(atoms) * np.prod(repeats))
    return matrix, heights, spans, atom_count


def _initial_relaxation(
    atoms: Atoms,
    calculator: Calculator | Callable[..., Calculator],
    config: RunConfig,
    paths: _IterationPaths,
) -> tuple[Atoms, float]:
    metrics_path = paths.relaxation_dir / "metrics.json"
    if paths.relaxation_structure.exists() and metrics_path.exists():
        relaxed = load_structure(paths.relaxation_structure)
        metrics = _read_json(metrics_path)
    else:
        outcome = relax_atoms(
            atoms,
            calculator,
            config.relaxation,
            context=CalculationContext(
                stage="relaxation",
                workdir=paths.relaxation_dir,
                iteration_index=0,
            ),
            relaxed_structure=paths.relaxation_structure,
            trajectory_path=paths.relaxation_trajectory,
        )
        relaxed = outcome.atoms
        metrics = outcome.metrics
    if not paths.evaluated_structure.exists():
        write(paths.evaluated_structure, relaxed, format="extxyz")
    return relaxed, float(metrics["energy_eV"]) / len(relaxed)


def _run_phonopy_evaluation(
    atoms: Atoms,
    calculator: Calculator | Callable[..., Calculator],
    config: RunConfig,
    paths: _IterationPaths,
    *,
    iteration_index: int,
    resume: bool,
) -> tuple[Phonopy, MeshData, dict[str, Any]]:
    complete_files = (
        paths.phonopy_parameters,
        paths.force_constants,
        paths.mesh_yaml,
        paths.mesh_arrays,
        paths.phonopy_settings,
    )
    if all(path.exists() for path in complete_files):
        phonon, mesh = _load_phonopy_result(paths)
        return phonon, mesh, _read_json(paths.phonopy_settings)

    paths.phonopy_dir.mkdir(parents=True, exist_ok=True)
    displacement_root = paths.phonopy_dir / "displacements"
    displacement_root.mkdir(exist_ok=True)
    matrix, heights, spans, atom_count = automatic_supercell_matrix(
        atoms, config.phonopy.minimum_supercell_span_angstrom
    )
    phonon = Phonopy(ase_to_phonopy(atoms), supercell_matrix=matrix)
    phonon.generate_displacements()
    displaced_supercells = phonon.supercells_with_displacements
    if not displaced_supercells:
        raise DisplacementError("Phonopy generated no displaced supercells")
    phonopy_manifest = (
        _read_json(paths.phonopy_manifest)
        if paths.phonopy_manifest.exists()
        else {
            "schema_version": 1,
            "status": "running",
            "completed_displacements": [],
        }
    )
    forces: list[np.ndarray] = []
    completed: list[int] = []
    for displacement_index, phonopy_supercell in enumerate(displaced_supercells):
        supercell = phonopy_to_ase(phonopy_supercell)
        directory = displacement_root / f"{displacement_index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        structure_path = directory / "structure.extxyz"
        result_path = directory / "result.npz"
        metadata_path = directory / "result.json"
        structure_hash = _hash_payload(_atoms_payload(supercell))
        if result_path.exists() and metadata_path.exists() and resume:
            force = _load_displacement_checkpoint(
                result_path,
                metadata_path,
                displacement_index,
                structure_hash,
                len(supercell),
            )
        else:
            if result_path.exists() or metadata_path.exists():
                raise ResumeMismatchError(
                    f"incomplete displacement checkpoint {displacement_index}"
                )
            write(structure_path, supercell, format="extxyz")
            supercell.calc = calculator_for(
                calculator,
                CalculationContext(
                    stage="displacement",
                    workdir=directory,
                    iteration_index=iteration_index,
                    displacement_index=displacement_index,
                ),
            )
            try:
                force = np.asarray(supercell.get_forces(), dtype=float)
            except Exception as exc:
                raise DisplacementError(
                    f"force evaluation failed for displacement {displacement_index}: {exc}"
                ) from exc
            _validate_forces(force, len(supercell), f"displacement {displacement_index}")
            _save_displacement_checkpoint(
                result_path,
                metadata_path,
                displacement_index,
                structure_hash,
                force,
            )
        forces.append(force)
        completed.append(displacement_index)
        phonopy_manifest["completed_displacements"] = completed.copy()
        _atomic_json(paths.phonopy_manifest, phonopy_manifest)

    phonon.forces = np.asarray(forces, dtype=float)
    phonon.produce_force_constants()
    if phonon.force_constants is None or not np.all(np.isfinite(phonon.force_constants)):
        raise DisplacementError("Phonopy produced invalid force constants")
    phonon.save(
        filename=str(paths.phonopy_parameters),
        settings={"force_sets": True, "displacements": True, "force_constants": True},
    )
    write_FORCE_CONSTANTS(phonon.force_constants, filename=str(paths.force_constants))
    mesh_result = phonon.run_mesh(
        float(config.phonopy.mesh_length), with_eigenvectors=True
    )
    required = ("qpoints", "weights", "frequencies", "eigenvectors", "mesh_numbers")
    if any(getattr(mesh_result, name, None) is None for name in required):
        raise DisplacementError("Phonopy mesh result is incomplete")
    mesh = MeshData(
        qpoints=np.asarray(mesh_result.qpoints, dtype=float),
        weights=np.asarray(mesh_result.weights, dtype=int),
        frequencies=np.asarray(mesh_result.frequencies, dtype=float),
        eigenvectors=np.asarray(mesh_result.eigenvectors, dtype=complex),
        mesh_numbers=np.asarray(mesh_result.mesh_numbers, dtype=int),
        mesh_length=float(config.phonopy.mesh_length),
    )
    if not (
        np.all(np.isfinite(mesh.qpoints))
        and np.all(np.isfinite(mesh.frequencies))
        and np.all(np.isfinite(mesh.eigenvectors.real))
        and np.all(np.isfinite(mesh.eigenvectors.imag))
    ):
        raise DisplacementError("Phonopy mesh contains non-finite values")
    mesh_result.write_yaml(filename=str(paths.mesh_yaml))
    _atomic_npz(
        paths.mesh_arrays,
        qpoints=mesh.qpoints,
        weights=mesh.weights,
        frequencies=mesh.frequencies,
        eigenvectors=mesh.eigenvectors,
        mesh_numbers=mesh.mesh_numbers,
        mesh_length=np.asarray(mesh.mesh_length),
    )
    metadata = {
        "supercell_matrix": matrix.tolist(),
        "unitcell_face_heights_angstrom": heights.tolist(),
        "supercell_spans_angstrom": spans.tolist(),
        "supercell_atom_count": atom_count,
        "number_of_displacements": len(displaced_supercells),
        "mesh_length": mesh.mesh_length,
        "mesh_numbers": mesh.mesh_numbers.tolist(),
        "number_of_irreducible_qpoints": len(mesh.qpoints),
    }
    _atomic_json(paths.phonopy_settings, metadata)
    phonopy_manifest.update(status="complete", completed_displacements=completed)
    _atomic_json(paths.phonopy_manifest, phonopy_manifest)
    return phonon, mesh, metadata


def _load_phonopy_result(paths: _IterationPaths) -> tuple[Phonopy, MeshData]:
    try:
        phonon = load_phonopy(
            phonopy_yaml=str(paths.phonopy_parameters), produce_fc=False
        )
        with np.load(paths.mesh_arrays, allow_pickle=False) as arrays:
            mesh = MeshData(
                qpoints=np.asarray(arrays["qpoints"], dtype=float),
                weights=np.asarray(arrays["weights"], dtype=int),
                frequencies=np.asarray(arrays["frequencies"], dtype=float),
                eigenvectors=np.asarray(arrays["eigenvectors"], dtype=complex),
                mesh_numbers=np.asarray(arrays["mesh_numbers"], dtype=int),
                mesh_length=float(np.asarray(arrays["mesh_length"])),
            )
    except Exception as exc:
        raise OutputDirectoryError(f"cannot load Phonopy iteration: {exc}") from exc
    return phonon, mesh


def _artifact_paths(output: Path) -> ArtifactPaths:
    final = output / "final"
    return ArtifactPaths(
        output_dir=output,
        manifest=output / "manifest.json",
        resolved_config=output / "config.resolved.yaml",
        fingerprint=output / "fingerprint.json",
        input_structure=output / "input.extxyz",
        iterations_dir=output / "iterations",
        history=output / "history.json",
        summary=output / "summary.json",
        final_dir=final,
        final_structure=final / "structure.extxyz",
        final_phonopy_parameters=final / "phonopy_params.yaml",
        final_force_constants=final / "FORCE_CONSTANTS",
        final_mesh_yaml=final / "mesh.yaml",
        final_mesh_arrays=final / "mesh_data.npz",
    )


def _iteration_paths(artifacts: ArtifactPaths, index: int) -> _IterationPaths:
    directory = artifacts.iterations_dir / f"{index:04d}"
    relaxation = directory / "relaxation"
    phonopy_dir = directory / "phonopy"
    return _IterationPaths(
        directory=directory,
        evaluated_structure=directory / "structure.extxyz",
        accepted_primitive=directory / "accepted_primitive.extxyz",
        relaxation_dir=relaxation,
        relaxation_structure=relaxation / "relaxed.extxyz",
        relaxation_trajectory=relaxation / "trajectory.traj",
        phonopy_dir=phonopy_dir,
        phonopy_manifest=phonopy_dir / "manifest.json",
        phonopy_parameters=phonopy_dir / "phonopy_params.yaml",
        force_constants=phonopy_dir / "FORCE_CONSTANTS",
        mesh_yaml=phonopy_dir / "mesh.yaml",
        mesh_arrays=phonopy_dir / "mesh_data.npz",
        phonopy_settings=phonopy_dir / "settings.json",
        instabilities_dir=directory / "instabilities",
        candidates_dir=directory / "candidates",
        selection=directory / "selection.json",
        selected_structure=directory / "selected.extxyz",
    )


def _prepare_output(
    artifacts: ArtifactPaths,
    atoms: Atoms,
    config: RunConfig,
    fingerprint_payload: dict[str, Any],
    fingerprint: str,
    resume: bool,
) -> dict[str, Any]:
    output = artifacts.output_dir
    if output.exists() and not output.is_dir():
        raise OutputDirectoryError(f"output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    entries = list(output.iterdir())
    if artifacts.manifest.exists():
        manifest = _read_json(artifacts.manifest)
        if manifest.get("schema_version") != _MANIFEST_SCHEMA:
            raise ResumeMismatchError("existing unified manifest schema is not supported")
        if manifest.get("fingerprint") != fingerprint:
            raise ResumeMismatchError(
                "output belongs to a different structure, configuration, calculator, "
                "or dependency version"
            )
        if not resume:
            raise OutputDirectoryError("matching output exists and resume is disabled")
        return manifest
    if entries:
        raise OutputDirectoryError("nonempty output directory has no unified manifest")
    artifacts.iterations_dir.mkdir(parents=True)
    write(artifacts.input_structure, atoms, format="extxyz")
    _atomic_yaml(artifacts.resolved_config, _redact(config.resolved_payload()))
    _atomic_json(
        artifacts.fingerprint,
        {"fingerprint": fingerprint, "inputs": _redact(fingerprint_payload)},
    )
    _atomic_json(
        artifacts.history,
        {"schema_version": _HISTORY_SCHEMA, "iterations": []},
    )
    manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "fingerprint": fingerprint,
        "status": "running",
        "stage": "initialization",
        "completed_evaluations": 0,
        "error": None,
    }
    _atomic_json(artifacts.manifest, manifest)
    return manifest


def _load_history(artifacts: ArtifactPaths) -> dict[str, Any]:
    payload = _read_json(artifacts.history)
    if payload.get("schema_version") != _HISTORY_SCHEMA:
        raise ResumeMismatchError("history schema is not supported")
    if not isinstance(payload.get("iterations"), list):
        raise OutputDirectoryError("workflow history has invalid iterations")
    return payload


def _write_history(artifacts: ArtifactPaths, history: list[dict[str, Any]]) -> None:
    _atomic_json(
        artifacts.history,
        {"schema_version": _HISTORY_SCHEMA, "iterations": history},
    )


def _finish_manifest(
    manifest: dict[str, Any],
    artifacts: ArtifactPaths,
    status: RunStatus,
    history: list[dict[str, Any]],
) -> None:
    manifest.update(
        status=status,
        stage="complete",
        completed_evaluations=len(history),
        error=None,
    )
    _atomic_json(artifacts.manifest, manifest)
    _atomic_json(
        artifacts.summary,
        {
            "status": status,
            "number_of_evaluations": len(history),
            "minimum_frequency_thz": history[-1]["minimum_frequency_thz"],
            "final_structure": (
                str(artifacts.final_structure) if status == "stable" else None
            ),
            "history": str(artifacts.history),
            "termination": status,
        },
    )


def _repair_terminal_artifacts(
    manifest: dict[str, Any],
    artifacts: ArtifactPaths,
    config: RunConfig,
    history: list[dict[str, Any]],
    status: RunStatus,
) -> None:
    """Finish idempotent terminal writes after an interrupted finalization."""

    if not history:
        raise OutputDirectoryError("terminal workflow has no history to recover")
    last = history[-1]
    paths = _iteration_paths(artifacts, int(last["index"]))
    if status == "stable":
        required = (
            artifacts.final_structure,
            artifacts.final_phonopy_parameters,
            artifacts.final_force_constants,
            artifacts.final_mesh_yaml,
            artifacts.final_mesh_arrays,
        )
        if not all(path.exists() for path in required):
            atoms = load_structure(Path(last["evaluated_structure"]))
            _export_final(artifacts, paths, atoms, config)
    _finish_manifest(manifest, artifacts, status, history)


def _write_ranking_only_report(
    paths: _IterationPaths,
    config: RunConfig,
    groups: tuple,
    mesh: MeshData,
) -> None:
    """Write instability analysis when candidate expansion is not entered."""

    report = paths.instabilities_dir / "soft_modes.json"
    if report.exists():
        return
    _atomic_json(
        report,
        {
            "schema_version": 2,
            "status": "analysis_only",
            "settings": config.soft_modes.model_dump(mode="json"),
            "mesh_numbers": mesh.mesh_numbers.tolist(),
            "soft_mode_groups": [_jsonable(asdict(group)) for group in groups],
            "generated_candidates": 0,
        },
    )


def _export_final(
    artifacts: ArtifactPaths,
    paths: _IterationPaths,
    atoms: Atoms,
    config: RunConfig,
) -> None:
    temporary = artifacts.output_dir / ".final.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    write(temporary / "structure.extxyz", atoms, format="extxyz")
    shutil.copy2(paths.phonopy_parameters, temporary / "phonopy_params.yaml")
    shutil.copy2(paths.force_constants, temporary / "FORCE_CONSTANTS")
    shutil.copy2(paths.mesh_yaml, temporary / "mesh.yaml")
    shutil.copy2(paths.mesh_arrays, temporary / "mesh_data.npz")
    _atomic_yaml(temporary / "config.resolved.yaml", _redact(config.resolved_payload()))
    if artifacts.final_dir.exists():
        shutil.rmtree(artifacts.final_dir)
    os.replace(temporary, artifacts.final_dir)


def _result_from_components(
    status: RunStatus,
    atoms: Atoms,
    phonon: Phonopy,
    mesh: MeshData,
    history: list[dict[str, Any]],
    artifacts: ArtifactPaths,
) -> RunResult:
    return RunResult(
        status=status,
        relaxed_atoms=atoms,
        phonon=phonon,
        mesh=mesh,
        iterations=tuple(_iteration_summary(item) for item in history),
        artifacts=artifacts,
    )


def _iteration_summary(payload: dict[str, Any]) -> IterationSummary:
    return IterationSummary(
        index=int(payload["index"]),
        status=payload["status"],
        evaluated_structure=Path(payload["evaluated_structure"]),
        minimum_frequency_thz=float(payload["minimum_frequency_thz"]),
        number_of_soft_mode_groups=int(payload["number_of_soft_mode_groups"]),
        supercell_matrix=tuple(
            tuple(int(value) for value in row) for row in payload["supercell_matrix"]
        ),
        mesh_numbers=tuple(int(value) for value in payload["mesh_numbers"]),
        selected_candidate_id=payload.get("selected_candidate_id"),
        selected_structure=(
            Path(payload["selected_structure"])
            if payload.get("selected_structure")
            else None
        ),
        selected_energy_per_atom_eV=payload.get("selected_energy_per_atom_eV"),
        energy_change_per_atom_eV=payload.get("energy_change_per_atom_eV"),
        candidate_failures=int(payload.get("number_of_failed_candidates", 0)),
    )


def _validate_forces(forces: np.ndarray, atom_count: int, label: str) -> None:
    if forces.shape != (atom_count, 3):
        raise DisplacementError(
            f"calculator returned force shape {forces.shape} for {label}; "
            f"expected {(atom_count, 3)}"
        )
    if not np.all(np.isfinite(forces)):
        raise DisplacementError(f"calculator returned non-finite forces for {label}")


def _save_displacement_checkpoint(
    result_path: Path,
    metadata_path: Path,
    index: int,
    structure_hash: str,
    forces: np.ndarray,
) -> None:
    _atomic_npz(result_path, forces=forces)
    _atomic_json(
        metadata_path,
        {
            "displacement_index": index,
            "structure_hash": structure_hash,
            "atom_count": len(forces),
        },
    )


def _load_displacement_checkpoint(
    result_path: Path,
    metadata_path: Path,
    index: int,
    structure_hash: str,
    atom_count: int,
) -> np.ndarray:
    try:
        metadata = _read_json(metadata_path)
        if metadata["displacement_index"] != index:
            raise ValueError("index mismatch")
        if metadata["structure_hash"] != structure_hash:
            raise ValueError("structure fingerprint mismatch")
        with np.load(result_path, allow_pickle=False) as result:
            forces = np.asarray(result["forces"], dtype=float)
    except Exception as exc:
        raise ResumeMismatchError(f"invalid displacement checkpoint {index}: {exc}") from exc
    _validate_forces(forces, atom_count, f"resumed displacement {index}")
    return forces


def _fingerprint_payload(
    atoms: Atoms,
    config: RunConfig,
    calculator: Calculator | Callable[..., Calculator],
) -> dict[str, Any]:
    return {
        "structure": _atoms_payload(atoms),
        "config": config.resolved_payload(),
        "calculator": _calculator_identity(calculator),
        "versions": {
            "ase": ase.__version__,
            "phonopy": phonopy.__version__,
            "phonokiller": _package_version(),
        },
    }


def _atoms_payload(atoms: Atoms) -> dict[str, Any]:
    payload = {
        "numbers": atoms.numbers.tolist(),
        "cell": np.asarray(atoms.cell.array, dtype=float).tolist(),
        "positions": np.asarray(atoms.positions, dtype=float).tolist(),
        "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
        "masses": np.asarray(atoms.get_masses(), dtype=float).tolist(),
    }
    if atoms.has("initial_magmoms"):
        payload["initial_magmoms"] = np.asarray(
            atoms.get_initial_magnetic_moments(), dtype=float
        ).tolist()
    return payload


def _calculator_identity(calculator: Calculator | Callable[..., Calculator]) -> Any:
    if isinstance(calculator, Calculator):
        try:
            parameters = calculator.todict()
        except Exception:
            parameters = {}
        return {
            "kind": "instance",
            "class": f"{type(calculator).__module__}:{type(calculator).__qualname__}",
            "parameters": _jsonable(parameters),
        }
    return {
        "kind": "factory",
        "callable": f"{getattr(calculator, '__module__', type(calculator).__module__)}:"
        f"{getattr(calculator, '__qualname__', type(calculator).__qualname__)}",
    }


def _package_version() -> str:
    try:
        return importlib.metadata.version("phonokiller")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": f"{type(value).__module__}:{type(value).__qualname__}"}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if _SECRET_KEY.search(str(key)) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OutputDirectoryError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OutputDirectoryError(f"JSON artifact is not an object: {path}")
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(_jsonable(payload), stream, sort_keys=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
