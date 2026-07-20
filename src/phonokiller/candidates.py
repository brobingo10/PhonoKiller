"""Internal candidate relaxation, primitive reduction, and deduplication."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Literal

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read, write
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
import numpy as np
import spglib
import yaml

from .config import RunConfig
from .exceptions import CandidateReductionError, OutputDirectoryError, ResumeMismatchError
from .models import (
    CalculationContext,
    CandidateReductionArtifactPaths,
    CandidateReductionResult,
    CandidateResult,
    DistortionCandidate,
    DuplicateGroup,
    ProgressCallback,
)
from .relaxation import relax_atoms
from .structure import load_structure, validate_structure


_SCHEMA = 2


@dataclass(slots=True)
class _SuccessfulCandidate:
    result: CandidateResult
    decoration_tokens: list[str]


def reduce_candidates(
    candidates: tuple[DistortionCandidate, ...],
    calculator: Calculator | Callable[..., Calculator],
    config: RunConfig,
    output_dir: str | Path,
    *,
    iteration_index: int,
    resume: bool = True,
    progress: ProgressCallback | None = None,
) -> CandidateReductionResult:
    """Relax exact generated candidates and deduplicate non-ideal primitives."""

    if not candidates:
        raise CandidateReductionError("no generated candidates were supplied")
    output = Path(output_dir).resolve()
    artifacts = _artifact_paths(output)
    fingerprint_payload = {
        "candidates": [
            {
                "candidate_id": item.candidate_id,
                "path": str(item.structure_path.resolve()),
                "sha256": _hash_file(item.structure_path),
            }
            for item in candidates
        ],
        "candidate_relaxation": config.effective_candidate_relaxation().model_dump(
            mode="json"
        ),
        "symmetry": config.symmetry.model_dump(mode="json"),
        "deduplication": config.deduplication.model_dump(mode="json"),
        "iteration_index": iteration_index,
    }
    fingerprint = _hash_payload(fingerprint_payload)
    manifest = _prepare_output(
        artifacts, config, fingerprint_payload, fingerprint, resume, len(candidates)
    )
    results: list[CandidateResult] = []
    successful: list[_SuccessfulCandidate] = []
    try:
        manifest.update(status="running", error=None)
        _atomic_json(artifacts.manifest, manifest)
        if progress is not None:
            progress(
                f"Candidate relaxation: processing {len(candidates)} generated "
                "structure(s) sequentially."
            )
        for candidate_index, candidate in enumerate(candidates):
            label = (
                f"Candidate {candidate_index + 1}/{len(candidates)} "
                f"({candidate.candidate_id})"
            )
            manifest["stage"] = f"candidate:{candidate_index}"
            checkpoint = _result_path(artifacts, candidate_index)
            if resume and checkpoint.exists():
                payload = _read_json(checkpoint)
                if payload.get("status") == "success":
                    resumed = _load_successful(
                        artifacts, candidate_index, candidate, payload
                    )
                    results.append(resumed.result)
                    successful.append(resumed)
                    manifest["checkpoints"][candidate_index] = "success"
                    _atomic_json(artifacts.manifest, manifest)
                    if progress is not None:
                        progress(f"{label}: reused successful checkpoint.")
                    continue
                _archive_failed_result(checkpoint)
            try:
                if progress is not None:
                    progress(f"{label}: relaxation started.")
                completed = _process_candidate(
                    candidate,
                    candidate_index,
                    calculator,
                    config,
                    artifacts,
                    iteration_index=iteration_index,
                    progress=progress,
                )
            except Exception as exc:
                failed = CandidateResult(
                    index=candidate_index,
                    candidate_id=candidate.candidate_id,
                    source=str(candidate.structure_path),
                    status="failed",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
                _atomic_json(checkpoint, _candidate_payload(failed))
                results.append(failed)
                manifest["checkpoints"][candidate_index] = "failed"
                if progress is not None:
                    progress(f"{label}: failed ({type(exc).__name__}: {exc}).")
            else:
                results.append(completed.result)
                successful.append(completed)
                manifest["checkpoints"][candidate_index] = "success"
                if progress is not None:
                    progress(
                        f"{label}: complete; energy "
                        f"{float(completed.result.energy_per_atom_eV):.8f} eV/atom; "
                        f"max force "
                        f"{float(completed.result.max_force_eV_per_A):.6f} "
                        "eV/Angstrom."
                    )
            _atomic_json(artifacts.manifest, manifest)

        manifest["stage"] = "deduplication"
        _atomic_json(artifacts.manifest, manifest)
        if progress is not None:
            progress(
                f"Deduplication: comparing {len(successful)} successful "
                "primitive structure(s)."
            )
        groups, unique_structures = _deduplicate(successful, config, artifacts)
        status: Literal["complete", "partial"] = (
            "complete" if len(successful) == len(candidates) else "partial"
        )
        _write_summary(results, groups, status, artifacts)
        manifest.update(status=status, stage="complete", error=None)
        _atomic_json(artifacts.manifest, manifest)
        if progress is not None:
            progress(
                f"Deduplication complete: {len(groups)} unique structure(s); "
                f"{len(candidates) - len(successful)} candidate failure(s)."
            )
        return CandidateReductionResult(
            status=status,
            candidates=results,
            duplicate_groups=groups,
            unique_structures=unique_structures,
            artifacts=artifacts,
        )
    except BaseException as exc:
        manifest.update(
            status="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _atomic_json(artifacts.manifest, manifest)
        raise


def make_nonideal_primitive(atoms: Atoms, config: RunConfig) -> Atoms:
    """Return the smallest spglib primitive without idealizing its geometry."""

    primitive, _, _ = _make_primitive(atoms, config)
    return primitive


def structures_equivalent(left: Atoms, right: Atoms, config: RunConfig) -> bool:
    """Compare already primitive structures with decorated-site awareness."""

    if len(left) != len(right):
        return False
    left_tokens = _decoration_tokens(left)
    right_tokens = _decoration_tokens(right)
    if Counter(left_tokens) != Counter(right_tokens):
        return False
    all_tokens = sorted(set(left_tokens + right_tokens))
    comparison_number = {token: index + 1 for index, token in enumerate(all_tokens)}
    left_cmp = Atoms(
        numbers=[comparison_number[token] for token in left_tokens],
        cell=left.cell.array,
        scaled_positions=left.get_scaled_positions(wrap=True),
        pbc=True,
    )
    right_cmp = Atoms(
        numbers=[comparison_number[token] for token in right_tokens],
        cell=right.cell.array,
        scaled_positions=right.get_scaled_positions(wrap=True),
        pbc=True,
    )
    return _structures_equivalent(left_cmp, right_cmp, config)


def _artifact_paths(output: Path) -> CandidateReductionArtifactPaths:
    return CandidateReductionArtifactPaths(
        output_dir=output,
        manifest=output / "manifest.json",
        resolved_config=output / "config.resolved.yaml",
        fingerprint=output / "fingerprint.json",
        candidates_dir=output / "items",
        unique_dir=output / "unique",
        deduplication=output / "deduplication.json",
        summary=output / "summary.json",
    )


def _prepare_output(
    artifacts: CandidateReductionArtifactPaths,
    config: RunConfig,
    fingerprint_payload: dict[str, Any],
    fingerprint: str,
    resume: bool,
    count: int,
) -> dict[str, Any]:
    output = artifacts.output_dir
    if output.exists() and not output.is_dir():
        raise OutputDirectoryError(f"candidate output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    entries = list(output.iterdir())
    if artifacts.manifest.exists():
        manifest = _read_json(artifacts.manifest)
        if manifest.get("schema_version") != _SCHEMA:
            raise ResumeMismatchError("candidate manifest schema is not supported")
        if manifest.get("fingerprint") != fingerprint:
            raise ResumeMismatchError("candidate checkpoint fingerprint changed")
        if not resume:
            raise OutputDirectoryError("candidate output exists and resume is disabled")
        return manifest
    if entries:
        raise OutputDirectoryError("nonempty candidate output has no manifest")
    artifacts.candidates_dir.mkdir()
    artifacts.unique_dir.mkdir()
    _atomic_yaml(
        artifacts.resolved_config,
        {
            "candidate_relaxation": config.effective_candidate_relaxation().model_dump(
                mode="json"
            ),
            "symmetry": config.symmetry.model_dump(mode="json"),
            "deduplication": config.deduplication.model_dump(mode="json"),
        },
    )
    _atomic_json(artifacts.fingerprint, fingerprint_payload)
    manifest = {
        "schema_version": _SCHEMA,
        "fingerprint": fingerprint,
        "status": "running",
        "stage": "candidate:0",
        "error": None,
        "checkpoints": ["pending"] * count,
    }
    _atomic_json(artifacts.manifest, manifest)
    return manifest


def _candidate_dir(artifacts: CandidateReductionArtifactPaths, index: int) -> Path:
    return artifacts.candidates_dir / f"{index:04d}"


def _result_path(artifacts: CandidateReductionArtifactPaths, index: int) -> Path:
    return _candidate_dir(artifacts, index) / "result.json"


def _process_candidate(
    candidate: DistortionCandidate,
    candidate_index: int,
    calculator: Calculator | Callable[..., Calculator],
    config: RunConfig,
    artifacts: CandidateReductionArtifactPaths,
    *,
    iteration_index: int,
    progress: ProgressCallback | None,
) -> _SuccessfulCandidate:
    directory = _candidate_dir(artifacts, candidate_index)
    directory.mkdir(parents=True, exist_ok=True)
    atoms = load_structure(candidate.structure_path)
    original_composition = Counter(_decoration_tokens(atoms))
    write(directory / "input.extxyz", atoms, format="extxyz")
    relaxation_dir = directory / "relaxation"
    outcome = relax_atoms(
        atoms,
        calculator,
        config.effective_candidate_relaxation(),
        context=CalculationContext(
            stage="candidate_relaxation",
            workdir=relaxation_dir,
            iteration_index=iteration_index,
            candidate_index=candidate_index,
            candidate_id=candidate.candidate_id,
        ),
        relaxed_structure=relaxation_dir / "relaxed.extxyz",
        trajectory_path=relaxation_dir / "trajectory.traj",
        progress=progress,
    )
    if Counter(_decoration_tokens(outcome.atoms)) != original_composition:
        raise CandidateReductionError("candidate relaxation changed the composition")
    energy = float(outcome.metrics["energy_eV"])
    max_force = float(outcome.metrics["max_force_eV_per_A"])
    if not np.isfinite(energy) or not np.isfinite(max_force):
        raise CandidateReductionError("candidate metrics are non-finite")
    primitive, tokens, spacegroup = _make_primitive(outcome.atoms, config)
    primitive.info.update(
        spacegroup_number=spacegroup[0], spacegroup_symbol=spacegroup[1]
    )
    write(directory / "primitive.extxyz", primitive, format="extxyz")
    result = CandidateResult(
        index=candidate_index,
        candidate_id=candidate.candidate_id,
        source=str(candidate.structure_path),
        status="success",
        relaxed_atoms=outcome.atoms,
        primitive_atoms=primitive,
        energy_eV=energy,
        energy_per_atom_eV=energy / len(outcome.atoms),
        max_force_eV_per_A=max_force,
        spacegroup_number=spacegroup[0],
        spacegroup_symbol=spacegroup[1],
    )
    _atomic_json(_result_path(artifacts, candidate_index), _candidate_payload(result, tokens))
    return _SuccessfulCandidate(result=result, decoration_tokens=tokens)


def _make_primitive(
    atoms: Atoms, config: RunConfig
) -> tuple[Atoms, list[str], tuple[int, str]]:
    tokens = _decoration_tokens(atoms)
    unique_tokens = sorted(set(tokens))
    type_by_token = {token: index + 1 for index, token in enumerate(unique_tokens)}
    token_by_type = {value: key for key, value in type_by_token.items()}
    spglib_cell = (
        np.asarray(atoms.cell.array, dtype=float),
        np.mod(np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float), 1.0),
        np.asarray([type_by_token[token] for token in tokens], dtype=int),
    )
    standardized = spglib.standardize_cell(
        spglib_cell,
        to_primitive=True,
        no_idealize=True,
        symprec=config.symmetry.symprec,
        angle_tolerance=config.symmetry.angle_tolerance,
    )
    if standardized is None:
        raise CandidateReductionError("spglib could not construct a primitive cell")
    lattice, scaled_positions, type_ids = standardized
    primitive_tokens = [token_by_type[int(type_id)] for type_id in type_ids]
    decorations = [json.loads(token) for token in primitive_tokens]
    primitive = Atoms(
        numbers=[int(item["number"]) for item in decorations],
        masses=[float(item["mass"]) for item in decorations],
        cell=np.asarray(lattice, dtype=float),
        scaled_positions=np.mod(np.asarray(scaled_positions, dtype=float), 1.0),
        pbc=True,
    )
    magnetic = [item["magmom"] for item in decorations]
    if any(value is not None for value in magnetic):
        if any(value is None for value in magnetic):
            raise CandidateReductionError("inconsistent primitive magnetic decorations")
        primitive.set_initial_magnetic_moments(np.asarray(magnetic, dtype=float))
    validate_structure(primitive)
    dataset = spglib.get_symmetry_dataset(
        (lattice, scaled_positions, type_ids),
        symprec=config.symmetry.symprec,
        angle_tolerance=config.symmetry.angle_tolerance,
    )
    if dataset is None:
        raise CandidateReductionError("spglib could not identify primitive symmetry")
    return primitive, primitive_tokens, (int(dataset.number), str(dataset.international))


def _decoration_tokens(atoms: Atoms) -> list[str]:
    masses = np.asarray(atoms.get_masses(), dtype=float)
    magnetic = (
        np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
        if atoms.has("initial_magmoms")
        else None
    )
    tokens: list[str] = []
    for index, (number, mass) in enumerate(zip(atoms.numbers, masses, strict=True)):
        magmom: float | list[float] | None = None
        if magnetic is not None:
            value = magnetic[index]
            magmom = (
                float(np.round(value, 12))
                if np.ndim(value) == 0
                else np.round(value, 12).tolist()
            )
        tokens.append(
            json.dumps(
                {
                    "number": int(number),
                    "mass": float(np.round(mass, 12)),
                    "magmom": magmom,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return tokens


def _candidate_payload(
    result: CandidateResult, tokens: list[str] | None = None
) -> dict[str, Any]:
    payload = {
        "index": result.index,
        "candidate_id": result.candidate_id,
        "source": result.source,
        "status": result.status,
        "energy_eV": result.energy_eV,
        "energy_per_atom_eV": result.energy_per_atom_eV,
        "max_force_eV_per_A": result.max_force_eV_per_A,
        "spacegroup_number": result.spacegroup_number,
        "spacegroup_symbol": result.spacegroup_symbol,
        "duplicate_group": result.duplicate_group,
        "is_representative": result.is_representative,
        "error": result.error,
    }
    if tokens is not None:
        payload["primitive_decoration_tokens"] = tokens
    return payload


def _load_successful(
    artifacts: CandidateReductionArtifactPaths,
    index: int,
    candidate: DistortionCandidate,
    payload: dict[str, Any],
) -> _SuccessfulCandidate:
    directory = _candidate_dir(artifacts, index)
    try:
        relaxed = read(directory / "relaxation" / "relaxed.extxyz", index=-1)
        primitive = read(directory / "primitive.extxyz", index=-1)
        validate_structure(relaxed)
        validate_structure(primitive)
        tokens = list(payload["primitive_decoration_tokens"])
        result = CandidateResult(
            index=index,
            candidate_id=candidate.candidate_id,
            source=str(candidate.structure_path),
            status="success",
            relaxed_atoms=relaxed,
            primitive_atoms=primitive,
            energy_eV=float(payload["energy_eV"]),
            energy_per_atom_eV=float(payload["energy_per_atom_eV"]),
            max_force_eV_per_A=float(payload["max_force_eV_per_A"]),
            spacegroup_number=int(payload["spacegroup_number"]),
            spacegroup_symbol=str(payload["spacegroup_symbol"]),
        )
    except Exception as exc:
        raise ResumeMismatchError(f"invalid candidate checkpoint {index}: {exc}") from exc
    return _SuccessfulCandidate(result=result, decoration_tokens=tokens)


def _archive_failed_result(path: Path) -> None:
    attempt = 1
    while path.with_name(f"failed-attempt-{attempt:03d}.json").exists():
        attempt += 1
    os.replace(path, path.with_name(f"failed-attempt-{attempt:03d}.json"))


def _deduplicate(
    successful: list[_SuccessfulCandidate],
    config: RunConfig,
    artifacts: CandidateReductionArtifactPaths,
) -> tuple[list[DuplicateGroup], list[Atoms]]:
    all_tokens = sorted({token for item in successful for token in item.decoration_tokens})
    if len(all_tokens) > 118:
        raise CandidateReductionError("more than 118 decorated site types cannot be compared")
    comparison_number = {token: index + 1 for index, token in enumerate(all_tokens)}
    comparison_atoms = [_comparison_atoms(item, comparison_number) for item in successful]
    parent = list(range(len(successful)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    signatures = [Counter(item.decoration_tokens) for item in successful]
    for left in range(len(successful)):
        for right in range(left + 1, len(successful)):
            if signatures[left] == signatures[right] and _structures_equivalent(
                comparison_atoms[left], comparison_atoms[right], config
            ):
                union(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(len(successful)):
        grouped.setdefault(find(index), []).append(index)
    components = sorted(
        grouped.values(),
        key=lambda members: min(successful[item].result.index for item in members),
    )
    for old in artifacts.unique_dir.glob("unique_*.extxyz"):
        old.unlink()
    groups: list[DuplicateGroup] = []
    structures: list[Atoms] = []
    payloads: list[dict[str, Any]] = []
    for group_index, members in enumerate(components):
        representative_position = min(
            members,
            key=lambda item: (
                float(successful[item].result.energy_per_atom_eV),
                float(successful[item].result.max_force_eV_per_A),
                successful[item].result.candidate_id,
            ),
        )
        representative = successful[representative_position].result
        member_indices = tuple(sorted(successful[item].result.index for item in members))
        for member in members:
            successful[member].result.duplicate_group = group_index
        representative.is_representative = True
        structure = representative.primitive_atoms.copy()  # type: ignore[union-attr]
        structure_path = artifacts.unique_dir / f"unique_{group_index:04d}.extxyz"
        write(structure_path, structure, format="extxyz")
        structures.append(structure)
        groups.append(
            DuplicateGroup(
                index=group_index,
                representative_index=representative.index,
                member_indices=member_indices,
                structure_path=structure_path,
            )
        )
        payloads.append(
            {
                "group_index": group_index,
                "representative_index": representative.index,
                "member_indices": list(member_indices),
                "representative_candidate_id": representative.candidate_id,
                "representative_energy_per_atom_eV": representative.energy_per_atom_eV,
                "structure": str(structure_path),
            }
        )
    for item in successful:
        _atomic_json(
            _result_path(artifacts, item.result.index),
            _candidate_payload(item.result, item.decoration_tokens),
        )
    _atomic_json(
        artifacts.deduplication,
        {
            "number_of_successful_candidates": len(successful),
            "number_of_unique_structures": len(groups),
            "groups": payloads,
        },
    )
    return groups, structures


def _comparison_atoms(
    item: _SuccessfulCandidate, numbers: dict[str, int]
) -> Atoms:
    primitive = item.result.primitive_atoms
    assert primitive is not None
    return Atoms(
        numbers=[numbers[token] for token in item.decoration_tokens],
        cell=primitive.cell.array,
        scaled_positions=primitive.get_scaled_positions(wrap=True),
        pbc=True,
    )


def _structures_equivalent(left: Atoms, right: Atoms, config: RunConfig) -> bool:
    if len(left) != len(right):
        return False

    def compare(reference: Atoms, other: Atoms) -> bool:
        spacing = (reference.get_volume() / len(reference)) ** (1.0 / 3.0)
        matcher = SymmetryEquivalenceCheck(
            angle_tol=config.deduplication.cell_angle_tolerance_degrees,
            ltol=config.deduplication.cell_length_relative_tolerance * len(reference),
            stol=config.deduplication.site_tolerance_angstrom / spacing,
            vol_tol=config.deduplication.primitive_volume_tolerance_angstrom3,
            scale_volume=config.deduplication.scale_volume,
            to_primitive=False,
        )
        return bool(matcher.compare(reference, other))

    return compare(left, right) and compare(right, left)


def _write_summary(
    results: list[CandidateResult],
    groups: list[DuplicateGroup],
    status: str,
    artifacts: CandidateReductionArtifactPaths,
) -> None:
    failed = [item for item in results if item.status == "failed"]
    _atomic_json(
        artifacts.summary,
        {
            "status": status,
            "number_of_candidates": len(results),
            "number_of_successful_candidates": len(results) - len(failed),
            "number_of_failed_candidates": len(failed),
            "number_of_unique_structures": len(groups),
            "failed_candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "source": item.source,
                    "error": item.error,
                }
                for item in failed
            ],
        },
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OutputDirectoryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OutputDirectoryError(f"JSON artifact is not an object: {path}")
    return payload


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
    os.replace(temporary, path)
