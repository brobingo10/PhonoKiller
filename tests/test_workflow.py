from __future__ import annotations

import json
from pathlib import Path
import shutil

from ase import Atoms
from ase.io import write
import numpy as np
import pytest
from phonopy import Phonopy

from phonokiller import (
    CandidateReductionError,
    ResumeMismatchError,
    RunConfig,
    automatic_supercell_matrix,
    load_workflow_result,
    run_workflow,
)
from phonokiller.cli import _build_parser, main
from phonokiller.models import (
    CandidateReductionArtifactPaths,
    CandidateReductionResult,
    CandidateResult,
    DisplacementStatistics,
    DistortionCandidate,
    DuplicateGroup,
    ExcludedDuplicateGroup,
    MeshData,
    SoftModeResult,
)
import phonokiller.workflow as workflow
from tests.helpers import ZeroCalculator, ZeroStressCalculator


def simple_crystal() -> Atoms:
    return Atoms("Al", scaled_positions=[[0, 0, 0]], cell=[4.0, 4.0, 4.0], pbc=True)


def fast_config(**updates) -> RunConfig:
    payload = {
        "relaxation": {"force_tolerance": 100.0, "max_steps": 1},
        "phonopy": {
            "minimum_supercell_span_angstrom": 4.0,
            "mesh_length": 4.0,
        },
    }
    for section, values in updates.items():
        payload.setdefault(section, {}).update(values)
    return RunConfig.model_validate(payload)


def test_automatic_supercell_reaches_face_spans_without_atom_cap() -> None:
    atoms = Atoms(
        "Al2",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=[[3.0, 0.0, 0.0], [1.2, 4.0, 0.0], [0.5, 0.7, 5.0]],
        pbc=True,
    )
    matrix, heights, spans, atom_count = automatic_supercell_matrix(atoms, 10.0)
    assert np.all(spans >= 10.0)
    np.testing.assert_array_equal(matrix, np.diag(np.diag(matrix)))
    assert atom_count == len(atoms) * round(np.linalg.det(matrix))
    assert np.all(spans - heights < 10.0)


def test_stable_end_to_end_exports_and_resumes(tmp_path) -> None:
    output = tmp_path / "run"
    atoms = simple_crystal()
    config = fast_config()
    result = run_workflow(atoms, ZeroCalculator(), config, output)
    assert result.status == "stable"
    assert len(result.iterations) == 1
    assert result.artifacts.final_structure.exists()
    assert result.artifacts.final_phonopy_parameters.exists()
    assert result.artifacts.final_force_constants.exists()
    assert result.artifacts.final_mesh_yaml.exists()
    assert result.artifacts.final_mesh_arrays.exists()
    assert json.loads(result.artifacts.history.read_text())["iterations"][0][
        "status"
    ] == "stable"

    resumed = run_workflow(atoms, ZeroCalculator(), config, output)
    loaded = load_workflow_result(output)
    assert resumed.status == loaded.status == "stable"
    np.testing.assert_allclose(resumed.mesh.frequencies, loaded.mesh.frequencies)


def test_workflow_reports_live_progress_and_terminal_resume(tmp_path) -> None:
    output = tmp_path / "run"
    events: list[str] = []
    run_workflow(
        simple_crystal(),
        ZeroCalculator(),
        fast_config(),
        output,
        progress=events.append,
    )
    assert any("Initial relaxation started" in event for event in events)
    assert any("Initial relaxation: step" in event for event in events)
    assert any("Phonopy generated" in event for event in events)
    assert any("Displacement 1/" in event for event in events)
    assert any("Phonopy mesh complete" in event for event in events)
    assert events[-1].startswith("Workflow stable")

    resumed_events: list[str] = []
    run_workflow(
        simple_crystal(),
        ZeroCalculator(),
        fast_config(),
        output,
        progress=resumed_events.append,
    )
    assert any("terminal workflow checkpoint found" in event for event in resumed_events)


def test_loading_terminal_result_does_not_recompute_mesh(monkeypatch, tmp_path) -> None:
    output = tmp_path / "run"
    run_workflow(simple_crystal(), ZeroCalculator(), fast_config(), output)

    def unexpected(*args, **kwargs):
        raise AssertionError("loading a completed result must not rerun the mesh")

    monkeypatch.setattr(Phonopy, "run_mesh", unexpected)
    loaded = load_workflow_result(output)
    assert loaded.status == "stable"


def test_displacement_checkpoints_request_and_store_only_forces(tmp_path) -> None:
    output = tmp_path / "run"
    run_workflow(simple_crystal(), ZeroStressCalculator(), fast_config(), output)
    checkpoints = sorted(
        (output / "iterations" / "0000" / "phonopy" / "displacements").glob(
            "*/result.npz"
        )
    )
    assert checkpoints
    for checkpoint in checkpoints:
        with np.load(checkpoint, allow_pickle=False) as arrays:
            assert set(arrays.files) == {"forces"}


def test_terminal_history_repairs_interrupted_finalization(tmp_path) -> None:
    output = tmp_path / "run"
    atoms = simple_crystal()
    config = fast_config()
    result = run_workflow(atoms, ZeroCalculator(), config, output)
    shutil.rmtree(result.artifacts.final_dir)
    result.artifacts.summary.unlink()
    manifest = json.loads(result.artifacts.manifest.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    result.artifacts.manifest.write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    repaired = run_workflow(atoms, ZeroCalculator(), config, output)
    assert repaired.status == "stable"
    assert repaired.artifacts.final_structure.exists()
    assert repaired.artifacts.summary.exists()


def test_phonopy_defaults_are_not_overridden(monkeypatch, tmp_path) -> None:
    calls: dict[str, list[tuple[tuple, dict]]] = {
        "init": [],
        "generate": [],
        "produce": [],
        "mesh": [],
    }
    original_init = Phonopy.__init__
    original_generate = Phonopy.generate_displacements
    original_produce = Phonopy.produce_force_constants
    original_mesh = Phonopy.run_mesh

    def init_spy(self, *args, **kwargs):
        calls["init"].append((args, kwargs.copy()))
        return original_init(self, *args, **kwargs)

    def generate_spy(self, *args, **kwargs):
        calls["generate"].append((args, kwargs.copy()))
        return original_generate(self, *args, **kwargs)

    def produce_spy(self, *args, **kwargs):
        calls["produce"].append((args, kwargs.copy()))
        return original_produce(self, *args, **kwargs)

    def mesh_spy(self, *args, **kwargs):
        calls["mesh"].append((args, kwargs.copy()))
        return original_mesh(self, *args, **kwargs)

    monkeypatch.setattr(Phonopy, "__init__", init_spy)
    monkeypatch.setattr(Phonopy, "generate_displacements", generate_spy)
    monkeypatch.setattr(Phonopy, "produce_force_constants", produce_spy)
    monkeypatch.setattr(Phonopy, "run_mesh", mesh_spy)
    run_workflow(simple_crystal(), ZeroCalculator(), fast_config(), tmp_path / "run")
    assert set(calls["init"][0][1]) == {"supercell_matrix"}
    assert calls["generate"][0] == ((), {})
    assert calls["produce"][0] == ((), {})
    assert calls["mesh"][0][0] == (4.0,)
    assert calls["mesh"][0][1] == {"with_eigenvectors": True}


def test_changed_config_refuses_resume(tmp_path) -> None:
    output = tmp_path / "run"
    run_workflow(simple_crystal(), ZeroCalculator(), fast_config(), output)
    changed = fast_config(phonopy={"mesh_length": 5.0})
    with pytest.raises(ResumeMismatchError):
        run_workflow(simple_crystal(), ZeroCalculator(), changed, output)


def unstable_mesh() -> MeshData:
    return MeshData(
        qpoints=np.zeros((1, 3)),
        weights=np.ones(1, dtype=int),
        frequencies=np.asarray([[-1.0, 1.0, 2.0]]),
        eigenvectors=np.eye(3, dtype=complex)[None, :, :],
        mesh_numbers=np.ones(3, dtype=int),
        mesh_length=1.0,
    )


def unstable_mesh_with_groups(count: int) -> MeshData:
    frequencies = np.asarray([[-1.0 + 0.1 * index for index in range(count)]])
    return MeshData(
        qpoints=np.zeros((1, 3)),
        weights=np.ones(1, dtype=int),
        frequencies=frequencies,
        eigenvectors=np.eye(count, dtype=complex)[None, :, :],
        mesh_numbers=np.ones(3, dtype=int),
        mesh_length=1.0,
    )


def fake_candidate(tmp_path: Path, group_rank: int) -> DistortionCandidate:
    return DistortionCandidate(
        candidate_id=f"g{group_rank:03d}_d0000_plus",
        group_rank=group_rank,
        band_indices=(group_rank - 1,),
        coefficients=(1,),
        frequencies_thz=(-1.0 + 0.1 * (group_rank - 1),),
        sign=1,
        phase_degrees=0.0,
        target_mean_displacement_angstrom=0.1,
        displacement_statistics=DisplacementStatistics(0.1, 0.1, 0.1),
        structure_path=tmp_path / f"unused-rank-{group_rank}.extxyz",
    )


def fake_reduction_artifacts(output: Path) -> CandidateReductionArtifactPaths:
    return CandidateReductionArtifactPaths(
        output_dir=output,
        manifest=output / "manifest.json",
        resolved_config=output / "config.yaml",
        fingerprint=output / "fingerprint.json",
        candidates_dir=output / "items",
        unique_dir=output / "unique",
        deduplication=output / "deduplication.json",
        summary=output / "summary.json",
    )


def fake_metadata() -> dict:
    return {
        "supercell_matrix": np.eye(3, dtype=int).tolist(),
        "unitcell_face_heights_angstrom": [4.0, 4.0, 4.0],
        "supercell_spans_angstrom": [4.0, 4.0, 4.0],
        "supercell_atom_count": 1,
    }


def test_unstable_last_evaluation_stops_before_candidate_generation(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        workflow,
        "_run_phonopy_evaluation",
        lambda *args, **kwargs: (object(), unstable_mesh(), fake_metadata()),
    )
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(workflow, "generate_soft_mode_candidates", unexpected)
    result = run_workflow(
        simple_crystal(),
        ZeroCalculator(),
        fast_config(search={"max_evaluations": 1}),
        tmp_path / "run",
    )
    assert result.status == "max_evaluations"
    assert not called
    assert not result.artifacts.final_dir.exists()


def test_loop_input_equivalent_first_group_falls_through_to_second(
    monkeypatch, tmp_path
) -> None:
    phonopy_calls = 0

    def fake_phonopy(*args, **kwargs):
        nonlocal phonopy_calls
        phonopy_calls += 1
        return object(), unstable_mesh_with_groups(3), fake_metadata()

    generated_ranks: list[int] = []

    def fake_generate(phonon, mesh, config, output, **kwargs):
        rank = int(kwargs["selected_group_rank"])
        generated_ranks.append(rank)
        candidate = fake_candidate(tmp_path, rank)
        groups = workflow.rank_soft_modes(mesh, config)
        return SoftModeResult(
            soft_mode_groups=groups,
            selected_mode_groups=(groups[rank - 1],),
            supercells=(),
            candidates=(candidate,),
            output_dir=output,
            report_path=output / "soft_modes.json",
        )

    def fake_reduce(candidates, calculator, config, output, **kwargs):
        candidate = candidates[0]
        primitive = simple_crystal()
        result = CandidateResult(
            index=0,
            candidate_id=candidate.candidate_id,
            source=str(candidate.structure_path),
            status="success",
            relaxed_atoms=primitive.copy(),
            primitive_atoms=primitive,
            energy_eV=float(candidate.group_rank),
            energy_per_atom_eV=float(candidate.group_rank),
            max_force_eV_per_A=0.0,
        )
        if candidate.group_rank == 1:
            result.exclusion_reason = "equivalent_to_loop_input"
            return CandidateReductionResult(
                status="complete",
                candidates=[result],
                duplicate_groups=[],
                unique_structures=[],
                artifacts=fake_reduction_artifacts(output),
                excluded_groups=[
                    ExcludedDuplicateGroup(
                        index=0,
                        representative_index=0,
                        member_indices=(0,),
                        candidate_ids=(candidate.candidate_id,),
                        reason="equivalent_to_loop_input",
                        matched_iteration_index=0,
                        reference_structure=Path(kwargs["loop_input_structure"]),
                    )
                ],
            )
        primitive.set_cell([5.0, 5.0, 5.0], scale_atoms=True)
        result.primitive_atoms = primitive
        result.relaxed_atoms = primitive.copy()
        result.duplicate_group = 0
        result.is_representative = True
        return CandidateReductionResult(
            status="complete",
            candidates=[result],
            duplicate_groups=[
                DuplicateGroup(0, 0, (0,), output / "unique" / "unique_0000.extxyz")
            ],
            unique_structures=[primitive],
            artifacts=fake_reduction_artifacts(output),
        )

    monkeypatch.setattr(workflow, "_run_phonopy_evaluation", fake_phonopy)
    monkeypatch.setattr(workflow, "generate_soft_mode_candidates", fake_generate)
    monkeypatch.setattr(workflow, "reduce_candidates", fake_reduce)
    events: list[str] = []
    result = run_workflow(
        simple_crystal(),
        ZeroCalculator(),
        fast_config(search={"max_evaluations": 2}),
        tmp_path / "run",
        progress=events.append,
    )

    assert result.status == "max_evaluations"
    assert phonopy_calls == 2
    assert generated_ranks == [1, 2]
    first = json.loads(result.artifacts.history.read_text())["iterations"][0]
    assert first["attempted_mode_group_ranks"] == [1, 2]
    assert first["selected_mode_group_rank"] == 2
    assert [item["status"] for item in first["mode_group_attempts"]] == [
        "history_equivalent",
        "novel_candidates_found",
    ]
    assert any("advancing to the next ranked group" in item for item in events)
    iteration = tmp_path / "run" / "iterations" / "0000"
    assert (iteration / "instabilities" / "soft_modes.json").exists()
    assert (iteration / "instabilities" / "preflight.json").exists()


def test_all_failed_group_falls_through_and_all_failed_search_raises(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        workflow,
        "_run_phonopy_evaluation",
        lambda *args, **kwargs: (
            object(),
            unstable_mesh_with_groups(2),
            fake_metadata(),
        ),
    )

    def fake_generate(phonon, mesh, config, output, **kwargs):
        rank = int(kwargs["selected_group_rank"])
        candidate = fake_candidate(tmp_path, rank)
        groups = workflow.rank_soft_modes(mesh, config)
        return SoftModeResult(
            soft_mode_groups=groups,
            selected_mode_groups=(groups[rank - 1],),
            supercells=(),
            candidates=(candidate,),
            output_dir=output,
            report_path=output / "soft_modes.json",
        )

    attempted: list[int] = []

    def fake_reduce(candidates, calculator, config, output, **kwargs):
        candidate = candidates[0]
        attempted.append(candidate.group_rank)
        failed = CandidateResult(
            index=0,
            candidate_id=candidate.candidate_id,
            source=str(candidate.structure_path),
            status="failed",
            error={"type": "RuntimeError", "message": "failed"},
        )
        return CandidateReductionResult(
            status="partial",
            candidates=[failed],
            duplicate_groups=[],
            unique_structures=[],
            artifacts=fake_reduction_artifacts(output),
        )

    monkeypatch.setattr(workflow, "generate_soft_mode_candidates", fake_generate)
    monkeypatch.setattr(workflow, "reduce_candidates", fake_reduce)
    with pytest.raises(CandidateReductionError, match="ranks 1, 2"):
        run_workflow(
            simple_crystal(),
            ZeroCalculator(),
            fast_config(
                soft_modes={"max_mode_groups": 2},
                search={"max_evaluations": 2},
            ),
            tmp_path / "run",
        )
    assert attempted == [1, 2]


def test_five_history_equivalent_groups_are_exhausted_in_rank_order(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        workflow,
        "_run_phonopy_evaluation",
        lambda *args, **kwargs: (
            object(),
            unstable_mesh_with_groups(6),
            fake_metadata(),
        ),
    )

    def fake_generate(phonon, mesh, config, output, **kwargs):
        rank = int(kwargs["selected_group_rank"])
        candidate = fake_candidate(tmp_path, rank)
        groups = workflow.rank_soft_modes(mesh, config)
        return SoftModeResult(
            soft_mode_groups=groups,
            selected_mode_groups=(groups[rank - 1],),
            supercells=(),
            candidates=(candidate,),
            output_dir=output,
            report_path=output / "soft_modes.json",
        )

    def fake_reduce(candidates, calculator, config, output, **kwargs):
        candidate = candidates[0]
        primitive = simple_crystal()
        result = CandidateResult(
            index=0,
            candidate_id=candidate.candidate_id,
            source=str(candidate.structure_path),
            status="success",
            relaxed_atoms=primitive.copy(),
            primitive_atoms=primitive,
            energy_eV=0.0,
            energy_per_atom_eV=0.0,
            max_force_eV_per_A=0.0,
            exclusion_reason="equivalent_to_loop_input",
        )
        return CandidateReductionResult(
            status="complete",
            candidates=[result],
            duplicate_groups=[],
            unique_structures=[],
            artifacts=fake_reduction_artifacts(output),
            excluded_groups=[
                ExcludedDuplicateGroup(
                    index=0,
                    representative_index=0,
                    member_indices=(0,),
                    candidate_ids=(candidate.candidate_id,),
                    reason="equivalent_to_loop_input",
                    matched_iteration_index=0,
                    reference_structure=Path(kwargs["loop_input_structure"]),
                )
            ],
        )

    monkeypatch.setattr(workflow, "generate_soft_mode_candidates", fake_generate)
    monkeypatch.setattr(workflow, "reduce_candidates", fake_reduce)
    result = run_workflow(
        simple_crystal(),
        ZeroCalculator(),
        fast_config(search={"max_evaluations": 2}),
        tmp_path / "run",
    )

    assert result.status == "cycle_detected"
    history = json.loads(result.artifacts.history.read_text())["iterations"][0]
    assert history["attempted_mode_group_ranks"] == [1, 2, 3, 4, 5]
    assert history["selected_mode_group_rank"] is None
    assert history["number_of_generated_candidates"] == 5
    assert all(
        item["status"] == "history_equivalent"
        for item in history["mode_group_attempts"]
    )
    selection = json.loads(
        (tmp_path / "run" / "iterations" / "0000" / "selection.json").read_text()
    )
    assert selection["ranking"] == []
    assert selection["termination_reason"] == "all_attempted_mode_groups_exhausted"


def test_all_loop_input_equivalent_candidates_terminate_as_cycle(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        workflow,
        "_run_phonopy_evaluation",
        lambda *args, **kwargs: (object(), unstable_mesh(), fake_metadata()),
    )
    generated = DistortionCandidate(
        candidate_id="g001_d0000_plus",
        group_rank=1,
        band_indices=(0,),
        coefficients=(1,),
        frequencies_thz=(-1.0,),
        sign=1,
        phase_degrees=0.0,
        target_mean_displacement_angstrom=0.1,
        displacement_statistics=DisplacementStatistics(0.1, 0.1, 0.1),
        structure_path=tmp_path / "unused.extxyz",
    )
    failed_generated = DistortionCandidate(
        candidate_id="g001_d0000_minus",
        group_rank=1,
        band_indices=(0,),
        coefficients=(1,),
        frequencies_thz=(-1.0,),
        sign=-1,
        phase_degrees=0.0,
        target_mean_displacement_angstrom=0.1,
        displacement_statistics=DisplacementStatistics(0.1, 0.1, 0.1),
        structure_path=tmp_path / "unused-failed.extxyz",
    )

    def fake_generate(phonon, mesh, config, output, **kwargs):
        groups = workflow.rank_soft_modes(mesh, config)
        return SoftModeResult(
            soft_mode_groups=groups,
            selected_mode_groups=groups,
            supercells=(),
            candidates=(generated, failed_generated),
            output_dir=output,
            report_path=output / "soft_modes.json",
        )

    def fake_reduce(candidates, calculator, config, output, **kwargs):
        primitive = simple_crystal()
        item = CandidateResult(
            index=0,
            candidate_id=generated.candidate_id,
            source=str(generated.structure_path),
            status="success",
            relaxed_atoms=primitive.copy(),
            primitive_atoms=primitive,
            energy_eV=1.0,
            energy_per_atom_eV=1.0,
            max_force_eV_per_A=0.0,
            exclusion_reason="equivalent_to_loop_input",
        )
        artifacts = CandidateReductionArtifactPaths(
            output_dir=output,
            manifest=output / "manifest.json",
            resolved_config=output / "config.yaml",
            fingerprint=output / "fingerprint.json",
            candidates_dir=output / "items",
            unique_dir=output / "unique",
            deduplication=output / "deduplication.json",
            summary=output / "summary.json",
        )
        failed = CandidateResult(
            index=1,
            candidate_id=failed_generated.candidate_id,
            source=str(failed_generated.structure_path),
            status="failed",
            error={"type": "RuntimeError", "message": "calculator failed"},
        )
        return CandidateReductionResult(
            status="partial",
            candidates=[item, failed],
            duplicate_groups=[],
            unique_structures=[],
            artifacts=artifacts,
            excluded_groups=[
                ExcludedDuplicateGroup(
                    index=0,
                    representative_index=0,
                    member_indices=(0,),
                    candidate_ids=(generated.candidate_id,),
                    reason="equivalent_to_loop_input",
                )
            ],
        )

    monkeypatch.setattr(workflow, "generate_soft_mode_candidates", fake_generate)
    monkeypatch.setattr(workflow, "reduce_candidates", fake_reduce)
    result = run_workflow(
        simple_crystal(), ZeroCalculator(), fast_config(), tmp_path / "run"
    )
    assert result.status == "cycle_detected"
    assert result.iterations[0].selected_candidate_id is None
    assert result.iterations[0].selected_structure is None
    assert result.iterations[0].energy_change_per_atom_eV is None
    history = json.loads(result.artifacts.history.read_text())["iterations"][0]
    assert history["termination_reason"] == "all_attempted_mode_groups_exhausted"
    assert history["number_of_excluded_loop_input_structures"] == 1
    assert history["number_of_failed_candidates"] == 1
    selection = json.loads(
        (tmp_path / "run" / "iterations" / "0000" / "selection.json").read_text()
    )
    assert selection["status"] == "cycle_detected"
    assert selection["ranking"] == []
    assert selection["selected_structure"] is None
    summary = json.loads(result.artifacts.summary.read_text())
    assert summary["termination_reason"] == "all_attempted_mode_groups_exhausted"
    assert not result.artifacts.final_dir.exists()


def test_loop_input_match_is_removed_before_ranking_next_candidate(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        workflow,
        "_run_phonopy_evaluation",
        lambda *args, **kwargs: (object(), unstable_mesh(), fake_metadata()),
    )

    def generated(candidate_id: str, index: int) -> DistortionCandidate:
        return DistortionCandidate(
            candidate_id=candidate_id,
            group_rank=1,
            band_indices=(0,),
            coefficients=(1,),
            frequencies_thz=(-1.0,),
            sign=1 if index == 0 else -1,
            phase_degrees=0.0,
            target_mean_displacement_angstrom=0.1,
            displacement_statistics=DisplacementStatistics(0.1, 0.1, 0.1),
            structure_path=tmp_path / f"unused-{index}.extxyz",
        )

    matching = generated("matching-lower-energy", 0)
    novel = generated("novel-higher-energy", 1)

    def fake_generate(phonon, mesh, config, output, **kwargs):
        groups = workflow.rank_soft_modes(mesh, config)
        return SoftModeResult(
            soft_mode_groups=groups,
            selected_mode_groups=groups,
            supercells=(),
            candidates=(matching, novel),
            output_dir=output,
            report_path=output / "soft_modes.json",
        )

    def fake_reduce(candidates, calculator, config, output, **kwargs):
        assert Path(kwargs["loop_input_structure"]).name == "accepted_primitive.extxyz"
        loop_input = simple_crystal()
        novel_primitive = simple_crystal()
        novel_primitive.set_cell([5.0, 5.0, 5.0], scale_atoms=True)
        results = [
            CandidateResult(
                index=0,
                candidate_id=matching.candidate_id,
                source=str(matching.structure_path),
                status="success",
                relaxed_atoms=loop_input.copy(),
                primitive_atoms=loop_input,
                energy_eV=0.0,
                energy_per_atom_eV=0.0,
                max_force_eV_per_A=0.0,
                exclusion_reason="equivalent_to_loop_input",
            ),
            CandidateResult(
                index=1,
                candidate_id=novel.candidate_id,
                source=str(novel.structure_path),
                status="success",
                relaxed_atoms=novel_primitive.copy(),
                primitive_atoms=novel_primitive,
                energy_eV=1.0,
                energy_per_atom_eV=1.0,
                max_force_eV_per_A=0.0,
                duplicate_group=0,
                is_representative=True,
            ),
        ]
        artifacts = CandidateReductionArtifactPaths(
            output_dir=output,
            manifest=output / "manifest.json",
            resolved_config=output / "config.yaml",
            fingerprint=output / "fingerprint.json",
            candidates_dir=output / "items",
            unique_dir=output / "unique",
            deduplication=output / "deduplication.json",
            summary=output / "summary.json",
        )
        return CandidateReductionResult(
            status="complete",
            candidates=results,
            duplicate_groups=[
                DuplicateGroup(
                    0,
                    1,
                    (1,),
                    output / "unique" / "unique_0000.extxyz",
                )
            ],
            unique_structures=[novel_primitive],
            artifacts=artifacts,
            excluded_groups=[
                ExcludedDuplicateGroup(
                    index=0,
                    representative_index=0,
                    member_indices=(0,),
                    candidate_ids=(matching.candidate_id,),
                    reason="equivalent_to_loop_input",
                )
            ],
        )

    monkeypatch.setattr(workflow, "generate_soft_mode_candidates", fake_generate)
    monkeypatch.setattr(workflow, "reduce_candidates", fake_reduce)
    result = run_workflow(
        simple_crystal(),
        ZeroCalculator(),
        fast_config(search={"max_evaluations": 2}),
        tmp_path / "run",
    )

    assert result.status == "max_evaluations"
    assert result.iterations[0].selected_candidate_id == novel.candidate_id
    first_history = json.loads(result.artifacts.history.read_text())["iterations"][0]
    assert first_history["number_of_excluded_loop_input_structures"] == 1
    assert first_history["excluded_candidate_ids"] == [matching.candidate_id]
    selection = json.loads(
        (tmp_path / "run" / "iterations" / "0000" / "selection.json").read_text()
    )
    assert [item["candidate_id"] for item in selection["ranking"]] == [
        novel.candidate_id
    ]


def test_previous_iteration_match_falls_through_to_next_group(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        workflow,
        "_run_phonopy_evaluation",
        lambda *args, **kwargs: (
            object(),
            unstable_mesh_with_groups(2),
            fake_metadata(),
        ),
    )

    def fake_generate(phonon, mesh, config, output, **kwargs):
        rank = int(kwargs["selected_group_rank"])
        candidate = fake_candidate(tmp_path, rank)
        groups = workflow.rank_soft_modes(mesh, config)
        return SoftModeResult(
            soft_mode_groups=groups,
            selected_mode_groups=(groups[rank - 1],),
            supercells=(),
            candidates=(candidate,),
            output_dir=output,
            report_path=output / "soft_modes.json",
        )

    def fake_reduce(candidates, calculator, config, output, **kwargs):
        iteration_index = int(kwargs["iteration_index"])
        candidate = candidates[0]
        primitive = simple_crystal()
        result = CandidateResult(
            index=0,
            candidate_id=f"candidate-{iteration_index}-{candidate.group_rank}",
            source=str(candidate.structure_path),
            status="success",
            relaxed_atoms=primitive.copy(),
            primitive_atoms=primitive,
            energy_eV=float(iteration_index + candidate.group_rank),
            energy_per_atom_eV=float(iteration_index + candidate.group_rank),
            max_force_eV_per_A=0.0,
        )
        if iteration_index == 1 and candidate.group_rank == 1:
            previous = kwargs["previous_accepted_structures"]
            assert [item[0] for item in previous] == [0]
            result.exclusion_reason = "equivalent_to_previous_iteration"
            return CandidateReductionResult(
                status="complete",
                candidates=[result],
                duplicate_groups=[],
                unique_structures=[],
                artifacts=fake_reduction_artifacts(output),
                excluded_groups=[
                    ExcludedDuplicateGroup(
                        index=0,
                        representative_index=0,
                        member_indices=(0,),
                        candidate_ids=(candidate.candidate_id,),
                        reason="equivalent_to_previous_iteration",
                        matched_iteration_index=0,
                        reference_structure=Path(previous[0][1]),
                    )
                ],
            )
        size = 5.0 if iteration_index == 0 else 6.0
        primitive.set_cell([size, size, size], scale_atoms=True)
        result.primitive_atoms = primitive
        result.relaxed_atoms = primitive.copy()
        result.duplicate_group = 0
        result.is_representative = True
        return CandidateReductionResult(
            status="complete",
            candidates=[result],
            duplicate_groups=[
                DuplicateGroup(0, 0, (0,), output / "unique" / "unique_0000.extxyz")
            ],
            unique_structures=[primitive],
            artifacts=fake_reduction_artifacts(output),
        )

    monkeypatch.setattr(workflow, "generate_soft_mode_candidates", fake_generate)
    monkeypatch.setattr(workflow, "reduce_candidates", fake_reduce)
    result = run_workflow(
        simple_crystal(),
        ZeroCalculator(),
        fast_config(search={"max_evaluations": 3}),
        tmp_path / "run",
    )

    assert result.status == "max_evaluations"
    history = json.loads(result.artifacts.history.read_text())["iterations"]
    assert history[1]["attempted_mode_group_ranks"] == [1, 2]
    assert history[1]["selected_mode_group_rank"] == 2
    excluded = history[1]["mode_group_attempts"][0]["excluded_groups"][0]
    assert excluded["reason"] == "equivalent_to_previous_iteration"
    assert excluded["matched_iteration_index"] == 0


def test_cli_has_only_unified_run_and_exports_stable_result(tmp_path, capsys) -> None:
    structure = tmp_path / "input.extxyz"
    config = tmp_path / "config.yaml"
    output = tmp_path / "output"
    write(structure, simple_crystal())
    config.write_text(
        """
calculator:
  factory: tests.helpers:make_zero_calculator
relaxation:
  force_tolerance: 100
  max_steps: 1
phonopy:
  minimum_supercell_span_angstrom: 4
  mesh_length: 4
""".strip()
        + "\n",
        encoding="utf-8",
    )
    assert main(["run", str(structure), "--config", str(config), "--output", str(output)]) == 0
    transcript = capsys.readouterr().out
    assert "workflow stable" in transcript
    assert "MORI>" not in transcript
    assert "\x1b[" not in transcript
    assert (output / "final" / "structure.extxyz").exists()
    choices = _build_parser()._subparsers._group_actions[0].choices
    assert set(choices) == {"run"}
