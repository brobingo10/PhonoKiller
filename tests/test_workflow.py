from __future__ import annotations

import json
import shutil

from ase import Atoms
from ase.io import write
import numpy as np
import pytest
from phonopy import Phonopy

from phonokiller import (
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


def test_uphill_winner_that_matches_history_terminates_as_cycle(
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

    def fake_generate(phonon, mesh, config, output, **kwargs):
        groups = workflow.rank_soft_modes(mesh, config)
        return SoftModeResult(
            soft_mode_groups=groups,
            selected_mode_groups=groups,
            supercells=(),
            candidates=(generated,),
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
            duplicate_group=0,
            is_representative=True,
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
        return CandidateReductionResult(
            status="complete",
            candidates=[item],
            duplicate_groups=[
                DuplicateGroup(0, 0, (0,), output / "unique" / "unique_0000.extxyz")
            ],
            unique_structures=[primitive],
            artifacts=artifacts,
        )

    monkeypatch.setattr(workflow, "generate_soft_mode_candidates", fake_generate)
    monkeypatch.setattr(workflow, "reduce_candidates", fake_reduce)
    result = run_workflow(
        simple_crystal(), ZeroCalculator(), fast_config(), tmp_path / "run"
    )
    assert result.status == "cycle_detected"
    assert result.iterations[0].energy_change_per_atom_eV == pytest.approx(1.0)
    assert not result.artifacts.final_dir.exists()


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
