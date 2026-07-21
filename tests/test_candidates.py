from __future__ import annotations

import json
from types import SimpleNamespace

from ase import Atoms
from ase.build import bulk
from ase.io import write
import numpy as np
import pytest

from phonokiller import RunConfig
from phonokiller.candidates import (
    make_nonideal_primitive,
    reduce_candidates,
    structures_equivalent,
)
from phonokiller.exceptions import CandidateReductionError
from phonokiller.models import DisplacementStatistics, DistortionCandidate
from phonokiller.relaxation import _calculator_device_info
from tests.helpers import OffsetEnergyCalculator, RaisingCalculator, ZeroCalculator


def config(**updates) -> RunConfig:
    payload = {
        "relaxation": {"force_tolerance": 100.0, "max_steps": 2},
        "symmetry": {"symprec": 0.05},
    }
    for section, values in updates.items():
        payload.setdefault(section, {}).update(values)
    return RunConfig.model_validate(payload)


def candidate(path, candidate_id: str) -> DistortionCandidate:
    return DistortionCandidate(
        candidate_id=candidate_id,
        group_rank=1,
        band_indices=(0,),
        coefficients=(1,),
        frequencies_thz=(-1.0,),
        sign=1,
        phase_degrees=0.0,
        target_mean_displacement_angstrom=0.1,
        displacement_statistics=DisplacementStatistics(0.1, 0.1, 0.1),
        structure_path=path,
    )


def nonmatching_loop_input(tmp_path):
    path = tmp_path / "loop-input.extxyz"
    write(path, bulk("Cu", "fcc", a=3.6))
    return path


class EnergyFactory:
    def __init__(self, energies):
        self.energies = energies
        self.calls: list[int] = []

    def __call__(self, *, context):
        assert context.stage == "candidate_relaxation"
        assert context.iteration_index == 2
        assert context.candidate_id is not None
        self.calls.append(context.candidate_index)
        return OffsetEnergyCalculator(self.energies[context.candidate_index])


class DeviceCalculator(ZeroCalculator):
    def __init__(self, device: str):
        super().__init__()
        self.device = device


class PlacedModel:
    def __init__(self, device: str):
        self.device = device

    def parameters(self):
        yield SimpleNamespace(device=self.device)


def test_relax_reduce_and_deduplicate_to_lowest_energy(tmp_path) -> None:
    primitive = bulk("Al", "fcc", a=4.05)
    first = primitive.repeat((2, 1, 1))
    second = first.copy()
    second.rotate(37, "z", rotate_cell=True)
    second = second[[1, 0]]
    first_path, second_path = tmp_path / "first.extxyz", tmp_path / "second.extxyz"
    write(first_path, first)
    write(second_path, second)
    factory = EnergyFactory([2.0, 1.0])
    result = reduce_candidates(
        (candidate(first_path, "a"), candidate(second_path, "b")),
        factory,
        config(),
        tmp_path / "results",
        iteration_index=2,
        loop_input_structure=nonmatching_loop_input(tmp_path),
    )
    assert result.status == "complete"
    assert len(result.duplicate_groups) == 1
    assert result.duplicate_groups[0].representative_index == 1
    assert result.candidates[1].candidate_id == "b"
    assert result.candidates[1].is_representative


def test_candidate_relaxation_reports_and_records_gpu_device(tmp_path) -> None:
    structure = tmp_path / "candidate.extxyz"
    write(structure, bulk("Al", "fcc", a=4.05))
    events: list[str] = []

    result = reduce_candidates(
        (candidate(structure, "gpu-check"),),
        lambda *, context: DeviceCalculator("cuda:0"),
        config(),
        tmp_path / "results",
        iteration_index=0,
        loop_input_structure=nonmatching_loop_input(tmp_path),
        progress=events.append,
    )

    assert result.status == "complete"
    assert any(
        "first force evaluation completed on cuda:0; GPU execution confirmed" in event
        for event in events
    )
    metrics = json.loads(
        (
            result.artifacts.candidates_dir / "0000" / "relaxation" / "metrics.json"
        ).read_text(encoding="utf-8")
    )
    assert metrics["calculator_device"] == "cuda:0"
    assert metrics["gpu_active"] is True
    assert metrics["calculator_device_source"] == "calculator"


def test_model_parameter_device_takes_priority_over_reported_device() -> None:
    calculator = DeviceCalculator("cuda:0")
    calculator.models = [PlacedModel("cpu")]

    info = _calculator_device_info(calculator)

    assert info.device == "cpu"
    assert info.gpu_active is False
    assert info.source == "model_parameters"


def test_primitive_reduction_is_not_idealized(tmp_path) -> None:
    atoms = Atoms(
        "Al2",
        scaled_positions=[[0.01, 0.0, 0.0], [0.51, 0.0, 0.0]],
        cell=[[8.0, 0.0, 0.0], [0.3, 4.0, 0.0], [0.1, 0.2, 4.0]],
        pbc=True,
    )
    primitive = make_nonideal_primitive(atoms, config(symmetry={"symprec": 0.02}))
    assert len(primitive) == 1
    # no_idealize retains the non-orthogonal lattice rather than snapping it.
    assert not np.allclose(primitive.cell.array, np.diag(np.diag(primitive.cell.array)))


class FailOnceFactory:
    def __init__(self):
        self.fail = True
        self.calls: list[int] = []

    def __call__(self, *, context):
        self.calls.append(context.candidate_index)
        if context.candidate_index == 1 and self.fail:
            return RaisingCalculator()
        return ZeroCalculator()


def test_partial_batch_retries_only_failure(tmp_path) -> None:
    paths = [tmp_path / "a.extxyz", tmp_path / "b.extxyz"]
    write(paths[0], bulk("Al", "fcc", a=4.05))
    write(paths[1], bulk("Al", "fcc", a=4.10))
    values = tuple(candidate(path, label) for path, label in zip(paths, ("a", "b")))
    factory = FailOnceFactory()
    output = tmp_path / "results"
    loop_input = nonmatching_loop_input(tmp_path)
    partial = reduce_candidates(
        values,
        factory,
        config(),
        output,
        iteration_index=0,
        loop_input_structure=loop_input,
    )
    assert partial.status == "partial"
    assert [item.status for item in partial.candidates] == ["success", "failed"]
    factory.fail = False
    complete = reduce_candidates(
        values,
        factory,
        config(),
        output,
        iteration_index=0,
        loop_input_structure=loop_input,
    )
    assert complete.status == "complete"
    assert factory.calls.count(0) == 1
    assert factory.calls.count(1) == 2


def test_loop_input_equivalent_group_is_excluded_and_resume_reuses_relaxations(
    tmp_path,
) -> None:
    loop_input = tmp_path / "accepted-primitive.extxyz"
    matching = tmp_path / "matching.extxyz"
    novel = tmp_path / "novel.extxyz"
    write(loop_input, bulk("Al", "fcc", a=4.05))
    write(matching, bulk("Al", "fcc", a=4.05))
    write(novel, bulk("Al", "fcc", a=5.0))
    factory = EnergyFactory([0.0, 2.0])
    values = (candidate(matching, "matching"), candidate(novel, "novel"))
    output = tmp_path / "results"

    result = reduce_candidates(
        values,
        factory,
        config(),
        output,
        iteration_index=2,
        loop_input_structure=loop_input,
    )

    assert [group.representative_index for group in result.duplicate_groups] == [1]
    assert len(result.excluded_groups) == 1
    assert result.excluded_groups[0].candidate_ids == ("matching",)
    assert result.candidates[0].exclusion_reason == "equivalent_to_loop_input"
    assert result.candidates[0].is_representative is False
    assert result.candidates[1].is_representative is True
    assert len(list(result.artifacts.unique_dir.glob("unique_*.extxyz"))) == 1
    deduplication = json.loads(result.artifacts.deduplication.read_text())
    assert deduplication["number_of_deduplicated_structures"] == 2
    assert deduplication["number_of_excluded_loop_input_structures"] == 1
    assert deduplication["number_of_unique_structures"] == 1
    assert deduplication["excluded_groups"][0]["reason"] == (
        "equivalent_to_loop_input"
    )
    assert deduplication["loop_input_structure"] == str(loop_input.resolve())
    summary = json.loads(result.artifacts.summary.read_text())
    assert summary["number_of_excluded_loop_input_candidates"] == 1
    assert summary["excluded_candidate_ids"] == ["matching"]
    matching_result = json.loads(
        (
            result.artifacts.candidates_dir / "0000" / "result.json"
        ).read_text()
    )
    assert matching_result["exclusion_reason"] == "equivalent_to_loop_input"

    resumed = reduce_candidates(
        values,
        factory,
        config(),
        output,
        iteration_index=2,
        loop_input_structure=loop_input,
    )
    assert factory.calls == [0, 1]
    assert resumed.excluded_groups[0].candidate_ids == ("matching",)


def test_entire_duplicate_group_matching_loop_input_is_removed_but_preserved(
    tmp_path,
) -> None:
    primitive = bulk("Al", "fcc", a=4.05)
    first = primitive.repeat((2, 1, 1))
    second = first.copy()
    second.rotate(37, "z", rotate_cell=True)
    second = second[[1, 0]]
    loop_input = tmp_path / "accepted-primitive.extxyz"
    first_path = tmp_path / "first.extxyz"
    second_path = tmp_path / "second.extxyz"
    write(loop_input, primitive)
    write(first_path, first)
    write(second_path, second)

    result = reduce_candidates(
        (candidate(first_path, "first"), candidate(second_path, "second")),
        EnergyFactory([2.0, 1.0]),
        config(),
        tmp_path / "results",
        iteration_index=2,
        loop_input_structure=loop_input,
    )

    assert result.duplicate_groups == []
    assert result.unique_structures == []
    assert len(result.excluded_groups) == 1
    assert result.excluded_groups[0].representative_index == 1
    assert result.excluded_groups[0].member_indices == (0, 1)
    assert result.excluded_groups[0].candidate_ids == ("first", "second")
    assert all(
        item.exclusion_reason == "equivalent_to_loop_input"
        for item in result.candidates
    )
    assert not list(result.artifacts.unique_dir.glob("unique_*.extxyz"))
    assert (result.artifacts.candidates_dir / "0000" / "primitive.extxyz").exists()
    assert (result.artifacts.candidates_dir / "0001" / "primitive.extxyz").exists()


def test_previous_accepted_structure_is_excluded_with_iteration_provenance(
    tmp_path,
) -> None:
    previous = tmp_path / "iteration-0001-accepted.extxyz"
    matching = tmp_path / "matching-previous.extxyz"
    write(previous, bulk("Al", "fcc", a=4.05))
    rotated = bulk("Al", "fcc", a=4.05)
    rotated.rotate(29, "z", rotate_cell=True)
    write(matching, rotated)

    result = reduce_candidates(
        (candidate(matching, "previous-match"),),
        EnergyFactory([0.0]),
        config(),
        tmp_path / "results",
        iteration_index=2,
        loop_input_structure=nonmatching_loop_input(tmp_path),
        previous_accepted_structures=((1, previous),),
    )

    assert result.duplicate_groups == []
    assert len(result.excluded_groups) == 1
    excluded = result.excluded_groups[0]
    assert excluded.reason == "equivalent_to_previous_iteration"
    assert excluded.matched_iteration_index == 1
    assert excluded.reference_structure == previous.resolve()
    assert result.candidates[0].exclusion_iteration_index == 1
    payload = json.loads(result.artifacts.deduplication.read_text())
    assert payload["number_of_excluded_previous_iteration_structures"] == 1
    assert payload["excluded_groups"][0]["matched_iteration_index"] == 1
    assert payload["excluded_groups"][0]["reference_structure"] == str(
        previous.resolve()
    )


def test_loop_input_equivalence_uses_configured_structural_tolerances() -> None:
    left = Atoms(
        ["Al", "Cu"],
        scaled_positions=[[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]],
        cell=[[4.0, 0.0, 0.0], [0.3, 4.2, 0.0], [0.1, 0.2, 4.4]],
        pbc=True,
    )
    right = left[[1, 0]]
    right.rotate(31, "z", rotate_cell=True)
    right.positions += 1.0e-4

    assert structures_equivalent(left, right, config())


def test_candidate_atom_cap_is_checked_before_output_or_calculator(tmp_path) -> None:
    structure = tmp_path / "oversized.extxyz"
    atoms = Atoms(
        numbers=np.ones(3501, dtype=int),
        positions=np.column_stack(
            [np.arange(3501, dtype=float), np.zeros(3501), np.zeros(3501)]
        ),
        cell=[3502.0, 2.0, 2.0],
        pbc=True,
    )
    write(structure, atoms)
    called = False

    def factory(*, context):
        nonlocal called
        called = True
        return ZeroCalculator()

    output = tmp_path / "results"
    with pytest.raises(CandidateReductionError, match="requires 3501 atoms"):
        reduce_candidates(
            (candidate(structure, "too-large"),),
            factory,
            config(),
            output,
            iteration_index=0,
            loop_input_structure=nonmatching_loop_input(tmp_path),
        )
    assert not called
    assert not output.exists()
