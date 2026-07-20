from __future__ import annotations

import json
from types import SimpleNamespace

from ase import Atoms
from ase.build import bulk
from ase.io import write
import numpy as np

from phonokiller import RunConfig
from phonokiller.candidates import make_nonideal_primitive, reduce_candidates
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
    partial = reduce_candidates(values, factory, config(), output, iteration_index=0)
    assert partial.status == "partial"
    assert [item.status for item in partial.candidates] == ["success", "failed"]
    factory.fail = False
    complete = reduce_candidates(values, factory, config(), output, iteration_index=0)
    assert complete.status == "complete"
    assert factory.calls.count(0) == 1
    assert factory.calls.count(1) == 2
