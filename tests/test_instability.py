from __future__ import annotations

from fractions import Fraction
import json

from ase.io import read
import numpy as np
import pytest
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonokiller import CandidateLimitError, SoftModeConfig
import phonokiller.instability as instability
from phonokiller.instability import (
    _integer_determinant,
    _is_exactly_commensurate,
    _minimum_commensurate_supercell,
    _rational_qpoint,
    candidate_count,
    generate_soft_mode_candidates,
    rank_soft_modes,
    ternary_directions,
)
from phonokiller.models import MeshData


def make_mesh(frequencies, *, qpoints=None, mesh_numbers=(2, 2, 2)) -> MeshData:
    frequency_array = np.asarray(frequencies, dtype=float)
    nqpoint, nband = frequency_array.shape
    if qpoints is None:
        qpoints = np.zeros((nqpoint, 3), dtype=float)
    return MeshData(
        qpoints=np.asarray(qpoints, dtype=float),
        weights=np.ones(nqpoint, dtype=int),
        frequencies=frequency_array,
        eigenvectors=np.repeat(
            np.eye(nband, dtype=complex)[None, :, :], nqpoint, axis=0
        ),
        mesh_numbers=np.asarray(mesh_numbers, dtype=int),
        mesh_length=100.0,
    )


def make_unstable_phonon() -> tuple[Phonopy, MeshData]:
    unitcell = PhonopyAtoms(
        symbols=["Al"],
        cell=np.eye(3) * 4.0,
        scaled_positions=[[0.5, 0.5, 0.5]],
        masses=[26.9815385],
    )
    phonon = Phonopy(unitcell, supercell_matrix=[1, 1, 1])
    force_constants = np.zeros((1, 1, 3, 3), dtype=float)
    force_constants[0, 0] = np.diag([-1.0, -1.0, -1.0])
    phonon.force_constants = force_constants
    result = phonon.run_mesh(1.0, with_eigenvectors=True)
    return phonon, MeshData(
        qpoints=np.asarray(result.qpoints, dtype=float),
        weights=np.asarray(result.weights, dtype=int),
        frequencies=np.asarray(result.frequencies, dtype=float),
        eigenvectors=np.asarray(result.eigenvectors, dtype=complex),
        mesh_numbers=np.asarray(result.mesh_numbers, dtype=int),
        mesh_length=1.0,
    )


def test_rank_soft_modes_groups_degeneracy_and_uses_strict_threshold() -> None:
    mesh = make_mesh(
        [[-0.2, -0.1995, 0.1], [-0.3, 0.2, 0.3], [-0.05, 0.4, 0.5]],
        qpoints=[[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]],
    )
    groups = rank_soft_modes(mesh, SoftModeConfig())
    assert len(groups) == 2
    assert groups[0].qpoint == (-0.5, 0.0, 0.0)
    assert groups[1].band_indices == (0, 1)


@pytest.mark.parametrize(
    ("degeneracy", "directions", "signed_candidates"),
    [(1, 1, 2), (2, 4, 8), (3, 13, 26)],
)
def test_ternary_directions_are_unique_modulo_sign(
    degeneracy, directions, signed_candidates
) -> None:
    values = ternary_directions(degeneracy)
    assert len(values) == directions
    assert len({value for value in values}) == directions
    assert all(next(item for item in value if item) > 0 for value in values)
    groups = rank_soft_modes(
        make_mesh([[-1.0] * degeneracy]), SoftModeConfig()
    )
    assert candidate_count(groups) == signed_candidates


def test_candidate_count_does_not_enumerate_directions(monkeypatch) -> None:
    groups = rank_soft_modes(make_mesh([[-1.0] * 12]), SoftModeConfig())

    def unexpected(_: int):
        raise AssertionError("candidate counting must not enumerate directions")

    monkeypatch.setattr(instability, "ternary_directions", unexpected)
    assert candidate_count(groups) == 3**12 - 1


@pytest.mark.parametrize(
    ("qpoint", "determinant"),
    [
        ((Fraction(0), Fraction(0), Fraction(0)), 1),
        ((Fraction(1, 2), Fraction(0), Fraction(0)), 2),
        ((Fraction(1, 3), Fraction(1, 3), Fraction(0)), 3),
        ((Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)), 2),
        ((Fraction(-1, 6), Fraction(1, 4), Fraction(1, 5)), 60),
    ],
)
def test_minimum_commensurate_supercell(qpoint, determinant) -> None:
    matrix = _minimum_commensurate_supercell(qpoint, np.diag([3.0, 4.0, 5.0]))
    assert _integer_determinant(matrix) == determinant
    assert _is_exactly_commensurate(matrix, qpoint)


def test_generalized_grid_fraction_reconstruction_uses_returned_d_diag() -> None:
    fractions = _rational_qpoint(
        (1 / 6, -1 / 4, 0.0), np.asarray([3, 4, 2]), 1.0e-8
    )
    assert fractions == (Fraction(1, 6), Fraction(-1, 4), Fraction(0))


def test_generate_exhaustive_candidates_with_exact_mean(tmp_path) -> None:
    phonon, mesh = make_unstable_phonon()
    result = generate_soft_mode_candidates(
        phonon,
        mesh,
        SoftModeConfig(),
        tmp_path / "instabilities",
        max_candidates=256,
    )
    assert len(result.candidates) == 26
    reference = read(result.supercells[0].reference_structure)
    plus = read(result.candidates[0].structure_path)
    minus = read(result.candidates[1].structure_path)
    np.testing.assert_allclose(
        plus.positions - reference.positions,
        -(minus.positions - reference.positions),
        atol=1.0e-12,
    )
    for candidate in result.candidates:
        assert candidate.displacement_statistics.mean_angstrom == pytest.approx(0.1)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["counts"]["generated_candidates"] == 26

    resumed = generate_soft_mode_candidates(
        phonon,
        mesh,
        SoftModeConfig(),
        result.output_dir,
        max_candidates=256,
    )
    assert [item.candidate_id for item in resumed.candidates] == [
        item.candidate_id for item in result.candidates
    ]


def test_degenerate_modes_are_batched_into_one_modulation(monkeypatch, tmp_path) -> None:
    phonon, mesh = make_unstable_phonon()
    original = phonon.run_modulations
    calls: list[list] = []

    def modulation_spy(*args, **kwargs):
        calls.append(kwargs["phonon_modes"])
        return original(*args, **kwargs)

    monkeypatch.setattr(phonon, "run_modulations", modulation_spy)
    generate_soft_mode_candidates(
        phonon,
        mesh,
        SoftModeConfig(),
        tmp_path / "batched",
        max_candidates=256,
    )
    assert len(calls) == 1
    assert [mode[1] for mode in calls[0]] == [0, 1, 2]


def test_candidate_cap_fails_before_generation(tmp_path) -> None:
    phonon, mesh = make_unstable_phonon()
    output = tmp_path / "too-many"
    with pytest.raises(CandidateLimitError, match="requires 26"):
        generate_soft_mode_candidates(
            phonon,
            mesh,
            SoftModeConfig(),
            output,
            max_candidates=25,
        )
    assert not output.exists()
