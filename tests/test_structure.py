from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from phonokiller import StructureValidationError
from phonokiller.structure import ase_to_phonopy, load_structure, phonopy_to_ase


@pytest.mark.parametrize(
    "atoms",
    [
        Atoms(),
        Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=[True, True, False]),
        Atoms("Al", positions=[[0, 0, 0]], cell=np.zeros((3, 3)), pbc=True),
    ],
)
def test_invalid_structures_are_rejected(atoms) -> None:
    with pytest.raises(StructureValidationError):
        load_structure(atoms)


def test_ase_phonopy_roundtrip_preserves_crystal_data(generic_al_crystal) -> None:
    generic_al_crystal.set_masses([26.0, 27.0])
    generic_al_crystal.set_initial_magnetic_moments([1.0, -1.0])
    converted = phonopy_to_ase(ase_to_phonopy(generic_al_crystal))
    assert converted.get_chemical_symbols() == generic_al_crystal.get_chemical_symbols()
    np.testing.assert_allclose(converted.cell.array, generic_al_crystal.cell.array)
    np.testing.assert_allclose(converted.get_scaled_positions(), generic_al_crystal.get_scaled_positions())
    np.testing.assert_allclose(converted.get_masses(), [26.0, 27.0])
    np.testing.assert_allclose(converted.get_initial_magnetic_moments(), [1.0, -1.0])
