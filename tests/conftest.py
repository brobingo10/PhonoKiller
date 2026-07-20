from __future__ import annotations

import pytest
from ase import Atoms
from ase.build import bulk


@pytest.fixture
def al_crystal() -> Atoms:
    return bulk("Al", "fcc", a=4.05, cubic=True)


@pytest.fixture
def generic_al_crystal() -> Atoms:
    return Atoms(
        "Al2",
        positions=[[0.2, 0.3, 0.4], [2.3, 2.0, 1.8]],
        cell=[5.0, 5.2, 5.4],
        pbc=True,
    )
