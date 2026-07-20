from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT


class ZeroCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(self.atoms), 3), dtype=float),
        }


class ZeroStressCalculator(ZeroCalculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy", "forces", "stress"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["stress"] = np.zeros(6, dtype=float)


class ConstantForceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(self.atoms.positions, dtype=float)
        self.results = {
            "energy": -float(positions[:, 0].sum()),
            "forces": np.tile([1.0, 0.0, 0.0], (len(self.atoms), 1)),
        }


class NonFiniteCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.full((len(self.atoms), 3), np.nan),
        }


class MalformedForceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(self.atoms), 2), dtype=float),
        }


class RaisingCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        raise RuntimeError("intentional calculator failure")


class OffsetEnergyCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, energy: float):
        super().__init__()
        self.energy = energy

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(self.energy),
            "forces": np.zeros((len(self.atoms), 3), dtype=float),
        }


def make_zero_calculator(*, context, **kwargs):
    return ZeroCalculator(**kwargs)


def make_emt_calculator(*, context, **kwargs):
    return EMT(**kwargs)


def make_raising_calculator(*, context, **kwargs):
    return RaisingCalculator(**kwargs)
