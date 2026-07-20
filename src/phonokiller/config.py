"""Validated configuration models for the unified PhonoKiller workflow."""

from __future__ import annotations

from enum import Enum
import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_MACE_FACTORY = "phonokiller.calculators:make_mace_calculator"
DEFAULT_MACE_MODEL = "medium"
DEFAULT_MACE_DEVICE = "cuda"
DEFAULT_MACE_DTYPE = "float32"
DEFAULT_MACE_DISPERSION = False


class RelaxationMode(str, Enum):
    POSITIONS = "positions"
    FULL_CELL = "full_cell"
    FIXED_SHAPE = "fixed_shape"


class OptimizerName(str, Enum):
    BFGS = "BFGS"
    LBFGS = "LBFGS"
    FIRE = "FIRE"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RelaxationConfig(StrictModel):
    mode: RelaxationMode = RelaxationMode.POSITIONS
    optimizer: OptimizerName = OptimizerName.BFGS
    force_tolerance: float = Field(default=0.005, gt=0)
    max_steps: int = Field(default=500, gt=0)


class CandidateRelaxationOverrides(StrictModel):
    """Optional candidate settings layered on the base relaxation settings."""

    mode: RelaxationMode | None = None
    optimizer: OptimizerName | None = None
    force_tolerance: float | None = Field(default=None, gt=0)
    max_steps: int | None = Field(default=None, gt=0)

    def resolve(self, base: RelaxationConfig) -> RelaxationConfig:
        overrides = self.model_dump(exclude_none=True)
        return RelaxationConfig.model_validate({**base.model_dump(), **overrides})


class PhonopyConfig(StrictModel):
    """Workflow inputs for automatic Phonopy cell and mesh sizing.

    All numerical Phonopy options that have library defaults are intentionally
    omitted. PhonoKiller only supplies the supercell matrix it must construct
    and requests eigenvectors from a length-based mesh.
    """

    minimum_supercell_span_angstrom: float = Field(default=10.0, gt=0)
    mesh_length: float = Field(default=100.0, gt=0)


class SoftModeConfig(StrictModel):
    frequency_threshold_thz: float = Field(default=-0.05, lt=0)
    degeneracy_tolerance_thz: float = Field(default=1.0e-3, gt=0)
    max_mode_groups: int = Field(default=5, gt=0)
    mean_displacement_angstrom: float = Field(default=0.10, gt=0)
    phase_degrees: float = 0.0
    qpoint_tolerance: float = Field(default=1.0e-8, gt=0)

    @field_validator(
        "frequency_threshold_thz",
        "degeneracy_tolerance_thz",
        "mean_displacement_angstrom",
        "phase_degrees",
        "qpoint_tolerance",
    )
    @classmethod
    def validate_finite_float(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("soft-mode numeric settings must be finite")
        return value


class SearchConfig(StrictModel):
    max_evaluations: int = Field(default=10, gt=0)
    max_candidates_per_iteration: int = Field(default=256, gt=0)


class CalculatorConfig(StrictModel):
    factory: str = Field(min_length=3)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("factory")
    @classmethod
    def validate_factory(cls, value: str) -> str:
        module, separator, attribute = value.partition(":")
        if not separator or not module.strip() or not attribute.strip():
            raise ValueError("factory must use 'module:attribute' syntax")
        return value


def _default_mace_calculator_config() -> CalculatorConfig:
    return CalculatorConfig(
        factory=DEFAULT_MACE_FACTORY,
        kwargs={
            "model": DEFAULT_MACE_MODEL,
            "device": DEFAULT_MACE_DEVICE,
            "default_dtype": DEFAULT_MACE_DTYPE,
            "dispersion": DEFAULT_MACE_DISPERSION,
        },
    )


class SymmetryReductionConfig(StrictModel):
    symprec: float = Field(default=0.15, gt=0)
    angle_tolerance: float = Field(default=-1.0, ge=-1.0)


class DeduplicationConfig(StrictModel):
    site_tolerance_angstrom: float = Field(default=0.15, gt=0)
    cell_length_relative_tolerance: float = Field(default=0.01, gt=0)
    cell_angle_tolerance_degrees: float = Field(default=1.0, gt=0)
    primitive_volume_tolerance_angstrom3: float = Field(default=0.1, gt=0)
    scale_volume: bool = False


class RunConfig(StrictModel):
    relaxation: RelaxationConfig = Field(default_factory=RelaxationConfig)
    candidate_relaxation: CandidateRelaxationOverrides = Field(
        default_factory=CandidateRelaxationOverrides
    )
    phonopy: PhonopyConfig = Field(default_factory=PhonopyConfig)
    soft_modes: SoftModeConfig = Field(default_factory=SoftModeConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    symmetry: SymmetryReductionConfig = Field(default_factory=SymmetryReductionConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    calculator: CalculatorConfig = Field(
        default_factory=_default_mace_calculator_config
    )

    def effective_candidate_relaxation(self) -> RelaxationConfig:
        return self.candidate_relaxation.resolve(self.relaxation)

    def resolved_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["candidate_relaxation_effective"] = (
            self.effective_candidate_relaxation().model_dump(mode="json")
        )
        return payload


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("the YAML document must be a mapping")
    # Resolved workflow artifacts include this derived audit field; it is not
    # an independent user-configurable section.
    payload.pop("candidate_relaxation_effective", None)
    return RunConfig.model_validate(payload)
