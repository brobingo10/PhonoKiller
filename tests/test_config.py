from __future__ import annotations

import pytest
from pydantic import ValidationError

from phonokiller import RunConfig
from phonokiller.config import (
    DEFAULT_MACE_DEVICE,
    DEFAULT_MACE_DISPERSION,
    DEFAULT_MACE_DTYPE,
    DEFAULT_MACE_FACTORY,
    DEFAULT_MACE_MODEL,
)


def test_unified_defaults_and_candidate_inheritance() -> None:
    config = RunConfig()
    assert config.relaxation.mode.value == "positions"
    assert config.relaxation.optimizer.value == "BFGS"
    assert config.phonopy.minimum_supercell_span_angstrom == 10.0
    assert config.phonopy.mesh_length == 100.0
    assert config.soft_modes.frequency_threshold_thz == -0.05
    assert config.soft_modes.degeneracy_tolerance_thz == 1.0e-3
    assert config.soft_modes.max_mode_groups == 5
    assert config.soft_modes.mean_displacement_angstrom == 0.1
    assert config.search.max_evaluations == 10
    assert config.search.max_candidates_per_iteration == 256
    assert config.calculator.factory == DEFAULT_MACE_FACTORY
    assert config.calculator.kwargs == {
        "model": DEFAULT_MACE_MODEL,
        "device": DEFAULT_MACE_DEVICE,
        "default_dtype": DEFAULT_MACE_DTYPE,
        "dispersion": DEFAULT_MACE_DISPERSION,
    }
    assert config.effective_candidate_relaxation() == config.relaxation


def test_candidate_overrides_layer_on_base() -> None:
    config = RunConfig.model_validate(
        {
            "relaxation": {
                "mode": "full_cell",
                "optimizer": "FIRE",
                "force_tolerance": 0.01,
                "max_steps": 100,
            },
            "candidate_relaxation": {"mode": "positions", "max_steps": 800},
        }
    )
    effective = config.effective_candidate_relaxation()
    assert effective.mode.value == "positions"
    assert effective.optimizer.value == "FIRE"
    assert effective.force_tolerance == 0.01
    assert effective.max_steps == 800


@pytest.mark.parametrize(
    "payload",
    [
        {"phonopy": {"minimum_supercell_span_angstrom": 0}},
        {"phonopy": {"mesh_length": 0}},
        {"soft_modes": {"frequency_threshold_thz": 0}},
        {"soft_modes": {"mean_displacement_angstrom": -0.1}},
        {"search": {"max_evaluations": 0}},
        {"search": {"max_candidates_per_iteration": 0}},
        {"candidate_relaxation": {"max_steps": 0}},
    ],
)
def test_invalid_unified_config_is_rejected(payload) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_removed_stage_sections_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"candidate_reduction": {"relaxation": {}}})
