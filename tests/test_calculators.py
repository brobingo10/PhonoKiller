from __future__ import annotations

from pathlib import Path

import phonokiller.calculators as calculators


def test_named_model_uses_mace_mp_with_float32_defaults(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def mace_mp(**kwargs):
        calls.update(kwargs)
        return "mace-mp-calculator"

    monkeypatch.setattr(calculators, "_mace_api", lambda: (object, mace_mp))
    calculator = calculators.make_mace_calculator(context=object(), model="medium")

    assert calculator == "mace-mp-calculator"
    assert calls == {
        "model": "medium",
        "device": "cuda",
        "default_dtype": "float32",
        "dispersion": False,
    }


def test_local_checkpoint_uses_mace_calculator(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "fine_tuned.model"
    checkpoint.touch()
    calls: dict[str, object] = {}

    class FakeMACECalculator:
        def __init__(self, **kwargs) -> None:
            calls.update(kwargs)

    monkeypatch.setattr(
        calculators,
        "_mace_api",
        lambda: (FakeMACECalculator, lambda **kwargs: None),
    )
    calculator = calculators.make_mace_calculator(
        context=object(),
        model=str(checkpoint),
        device="cpu",
        default_dtype="float64",
    )

    assert isinstance(calculator, FakeMACECalculator)
    assert calls == {
        "model_paths": str(checkpoint.resolve()),
        "device": "cpu",
        "default_dtype": "float64",
    }


def test_missing_local_checkpoint_fails_before_importing_mace(tmp_path) -> None:
    missing = Path(tmp_path / "missing.model")

    try:
        calculators.make_mace_calculator(context=object(), model=str(missing))
    except FileNotFoundError as error:
        assert str(missing) in str(error)
    else:
        raise AssertionError("a missing local model checkpoint must be rejected")
