"""Built-in ASE calculator factories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator

from .config import (
    DEFAULT_MACE_DEVICE,
    DEFAULT_MACE_DISPERSION,
    DEFAULT_MACE_DTYPE,
    DEFAULT_MACE_MODEL,
)


def make_mace_calculator(
    *,
    context: object,
    model: str = DEFAULT_MACE_MODEL,
    device: str = DEFAULT_MACE_DEVICE,
    default_dtype: str = DEFAULT_MACE_DTYPE,
    dispersion: bool = DEFAULT_MACE_DISPERSION,
    **kwargs: Any,
) -> Calculator:
    """Return a fresh MACE ASE calculator for one PhonoKiller context.

    A value identifying an existing file is loaded as a local MACE checkpoint.
    Any other value is passed to :func:`mace.calculators.mace_mp` as a MACE-MP
    model name, such as ``"medium"``.
    """

    del context
    model_path = _local_model_path(model)
    MACECalculator, mace_mp = _mace_api()
    if model_path is not None:
        return MACECalculator(
            model_paths=str(model_path),
            device=device,
            default_dtype=default_dtype,
            **kwargs,
        )
    return mace_mp(
        model=model,
        device=device,
        default_dtype=default_dtype,
        dispersion=dispersion,
        **kwargs,
    )


def _mace_api() -> tuple[type[Calculator], Any]:
    try:
        from mace.calculators import MACECalculator, mace_mp
    except ImportError as exc:
        raise ImportError(
            "MACE is the default calculator. Install a CUDA-compatible PyTorch "
            "build, then install PhonoKiller with `pip install 'phonokiller[mace]'`."
        ) from exc
    return MACECalculator, mace_mp


def _local_model_path(model: str) -> Path | None:
    value = str(model).strip()
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    if _looks_like_path(value):
        raise FileNotFoundError(f"the MACE model path does not exist: {path}")
    return None


def _looks_like_path(value: str) -> bool:
    return (
        value.startswith((".", "~", "/", "\\"))
        or "/" in value
        or "\\" in value
        or value.endswith(".model")
    )
