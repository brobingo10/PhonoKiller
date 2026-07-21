"""Internal resource estimates used before candidate materialization."""

from __future__ import annotations

from typing import Any

from .config import OptimizerName


_FLOAT64_BYTES = 8
_LBFGS_DEFAULT_HISTORY = 100


def optimizer_state_estimate(
    atom_count: int, optimizer: OptimizerName
) -> dict[str, Any]:
    """Return a conservative optimizer-state estimate for one candidate.

    The estimate intentionally excludes calculator/model memory. It captures
    the scaling term controlled by the ASE optimizer: the dense inverse
    Hessian for BFGS, the default pair history for LBFGS, or the velocity array
    for FIRE.
    """

    if atom_count <= 0:
        raise ValueError("atom_count must be positive")
    degrees_of_freedom = 3 * atom_count
    if optimizer == OptimizerName.BFGS:
        bytes_required = degrees_of_freedom**2 * _FLOAT64_BYTES
        model = "dense_float64_hessian"
    elif optimizer == OptimizerName.LBFGS:
        bytes_required = (
            2 * _LBFGS_DEFAULT_HISTORY * degrees_of_freedom * _FLOAT64_BYTES
            + _LBFGS_DEFAULT_HISTORY * _FLOAT64_BYTES
        )
        model = "float64_s_y_history_100"
    else:
        bytes_required = degrees_of_freedom * _FLOAT64_BYTES
        model = "float64_velocity"
    return {
        "optimizer": optimizer.value,
        "degrees_of_freedom": degrees_of_freedom,
        "model": model,
        "bytes": int(bytes_required),
        "mib": float(bytes_required / 1024**2),
    }


def candidate_resource_violations(
    atom_count: int,
    optimizer: OptimizerName,
    *,
    max_candidate_atoms: int,
    max_dense_hessian_memory_mib: float,
    group_rank: int | None = None,
) -> tuple[str, ...]:
    """Return clear violations for an otherwise materializable candidate."""

    prefix = f"mode group {group_rank} " if group_rank is not None else "candidate "
    violations: list[str] = []
    if atom_count > max_candidate_atoms:
        violations.append(
            f"{prefix}requires {atom_count} atoms per candidate, exceeding "
            f"search.max_candidate_atoms={max_candidate_atoms}"
        )
    estimate = optimizer_state_estimate(atom_count, optimizer)
    if (
        optimizer == OptimizerName.BFGS
        and float(estimate["mib"]) > max_dense_hessian_memory_mib
    ):
        violations.append(
            f"{prefix}would allocate an estimated {float(estimate['mib']):.1f} MiB "
            "dense BFGS Hessian, exceeding "
            "search.max_dense_hessian_memory_mib="
            f"{max_dense_hessian_memory_mib:g}; use FIRE or LBFGS for candidates"
        )
    return tuple(violations)
