"""Command-line interface for the unified PhonoKiller workflow."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Sequence

from ase.calculators.calculator import Calculator

from .config import CalculatorConfig, load_run_config
from .exceptions import CalculatorValidationError
from ._interactive_cli import (
    InteractiveCancelled,
    collect_run_arguments,
    color_enabled,
    interactive_terminal_available,
    terminal_width,
    write_generated_configuration,
)
from .workflow import run_workflow


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    interactive = interactive_terminal_available(sys.stdin)
    if not raw_arguments and interactive:
        raw_arguments = ["run"]
    args = parser.parse_args(raw_arguments)
    generated_config = None
    missing = _missing_run_arguments(args)
    if missing:
        if not interactive:
            parser.error("the following arguments are required: " + ", ".join(missing))
        try:
            guided = collect_run_arguments(
                structure=args.structure,
                config=args.config,
                output=args.output,
                format=args.format,
                index=args.index,
                no_resume=args.no_resume,
                input_fn=input,
                stream=sys.stdout,
                width=terminal_width(),
                use_color=color_enabled(sys.stdout),
            )
        except InteractiveCancelled:
            print("\nMORI> Input cancelled; the workflow was not started.")
            return 130
        if guided is None:
            return 0
        args.structure = guided.structure
        args.config = guided.config
        args.output = guided.output
        args.format = guided.format
        args.index = guided.index
        args.no_resume = guided.no_resume
        generated_config = guided.generated_config
    try:
        if generated_config is not None:
            write_generated_configuration(args.config, generated_config)
            print(f"MORI> Generated configuration: {args.config}")
        config = load_run_config(args.config)
        if config.calculator is None:
            raise CalculatorValidationError(
                "run configuration must define calculator.factory"
            )
        result = run_workflow(
            args.structure,
            _load_calculator_factory(config.calculator),
            config,
            args.output,
            resume=not args.no_resume,
            format=args.format,
            index=args.index,
            progress=_print_progress,
        )
    except Exception as exc:
        print(f"phonokiller: error: {exc}", file=sys.stderr)
        return 2
    print(
        f"workflow {result.status}: {len(result.iterations)} "
        f"Phonopy evaluation(s) in {result.artifacts.output_dir}"
    )
    if result.status == "stable":
        print(f"Final structure: {result.artifacts.final_structure}")
        return 0
    print(f"Search history: {result.artifacts.history}")
    return 1


def _print_progress(message: str) -> None:
    """Write one immediately visible workflow progress event."""

    print(f"PHONOKILLER> {message}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phonokiller",
        description="Iteratively remove soft phonon modes from periodic structures.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run", help="run the complete relaxation and phonon-stability search"
    )
    run.add_argument(
        "structure", nargs="?", type=Path, help="input structure readable by ASE"
    )
    run.add_argument("--config", type=Path, help="unified YAML file")
    run.add_argument("--output", type=Path, help="workflow directory")
    run.add_argument("--format", default=None, help="explicit ASE input format")
    run.add_argument(
        "--index", type=int, default=-1, help="ASE frame index (default: -1)"
    )
    run.add_argument(
        "--no-resume",
        action="store_true",
        help="do not reuse matching workflow checkpoints",
    )
    return parser


def _missing_run_arguments(args: argparse.Namespace) -> list[str]:
    missing: list[str] = []
    if args.structure is None:
        missing.append("structure")
    if args.config is None:
        missing.append("--config")
    if args.output is None:
        missing.append("--output")
    return missing


def _load_calculator_factory(calculator_config: CalculatorConfig):
    module_name, _, attribute_name = calculator_config.factory.partition(":")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute_name)
    except (ImportError, AttributeError) as exc:
        raise CalculatorValidationError(
            f"cannot import calculator factory {calculator_config.factory!r}: {exc}"
        ) from exc
    if not callable(factory):
        raise CalculatorValidationError(
            f"calculator factory {calculator_config.factory!r} is not callable"
        )

    def provider(*, context) -> Calculator:
        try:
            calculator = factory(context=context, **calculator_config.kwargs)
        except Exception as exc:
            raise CalculatorValidationError(
                f"calculator factory failed for {context.stage}: {exc}"
            ) from exc
        if not isinstance(calculator, Calculator):
            raise CalculatorValidationError(
                f"calculator factory returned {type(calculator).__name__}, "
                "not an ASE Calculator"
            )
        return calculator

    return provider
