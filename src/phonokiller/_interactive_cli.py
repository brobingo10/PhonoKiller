"""Private character-guided input collection for the command-line interface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import textwrap
from typing import TextIO, TypeVar
from uuid import uuid4

import yaml

from .config import (
    DEFAULT_MACE_DEVICE,
    DEFAULT_MACE_DISPERSION,
    DEFAULT_MACE_DTYPE,
    DEFAULT_MACE_FACTORY,
    DEFAULT_MACE_MODEL,
    RunConfig,
    load_run_config,
)


_MORI_PORTRAIT = (
    "           \u2840\u2804\u2802\u280a\u2808                \u2809\u2810\u2802\u2824\u2840\u2840",
    "       \u2840\u2804\u2802\u2801       \u2880\u28c0\u28e0\u2864\u2864\u2824\u2824\u2824\u2804\u28c4\u2840        \u2808\u2812\u2804\u2840",
    "    \u2880\u2814\u2808      \u2840\u28e0\u2874\u283e\u281f\u281f\u280d\u2804\u2840      \u2808\u2819\u2832\u28a4\u28c0       \u2808\u2811\u2884\u2840",
    "  \u2840\u2806\u2801     \u28c0\u2874\u281a\u280b\u2805\u2802      \u2802         \u2839\u28f7\u28c4        \u2808\u2822\u2840",
    "\u28a0\u2818      \u28e0\u2812\u2809  \u2802           \u2884      \u2884\u28ff\u28ff\u28f7\u2844        \u2808\u2822\u2840",
    "\u2801     \u2880\u285e\u2801             \u2880 \u2880\u2874\u2863\u2850\u28a4\u2864\u28a6\u281e\u283f\u283b\u2805\u2818\u28bf\u28c6         \u2808",
    "     \u28f0\u28ff\u2880             \u28c4\u2806\u2811\u2829\u2808 \u2808 \u2802       \u28bb\u28c6",
    "    \u28f8\u28ff\u287f\u2878\u28e7\u2812\u2827\u2814\u2808\u281a\u2810 \u2808\u2804\u2812\u2818\u280a                 \u28bb\u2846",
    "   \u28b0\u2807\u2811\u2801     \u2890\u2804                        \u2808\u28bf\u2844",
    "   \u284f         \u28f7\u2840  \u2890\u2840                    \u2818\u28f7",
    "   \u2887     \u2828\u2884  \u2838\u28ff\u28e6\u2840 \u2818\u2886\u28c4\u28d1\u28c0\u2840      \u28e0\u2864\u2844       \u283a\u28c7",
    "   \u2811\u2844   \u2880\u2880\u28d8\u28f6\u28f4\u28e2\u28e4\u2874\u287e\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28f7    \u2880\u287c\u28b5\u28ff\u28be\u2840       \u28bb\u2840",
    "    \u2822\u2840\u2804\u2840 \u2836\u28dd\u28bf\u28ff\u28ff\u28ff\u28c2\u2840\u2860\u28f6\u28f4\u28fe\u28ff\u28ff\u28ff\u2842   \u28b8\u28df\u28f2\u28bd\u28ff\u2801       \u2818\u28c7",
    "        \u2821 \u2880\u283c\u28ff\u28ff\u28ff\u28ff\u28ff\u28fe\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u284e   \u2810\u28e7\u28fe\u287f\u2803   \u2820     \u2818\u2844",
    "        \u2808\u2804\u2810\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u2847   \u2808\u281b\u280b      \u2882     \u2831\u2844",
    "         \u2847\u2808\u28bf\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u2847    \u2840        \u2823    \u2884\u28ff\u2880",
    "        \u2820\u2841 \u2818\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u2842   \u2890          \u2801\u2822\u2844\u2890\u281c\u28af\u28b1",
    "        \u2810   \u2808\u28b6\u28fe\u28fe\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff\u28ff    \u28e8\u2802           \u2891  \u2808\u2842\u28c7",
    "        \u28a8     \u283b\u28ff\u28ff\u28ff\u28ff\u28ff\u287f\u289f\u28ff\u28ff    \u28fe             \u2840  \u2806\u28f9",
    "        \u28fc      \u2819\u283b\u281b\u2809\u2801 \u28bb\u28ff\u28ff\u2804   \u2813                \u2883\u280e",
    "    \u2828\u2884\u28c0\u2834\u2801 \u2880\u2802         \u2808\u2801           \u2820         \u2870\u2811",
    "    \u2808\u2821\u2804\u2840 \u2804\u285e\u2860                           \u2840\u2880\u2820\u280a\u281c\u2801",
    "      \u2820\u28a0\u2860\u283e\u280b  \u2840                       \u28c6\u2808 \u2802 \u2808",
    "          \u2810\u2810\u2888 \u2814\u2808\u2820\u2820\u2820                  \u283c\u28c4\u2840",
)

_MORI_COLORS = (
    201,
    200,
    199,
    198,
    197,
    196,
    196,
    202,
    202,
    208,
    208,
    214,
    214,
    208,
    208,
    202,
    202,
    196,
    197,
    198,
    199,
    200,
    201,
    201,
)
_PORTRAIT_WIDTH = max(len(line) for line in _MORI_PORTRAIT)
_SIDE_BY_SIDE_MINIMUM = _PORTRAIT_WIDTH + 52
_T = TypeVar("_T")


class InteractiveCancelled(Exception):
    """The guided interaction ended through EOF or an interrupt."""


@dataclass(frozen=True, slots=True)
class GuidedRunArguments:
    """Fully validated CLI arguments returned by the interactive guide."""

    structure: Path
    config: Path
    output: Path
    format: str | None
    index: int
    no_resume: bool
    generated_config: RunConfig | None = None


def interactive_terminal_available(stream: TextIO) -> bool:
    """Return whether *stream* supports an interactive conversation."""

    try:
        return bool(stream.isatty())
    except (AttributeError, OSError):
        return False


def color_enabled(stream: TextIO, environment: Mapping[str, str] | None = None) -> bool:
    """Return whether ANSI color should be emitted to *stream*."""

    values = os.environ if environment is None else environment
    return interactive_terminal_available(stream) and "NO_COLOR" not in values


def terminal_width() -> int:
    """Return a conservative terminal width for dialogue layout."""

    return max(20, shutil.get_terminal_size(fallback=(80, 24)).columns)


def render_turn(
    message: str,
    stream: TextIO,
    *,
    width: int,
    use_color: bool,
) -> None:
    """Render one compact Mori portrait and an informative dialogue turn."""

    side_by_side = width >= _SIDE_BY_SIDE_MINIMUM
    dialogue_width = max(24, width - _PORTRAIT_WIDTH - 3) if side_by_side else width
    dialogue = _dialogue_lines(message, dialogue_width)
    portrait = _portrait_lines(use_color, pad=side_by_side)
    stream.write("\n")
    if side_by_side:
        row_count = max(len(portrait), len(dialogue))
        for index in range(row_count):
            left = portrait[index] if index < len(portrait) else " " * _PORTRAIT_WIDTH
            right = dialogue[index] if index < len(dialogue) else ""
            stream.write(f"{left}   {right}".rstrip() + "\n")
    else:
        for line in portrait:
            stream.write(line.rstrip() + "\n")
        stream.write("\n")
        for line in dialogue:
            stream.write(line.rstrip() + "\n")
    stream.flush()


def collect_run_arguments(
    *,
    structure: Path | None,
    config: Path | None,
    output: Path | None,
    format: str | None,
    index: int | str,
    no_resume: bool,
    input_fn: Callable[[], str],
    stream: TextIO,
    width: int,
    use_color: bool,
) -> GuidedRunArguments | None:
    """Review every run argument and return confirmed values, or ``None``."""

    try:
        selected_structure = _ask_value(
            "The structure argument identifies one ASE-readable crystal file. "
            "PhonoKiller requires periodic boundary conditions in all three dimensions.",
            "Enter the structure file",
            _display_path(structure),
            _existing_file,
            input_fn,
            stream,
            width,
            use_color,
        )
        selected_config, generated_config = _collect_configuration(
            config,
            input_fn,
            stream,
            width,
            use_color,
        )
        selected_output = _ask_value(
            "The output argument selects the workflow directory for checkpoints, "
            "history, iteration artifacts, and the stable final export. Existing "
            "matching output can be resumed.",
            "Enter the output directory",
            _display_path(output) or "phonokiller-run",
            _output_directory,
            input_fn,
            stream,
            width,
            use_color,
        )
        selected_format = _ask_value(
            "The format argument optionally forces an ASE input format. Use 'auto' "
            "to let ASE infer the format from the file name.",
            "Enter the ASE format",
            format or "auto",
            _ase_format,
            input_fn,
            stream,
            width,
            use_color,
        )
        selected_index = _ask_value(
            "The index argument selects one integer frame from a multi-frame file. "
            "The value -1 selects the last frame.",
            "Enter the frame index",
            str(index),
            _frame_index,
            input_fn,
            stream,
            width,
            use_color,
        )
        resume = _ask_value(
            "Resume reuses checkpoints only when the structure, configuration, "
            "calculator, and dependency fingerprints match. Disabling resume "
            "requires an unused output directory.",
            "Enable resume? Enter yes or no",
            "no" if no_resume else "yes",
            _yes_or_no,
            input_fn,
            stream,
            width,
            use_color,
        )
        resolved = GuidedRunArguments(
            structure=selected_structure,
            config=selected_config,
            output=selected_output,
            format=selected_format,
            index=selected_index,
            no_resume=not resume,
            generated_config=generated_config,
        )
        render_turn(_summary(resolved), stream, width=width, use_color=use_color)
        confirmed = _ask_confirmation(input_fn, stream, width, use_color)
    except (EOFError, KeyboardInterrupt) as exc:
        raise InteractiveCancelled from exc
    if not confirmed:
        render_turn(
            "The workflow was not started. No workflow artifacts were created.",
            stream,
            width=width,
            use_color=use_color,
        )
        return None
    return resolved


def write_generated_configuration(path: Path, config: RunConfig) -> None:
    """Atomically write a configuration assembled by the guided conversation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config.model_dump(mode="json"), stream, sort_keys=False)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _collect_configuration(
    initial: Path | None,
    input_fn: Callable[[], str],
    stream: TextIO,
    width: int,
    use_color: bool,
) -> tuple[Path, RunConfig | None]:
    if initial is not None and initial.is_file():
        selected = _ask_value(
            "The configuration argument identifies an existing PhonoKiller YAML "
            "file. It defines the MACE calculator selection and all workflow "
            "settings.",
            "Enter the existing configuration file",
            str(initial),
            _configuration_file,
            input_fn,
            stream,
            width,
            use_color,
        )
        return selected, None

    destination = _ask_value(
        "Mori will generate the configuration from this conversation. The "
        "configuration destination must be a new YAML file and is written only "
        "after the final launch confirmation.",
        "Enter the new configuration file",
        str(initial) if initial is not None else str(_available_config_destination()),
        _configuration_destination,
        input_fn,
        stream,
        width,
        use_color,
    )
    generated = _build_configuration(input_fn, stream, width, use_color)
    return destination, generated


def _build_configuration(
    input_fn: Callable[[], str], stream: TextIO, width: int, use_color: bool
) -> RunConfig:
    model = _ask_value(
        "PhonoKiller uses MACE by default. Enter a MACE-MP model name, such as "
        "'small', 'medium', or 'large', or an existing local .model checkpoint "
        "path. The default uses CUDA and float32; MACE-MP model names are "
        "downloaded and cached by MACE.",
        "Enter the MACE model name or path",
        DEFAULT_MACE_MODEL,
        _mace_model,
        input_fn,
        stream,
        width,
        use_color,
    )
    payload: dict[str, object] = {
        "calculator": {
            "factory": DEFAULT_MACE_FACTORY,
            "kwargs": {
                "model": model,
                "device": DEFAULT_MACE_DEVICE,
                "default_dtype": DEFAULT_MACE_DTYPE,
                "dispersion": DEFAULT_MACE_DISPERSION,
            },
        }
    }
    for section, explanation in _configuration_sections():
        payload[section] = _ask_value(
            explanation,
            f"Enter {section} overrides as JSON",
            "{}",
            _validated_section(section),
            input_fn,
            stream,
            width,
            use_color,
        )
    return RunConfig.model_validate(payload)


def _ask_value(
    explanation: str,
    prompt: str,
    default: str | None,
    parser: Callable[[str], _T],
    input_fn: Callable[[], str],
    stream: TextIO,
    width: int,
    use_color: bool,
) -> _T:
    while True:
        render_turn(explanation, stream, width=width, use_color=use_color)
        suffix = f" [{default}]" if default is not None else ""
        stream.write(f"MORI> {prompt}{suffix}: ")
        stream.flush()
        response = input_fn().strip()
        candidate = response or default
        if candidate is None:
            render_turn(
                "A value is required for this argument.",
                stream,
                width=width,
                use_color=use_color,
            )
            continue
        try:
            return parser(candidate)
        except ValueError as exc:
            render_turn(
                f"The value is invalid: {exc}",
                stream,
                width=width,
                use_color=use_color,
            )


def _ask_confirmation(
    input_fn: Callable[[], str], stream: TextIO, width: int, use_color: bool
) -> bool:
    while True:
        render_turn(
            "Starting now will load the calculator and begin relaxation and force "
            "calculations. Enter yes to start; a blank response does not start the run.",
            stream,
            width=width,
            use_color=use_color,
        )
        stream.write("MORI> Start the workflow? [no]: ")
        stream.flush()
        response = input_fn().strip().lower()
        if response in {"yes", "y"}:
            return True
        if response in {"", "no", "n"}:
            return False
        render_turn(
            "The confirmation must be yes or no.",
            stream,
            width=width,
            use_color=use_color,
        )


def _dialogue_lines(message: str, width: int) -> list[str]:
    prefix = "MORI> "
    content_width = max(10, width - len(prefix))
    wrapped: list[str] = []
    for paragraph in message.splitlines() or [""]:
        if not paragraph:
            wrapped.append("")
            continue
        wrapped.extend(
            textwrap.wrap(
                paragraph,
                width=content_width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
    return [
        (prefix if index == 0 else " " * len(prefix)) + line
        for index, line in enumerate(wrapped)
    ]


def _portrait_lines(use_color: bool, *, pad: bool) -> list[str]:
    lines: list[str] = []
    for line, color in zip(_MORI_PORTRAIT, _MORI_COLORS, strict=True):
        visible = line.rstrip()
        if use_color:
            colored = f"\x1b[38;5;{color}m{visible}\x1b[0m"
            lines.append(
                colored + (" " * (_PORTRAIT_WIDTH - len(visible)) if pad else "")
            )
        else:
            lines.append(visible.ljust(_PORTRAIT_WIDTH) if pad else visible)
    return lines


def _clean_path(value: str) -> Path:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    if not cleaned:
        raise ValueError("the path cannot be empty")
    return Path(cleaned).expanduser()


def _existing_file(value: str) -> Path:
    path = _clean_path(value)
    if not path.is_file():
        raise ValueError("the path must identify an existing file")
    return path


def _configuration_file(value: str) -> Path:
    path = _existing_file(value)
    try:
        parsed = load_run_config(path)
    except Exception as exc:
        raise ValueError(
            f"the YAML failed PhonoKiller validation ({type(exc).__name__})"
        ) from exc
    if parsed.calculator is None:
        raise ValueError("the YAML must define calculator.factory")
    return path


def _available_config_destination() -> Path:
    preferred = Path("phonokiller.yaml")
    if not preferred.exists():
        return preferred
    generated = Path("phonokiller.generated.yaml")
    if not generated.exists():
        return generated
    counter = 2
    while True:
        numbered = Path(f"phonokiller.generated-{counter}.yaml")
        if not numbered.exists():
            return numbered
        counter += 1


def _configuration_destination(value: str) -> Path:
    path = _clean_path(value)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("the configuration destination must end in .yaml or .yml")
    if path.exists():
        raise ValueError("the configuration destination must not already exist")
    if path.parent.exists() and not path.parent.is_dir():
        raise ValueError("the configuration parent path is not a directory")
    return path


def _mace_model(value: str) -> str:
    selected = value.strip()
    if len(selected) >= 2 and selected[0] == selected[-1] and selected[0] in {'"', "'"}:
        selected = selected[1:-1].strip()
    if not selected:
        raise ValueError("enter a MACE-MP model name or a local model path")
    path = Path(selected).expanduser()
    if path.is_file():
        return str(path.resolve())
    if _looks_like_path(selected):
        raise ValueError("the local MACE model path must identify an existing file")
    return selected


def _looks_like_path(value: str) -> bool:
    return (
        value.startswith((".", "~", "/", "\\"))
        or "/" in value
        or "\\" in value
        or value.endswith(".model")
    )


def _json_mapping(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "enter a JSON object with double-quoted keys, or {} for defaults"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("the value must be a JSON object")
    return parsed


def _configuration_sections() -> tuple[tuple[str, str], ...]:
    return (
        (
            "relaxation",
            "Base relaxation settings control the ASE optimizer. The defaults are "
            "positions-only BFGS relaxation, 0.005 eV/angstrom force tolerance, "
            "and 500 steps. JSON keys are mode, optimizer, force_tolerance, and "
            "max_steps; use {} to keep all defaults.",
        ),
        (
            "candidate_relaxation",
            "Candidate relaxation overrides inherit every omitted base relaxation "
            "value. JSON keys are mode, optimizer, force_tolerance, and max_steps; "
            "use {} for complete inheritance.",
        ),
        (
            "phonopy",
            "Phonopy sizing defaults request a 10 angstrom minimum finite-displacement "
            "supercell span and a scalar mesh length of 100. JSON keys are "
            "minimum_supercell_span_angstrom and mesh_length; use {} for defaults.",
        ),
        (
            "soft_modes",
            "Soft-mode settings default to a -0.05 THz stability threshold, 0.001 "
            "THz degeneracy tolerance, five groups, 0.1 angstrom mean displacement, "
            "zero-degree phase, and 1e-8 q-point tolerance. Use {} for defaults.",
        ),
        (
            "search",
            "Search limits default to ten Phonopy evaluations and 256 candidates per "
            "iteration. JSON keys are max_evaluations and "
            "max_candidates_per_iteration; use {} for defaults.",
        ),
        (
            "symmetry",
            "Primitive reduction defaults use symprec 0.15 and automatic angle "
            "tolerance -1. JSON keys are symprec and angle_tolerance; use {} for "
            "defaults.",
        ),
        (
            "deduplication",
            "Deduplication compares sites, cell lengths, angles, and primitive "
            "volumes. The defaults are 0.15 angstrom, 0.01 relative length, 1 "
            "degree, 0.1 cubic angstrom, and no volume scaling. Use {} for defaults.",
        ),
    )


def _validated_section(section: str) -> Callable[[str], dict[str, object]]:
    def parse(value: str) -> dict[str, object]:
        selected = _json_mapping(value)
        try:
            RunConfig.model_validate({section: selected})
        except Exception as exc:
            errors = getattr(exc, "errors", None)
            if callable(errors):
                first = errors(include_url=False, include_input=False)[0]
                location = ".".join(str(part) for part in first["loc"])
                raise ValueError(f"{location}: {first['msg']}") from exc
            raise ValueError(
                f"the {section} section failed validation ({type(exc).__name__})"
            ) from exc
        return selected

    return parse


def _output_directory(value: str) -> Path:
    path = _clean_path(value)
    if path.exists() and not path.is_dir():
        raise ValueError("the output path exists and is not a directory")
    return path


def _ase_format(value: str) -> str | None:
    selected = value.strip()
    if selected.lower() in {"auto", "none"}:
        return None
    if not selected:
        raise ValueError("enter an ASE format or 'auto'")
    return selected


def _frame_index(value: str) -> int:
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError("the frame index must be an integer") from exc


def _yes_or_no(value: str) -> bool:
    selected = value.strip().lower()
    if selected in {"yes", "y"}:
        return True
    if selected in {"no", "n"}:
        return False
    raise ValueError("enter yes or no")


def _display_path(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _summary(arguments: GuidedRunArguments) -> str:
    configuration = str(arguments.config)
    if arguments.generated_config is not None:
        configuration += " (generated after confirmation)"
    return "\n".join(
        (
            "Resolved run arguments:",
            f"Structure: {arguments.structure}",
            f"Configuration: {configuration}",
            f"Output: {arguments.output}",
            f"ASE format: {arguments.format or 'automatic detection'}",
            f"Frame index: {arguments.index}",
            f"Resume: {'enabled' if not arguments.no_resume else 'disabled'}",
        )
    )
