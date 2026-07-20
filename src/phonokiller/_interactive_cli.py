"""Private character-guided input collection for the command-line interface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import textwrap
from typing import TextIO, TypeVar

from .config import load_run_config


_MORI_PORTRAIT = (
    "      __..----..__       //",
    "  _.-'  _..--.._  `-._ // ",
    " /    .'  /\\ /\\  `.   V/  ",
    "|    /   (o   o)   \\  /|  ",
    "|   |       ^       |// |  ",
    " \\  |    `---'     // /   ",
    "  `._\\   .-=-.    //.'    ",
    "     /`--|___|--'//\\      ",
    "    /___/|   |\\_//__\\     ",
    "      _/ |___| //\\_       ",
    "     /___/   \\//___\\      ",
    "             //)           ",
)
_MORI_COLORS = (201, 200, 199, 198, 197, 196, 202, 208, 209, 214, 215, 216)
_PORTRAIT_WIDTH = max(len(line) for line in _MORI_PORTRAIT)
_SIDE_BY_SIDE_MINIMUM = _PORTRAIT_WIDTH + 38
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
        selected_config = _ask_value(
            "The configuration argument identifies the YAML file containing the "
            "calculator factory and all relaxation, Phonopy, and search settings.",
            "Enter the configuration file",
            _display_path(config),
            _configuration_file,
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
        )
        render_turn(
            _summary(resolved), stream, width=width, use_color=use_color
        )
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
    return "\n".join(
        (
            "Resolved run arguments:",
            f"Structure: {arguments.structure}",
            f"Configuration: {arguments.config}",
            f"Output: {arguments.output}",
            f"ASE format: {arguments.format or 'automatic detection'}",
            f"Frame index: {arguments.index}",
            f"Resume: {'enabled' if not arguments.no_resume else 'disabled'}",
        )
    )
