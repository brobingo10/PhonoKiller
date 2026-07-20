from __future__ import annotations

from io import StringIO
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

import phonokiller.cli as cli
from phonokiller.config import load_run_config
from phonokiller._interactive_cli import (
    _MORI_PORTRAIT,
    _PORTRAIT_WIDTH,
    _clean_path,
    _configuration_destination,
    _configuration_file,
    _frame_index,
    _json_mapping,
    _mace_model,
    _output_directory,
    _validated_section,
    collect_run_arguments,
    color_enabled,
    render_turn,
    write_generated_configuration,
)


class TtyStringIO(StringIO):
    def isatty(self) -> bool:
        return True


def _configuration(path: Path) -> Path:
    path.write_text(
        "calculator:\n  factory: tests.helpers:make_zero_calculator\n",
        encoding="utf-8",
    )
    return path


def _answers(*values: str):
    iterator = iter(values)
    return lambda: next(iterator)


def test_plain_wide_portrait_snapshot() -> None:
    stream = StringIO()
    render_turn("Status message.", stream, width=120, use_color=False)
    lines = stream.getvalue().splitlines()
    assert len(_MORI_PORTRAIT) == 24
    assert _PORTRAIT_WIDTH == 48
    assert len(lines) == 25
    assert lines[1] == (
        f"{_MORI_PORTRAIT[0].ljust(_PORTRAIT_WIDTH)}   MORI> Status message."
    )
    assert lines[2] == _MORI_PORTRAIT[1].rstrip()
    assert lines[-1] == _MORI_PORTRAIT[-1].rstrip()


def test_colored_portrait_and_no_color_fallback() -> None:
    plain = StringIO()
    colored = StringIO()
    render_turn("Status message.", plain, width=120, use_color=False)
    render_turn("Status message.", colored, width=120, use_color=True)
    assert "\x1b[38;5;201m" in colored.getvalue()
    assert re.sub(r"\x1b\[[0-9;]*m", "", colored.getvalue()) == plain.getvalue()
    assert color_enabled(TtyStringIO(), {})
    assert not color_enabled(TtyStringIO(), {"NO_COLOR": "1"})


def test_narrow_layout_stacks_portrait_above_dialogue() -> None:
    stream = StringIO()
    render_turn("Narrow status.", stream, width=80, use_color=False)
    lines = stream.getvalue().splitlines()
    assert lines[1] == _MORI_PORTRAIT[0].rstrip()
    assert lines[25] == ""
    assert lines[26] == "MORI> Narrow status."


def test_guide_reviews_supplied_defaults_and_confirms(tmp_path) -> None:
    structure = tmp_path / "input.extxyz"
    structure.touch()
    config = _configuration(tmp_path / "config.yaml")
    output = tmp_path / "run"
    stream = StringIO()
    result = collect_run_arguments(
        structure=structure,
        config=config,
        output=output,
        format=None,
        index=-1,
        no_resume=False,
        input_fn=_answers("", "", "", "", "", "", "yes"),
        stream=stream,
        width=80,
        use_color=False,
    )
    assert result is not None
    assert result.structure == structure
    assert result.config == config
    assert result.output == output
    assert result.format is None
    assert result.index == -1
    assert not result.no_resume
    transcript = stream.getvalue()
    assert "The structure argument" in transcript
    assert "The configuration argument" in transcript
    assert "The output argument" in transcript
    assert "The format argument" in transcript
    assert "The index argument" in transcript
    assert "Resume reuses checkpoints" in transcript
    assert "Resolved run arguments:" in transcript


def test_invalid_frame_index_is_explained_and_reprompted(tmp_path) -> None:
    structure = tmp_path / "input.extxyz"
    structure.touch()
    config = _configuration(tmp_path / "config.yaml")
    stream = StringIO()
    result = collect_run_arguments(
        structure=structure,
        config=config,
        output=tmp_path / "run",
        format=None,
        index=-1,
        no_resume=False,
        input_fn=_answers("", "", "", "", "last", "0", "", "yes"),
        stream=stream,
        width=80,
        use_color=False,
    )
    assert result is not None
    assert result.index == 0
    transcript = " ".join(stream.getvalue().split())
    assert "The value is invalid: the frame index must" in transcript
    assert "be an integer" in transcript


def test_path_and_configuration_validation_do_not_disclose_contents(tmp_path) -> None:
    quoted = tmp_path / "quoted file.extxyz"
    quoted.touch()
    assert _clean_path(f'"{quoted}"') == quoted
    assert _clean_path("~") == Path.home()
    with pytest.raises(ValueError, match="existing file"):
        _configuration_file(str(tmp_path / "missing.yaml"))

    malformed = tmp_path / "secret.yaml"
    malformed.write_text("secret-token: [invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="failed PhonoKiller validation") as error:
        _configuration_file(str(malformed))
    assert "secret-token" not in str(error.value)

    occupied = tmp_path / "occupied"
    occupied.touch()
    with pytest.raises(ValueError, match="not a directory"):
        _output_directory(str(occupied))
    with pytest.raises(ValueError, match="integer"):
        _frame_index("last")

    model = tmp_path / "mace.model"
    model.touch()
    assert _mace_model(str(model)) == str(model.resolve())
    assert _mace_model("medium") == "medium"
    with pytest.raises(ValueError, match="existing file"):
        _mace_model(str(tmp_path / "missing.model"))


def test_generated_configuration_is_built_from_dialogue_and_written_afterward(
    tmp_path,
) -> None:
    structure = tmp_path / "input.extxyz"
    structure.touch()
    config = tmp_path / "generated.yaml"
    output = tmp_path / "run"
    stream = StringIO()
    result = collect_run_arguments(
        structure=None,
        config=None,
        output=None,
        format=None,
        index=-1,
        no_resume=False,
        input_fn=_answers(
            str(structure),
            str(config),
            "medium",
            '{"max_steps": 12}',
            "{}",
            "{}",
            "{}",
            "{}",
            "{}",
            "{}",
            str(output),
            "auto",
            "-1",
            "yes",
            "yes",
        ),
        stream=stream,
        width=80,
        use_color=False,
    )
    assert result is not None
    assert result.generated_config is not None
    assert result.generated_config.relaxation.max_steps == 12
    assert result.generated_config.phonopy.mesh_length == 100.0
    assert not config.exists()
    assert "MACE model name or path" in stream.getvalue()

    write_generated_configuration(config, result.generated_config)
    loaded = load_run_config(config)
    assert loaded.calculator is not None
    assert loaded.calculator.factory == "phonokiller.calculators:make_mace_calculator"
    assert loaded.calculator.kwargs == {
        "model": "medium",
        "device": "cuda",
        "default_dtype": "float32",
        "dispersion": False,
    }
    assert loaded.relaxation.max_steps == 12
    with pytest.raises(FileExistsError):
        write_generated_configuration(config, result.generated_config)


def test_generated_configuration_values_are_validated_without_disclosure(
    tmp_path,
) -> None:
    destination = tmp_path / "generated.yaml"
    assert _configuration_destination(str(destination)) == destination
    with pytest.raises(ValueError, match=r"\.yaml"):
        _configuration_destination(str(tmp_path / "generated.json"))
    with pytest.raises(ValueError, match="JSON object"):
        _json_mapping("[]")
    with pytest.raises(ValueError, match="search.max_evaluations") as error:
        _validated_section("search")('{"max_evaluations": 0}')
    assert 'max_evaluations": 0' not in str(error.value)


def test_explicit_refusal_creates_no_output(tmp_path) -> None:
    structure = tmp_path / "input.extxyz"
    structure.touch()
    config = _configuration(tmp_path / "config.yaml")
    output = tmp_path / "not-created"
    result = collect_run_arguments(
        structure=structure,
        config=config,
        output=output,
        format=None,
        index=-1,
        no_resume=False,
        input_fn=_answers("", "", "", "", "", "", "no"),
        stream=StringIO(),
        width=80,
        use_color=False,
    )
    assert result is None
    assert not output.exists()


def test_generated_configuration_refusal_creates_no_files(tmp_path) -> None:
    structure = tmp_path / "input.extxyz"
    structure.touch()
    config = tmp_path / "not-created.yaml"
    output = tmp_path / "not-created-run"
    result = collect_run_arguments(
        structure=None,
        config=None,
        output=None,
        format=None,
        index=-1,
        no_resume=False,
        input_fn=_answers(
            str(structure),
            str(config),
            "medium",
            "{}",
            "{}",
            "{}",
            "{}",
            "{}",
            "{}",
            "{}",
            str(output),
            "auto",
            "-1",
            "yes",
            "no",
        ),
        stream=StringIO(),
        width=80,
        use_color=False,
    )
    assert result is None
    assert not config.exists()
    assert not output.exists()


def test_invalid_confirmation_is_explained_and_reprompted(tmp_path) -> None:
    structure = tmp_path / "input.extxyz"
    structure.touch()
    config = _configuration(tmp_path / "config.yaml")
    stream = StringIO()
    result = collect_run_arguments(
        structure=structure,
        config=config,
        output=tmp_path / "run",
        format=None,
        index=-1,
        no_resume=False,
        input_fn=_answers("", "", "", "", "", "", "maybe", "yes"),
        stream=stream,
        width=80,
        use_color=False,
    )
    assert result is not None
    assert "confirmation must be yes or no" in " ".join(stream.getvalue().split())


def test_incomplete_noninteractive_command_exits_two_without_input(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli.sys, "stdin", StringIO())
    monkeypatch.setattr(
        "builtins.input", lambda: pytest.fail("input must not be requested")
    )
    with pytest.raises(SystemExit) as error:
        cli.main(["run"])
    assert error.value.code == 2
    diagnostic = capsys.readouterr().err
    assert "structure" in diagnostic
    assert "--config" in diagnostic
    assert "--output" in diagnostic


@pytest.mark.parametrize("interruption", [EOFError(), KeyboardInterrupt()])
def test_interruption_returns_130_without_output(
    monkeypatch, tmp_path, capsys, interruption
) -> None:
    monkeypatch.setattr(cli.sys, "stdin", TtyStringIO())

    def interrupted():
        raise interruption

    monkeypatch.setattr("builtins.input", interrupted)
    output = tmp_path / "not-created"
    assert cli.main(["run", "--output", str(output)]) == 130
    assert not output.exists()
    assert "workflow was not started" in capsys.readouterr().out


def test_guided_main_launches_only_after_confirmation(monkeypatch, tmp_path) -> None:
    structure = tmp_path / "input.extxyz"
    structure.touch()
    config = _configuration(tmp_path / "config.yaml")
    output = tmp_path / "run"
    responses = "\n".join((str(structure), "", "", "", "", "", "yes"))
    monkeypatch.setattr(cli.sys, "stdin", TtyStringIO(responses + "\n"))
    calls: list[tuple] = []

    def fake_workflow(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            status="stable",
            iterations=(),
            artifacts=SimpleNamespace(
                output_dir=output,
                final_structure=output / "final" / "structure.extxyz",
            ),
        )

    monkeypatch.setattr(cli, "run_workflow", fake_workflow)
    assert cli.main(["run", "--config", str(config), "--output", str(output)]) == 0
    assert len(calls) == 1
    assert calls[0][0][0] == structure
    assert calls[0][0][3] == output
    assert calls[0][1] == {"resume": True, "format": None, "index": -1}


def test_guided_main_refusal_does_not_load_calculator(monkeypatch, tmp_path) -> None:
    structure = tmp_path / "input.extxyz"
    structure.touch()
    config = _configuration(tmp_path / "config.yaml")
    output = tmp_path / "not-created"
    responses = "\n".join((str(structure), "", "", "", "", "", "no"))
    monkeypatch.setattr(cli.sys, "stdin", TtyStringIO(responses + "\n"))
    monkeypatch.setattr(
        cli,
        "_load_calculator_factory",
        lambda *args: pytest.fail("the calculator must not be loaded"),
    )
    assert cli.main(["run", "--config", str(config), "--output", str(output)]) == 0
    assert not output.exists()


def test_guided_main_generates_yaml_only_after_confirmation(
    monkeypatch, tmp_path
) -> None:
    structure = tmp_path / "input.extxyz"
    structure.touch()
    config = tmp_path / "mori-generated.yaml"
    output = tmp_path / "run"
    responses = "\n".join(
        (
            str(structure),
            str(config),
            "medium",
            "{}",
            "{}",
            "{}",
            "{}",
            '{"max_evaluations": 3}',
            "{}",
            "{}",
            str(output),
            "auto",
            "-1",
            "yes",
            "yes",
        )
    )
    monkeypatch.setattr(cli.sys, "stdin", TtyStringIO(responses + "\n"))
    calls: list[tuple] = []

    def fake_workflow(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            status="stable",
            iterations=(),
            artifacts=SimpleNamespace(
                output_dir=output,
                final_structure=output / "final" / "structure.extxyz",
            ),
        )

    monkeypatch.setattr(cli, "run_workflow", fake_workflow)
    assert cli.main(["run"]) == 0
    assert len(calls) == 1
    assert config.is_file()
    assert load_run_config(config).search.max_evaluations == 3


def test_complete_command_remains_prompt_free(monkeypatch, tmp_path, capsys) -> None:
    structure = tmp_path / "input.extxyz"
    structure.touch()
    config = _configuration(tmp_path / "config.yaml")
    output = tmp_path / "run"
    monkeypatch.setattr(cli.sys, "stdin", StringIO())
    monkeypatch.setattr(
        cli,
        "run_workflow",
        lambda *args, **kwargs: SimpleNamespace(
            status="stable",
            iterations=(),
            artifacts=SimpleNamespace(
                output_dir=output,
                final_structure=output / "final" / "structure.extxyz",
            ),
        ),
    )
    assert (
        cli.main(
            [
                "run",
                str(structure),
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert "MORI>" not in capsys.readouterr().out
