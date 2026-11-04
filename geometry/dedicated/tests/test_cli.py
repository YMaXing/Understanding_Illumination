"""CLI and config path handling: YAML-relative paths, precedence YAML < --set < --input/--output, infer errors."""

import os
from pathlib import Path

import pytest

from illum_baseline import DEFAULT_CONFIG
from illum_baseline.cli import _cfg, build_parser, main


def _yaml(tmp_path: Path, data_dir: str, out_dir: str) -> Path:
    text = Path(DEFAULT_CONFIG).read_text()
    text = text.replace("  data_dir: ../data ", f"  data_dir: {data_dir} ").replace(
        "  out_dir: ../baseline_output ", f"  out_dir: {out_dir} ")
    cfgdir = tmp_path / "cfgs"
    cfgdir.mkdir()
    p = cfgdir / "my.yaml"
    p.write_text(text)
    return p


def test_default_config_paths_resolve_next_to_the_repo():
    args = build_parser().parse_args(["infer"])
    cfg = _cfg(args)
    root = Path(DEFAULT_CONFIG).resolve().parent.parent
    assert cfg.data_dir == root / "data" and cfg.out_dir == root / "baseline_output"


def test_yaml_relative_paths_are_relative_to_the_yaml_file(tmp_path):
    y = _yaml(tmp_path, "../in", "../out")
    cfg = _cfg(build_parser().parse_args(["--config", str(y), "infer"]))
    assert cfg.data_dir == (tmp_path / "in").resolve() and cfg.out_dir == (tmp_path / "out").resolve()


def test_precedence_yaml_then_set_then_flags(tmp_path, monkeypatch):
    y = _yaml(tmp_path, "../in", "../out")
    monkeypatch.chdir(tmp_path)
    # --set beats the YAML (relative to the working directory) ...
    cfg = _cfg(build_parser().parse_args(["infer", "--config", str(y), "--set", "paths.data_dir=set_in"]))
    assert cfg.data_dir == (tmp_path / "set_in").resolve() and cfg.out_dir == (tmp_path / "out").resolve()
    # ... and --input/--output beat both; --config may come after the subcommand.
    cfg = _cfg(build_parser().parse_args(["infer", "--config", str(y), "--set", "paths.data_dir=set_in",
                                          "-i", "flag_in", "-o", "~/flag_out"]))
    assert cfg.data_dir == (tmp_path / "flag_in").resolve()
    assert cfg.out_dir == Path(os.path.expanduser("~/flag_out"))


def test_top_level_set_and_detector_flags(tmp_path):
    cfg = _cfg(build_parser().parse_args(["infer", "--set", "device=cpu", "--k-tau", "4"]))
    assert cfg.device == "cpu" and cfg.detector.k_tau == 4.0
    with pytest.raises(ValueError):
        _cfg(build_parser().parse_args(["infer", "--set", "nonsense=1"]))


def test_output_inside_input_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must not be inside"):
        _cfg(build_parser().parse_args(["infer", "-i", str(tmp_path), "-o", str(tmp_path / "out")]))


def test_infer_reports_missing_or_empty_input_without_loading_models(tmp_path, capsys):
    assert main(["infer", "-i", str(tmp_path / "nope"), "-o", str(tmp_path / "out")]) == 2
    assert "input folder not found" in capsys.readouterr().err
    (tmp_path / "empty").mkdir()
    assert main(["infer", "-i", str(tmp_path / "empty"), "-o", str(tmp_path / "out")]) == 2
    assert "no files match" in capsys.readouterr().err
