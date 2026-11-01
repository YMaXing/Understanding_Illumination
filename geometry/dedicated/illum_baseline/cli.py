"""Command line: python -m illum_baseline [--config CFG] {infer|audit|stage-a|stage-b|viz|qa|diag|...}.

End-to-end inference on a folder of TIFFs:

    python -m illum_baseline infer --input /path/to/tiffs --output /path/to/outputs

Paths come from the YAML (paths.data_dir / paths.out_dir; relative values are relative to the YAML file) and can
be overridden with --set paths.data_dir=... or, with the highest precedence, --input/-i and --output/-o (relative
values are relative to the working directory). Every subcommand accepts --config, --set, --input and --output.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

from . import DEFAULT_CONFIG
from .config import Config, DetectorCfg, SkyCfg, apply_paths, apply_sets, load_config, validate

# Detector/sky overrides: --<field> for detector fields, --sky-<field> for sky fields.
OVERRIDES = [("detector", f.name, f"--{f.name.replace('_', '-')}") for f in dataclasses.fields(DetectorCfg)] + [
    ("sky", f.name, f"--sky-{f.name.replace('_', '-')}") for f in dataclasses.fields(SkyCfg)]


def _num_or_null(s: str) -> float | None:
    return None if s.lower() in ("null", "none") else float(s)


def _bool(s: str) -> bool:
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def _add_overrides(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("detector/sky overrides (default: config value)")
    for section, name, flag in OVERRIDES:
        default = getattr({"detector": DetectorCfg, "sky": SkyCfg}[section](), name)
        typ = (_bool if isinstance(default, bool) else int if isinstance(default, int)
               else str if isinstance(default, str) else _num_or_null)
        g.add_argument(flag, dest=f"ov_{section}_{name}", type=typ, default=argparse.SUPPRESS)


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    for section, name, _ in OVERRIDES:
        key = f"ov_{section}_{name}"
        if hasattr(args, key):
            setattr(getattr(cfg, section), name, getattr(args, key))
    validate(cfg)
    return cfg


def _cfg(args: argparse.Namespace) -> Config:
    """YAML (--config) < --set < detector/sky flags < --input/--output."""
    cfg = apply_sets(load_config(args.config), args.set)
    cfg = _apply_overrides(cfg, args)
    return apply_paths(cfg, args.input, args.output)


def _quiet_da3() -> None:
    """DA3 prints per-call [INFO] lines to stdout; keep warnings only unless the user set DA3_LOG_LEVEL."""
    os.environ.setdefault("DA3_LOG_LEVEL", "WARN")


def cmd_infer(args: argparse.Namespace) -> int:
    """Stage A + Stage B (+ optional QA) for every TIFF under the input folder."""
    _quiet_da3()
    from .audit import list_files
    from .stage_a import run_stage_a
    from .stage_b import run_dir, run_stage_b

    cfg = _cfg(args)
    if args.run_name:
        cfg.run.name = args.run_name
    if cfg.paths.data_dir is None or not cfg.data_dir.is_dir():
        print(f"error: input folder not found: {cfg.paths.data_dir} (set paths.data_dir or pass --input)",
              file=sys.stderr)
        return 2
    if cfg.paths.out_dir is None:
        print("error: no output folder (set paths.out_dir or pass --output)", file=sys.stderr)
        return 2
    try:
        files = list_files(cfg)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    stems = [f.stem for f in files]
    if args.stems:
        missing = sorted(set(args.stems) - set(stems))
        if missing:
            print(f"error: stems not found under {cfg.data_dir}: {missing}", file=sys.stderr)
            return 2
        stems = sorted(args.stems)
    force = args.force or cfg.run.force
    print(f"[infer] {len(stems)} image(s) from {cfg.data_dir} -> {cfg.out_dir} (run {cfg.run.name!r})",
          file=sys.stderr)

    status_a = run_stage_a(cfg, force=force, stems=stems)
    status_b = run_stage_b(cfg, force=force, viz=not args.no_viz, stems=stems)
    rd = run_dir(cfg)
    summary = {
        "input": str(cfg.data_dir),
        "output": str(cfg.out_dir),
        "run": cfg.run.name,
        "n_images": len(stems),
        "stage_a": {k: status_a[k] for k in ("n_done", "n_skipped")},
        "stage_b": {k: status_b[k] for k in ("n_done", "n_skipped")},
        "weights": str(rd / "<stem>.npz") + "  (key 'w': float16, stored frame, original resolution)",
        "manifest": status_b["manifest"],
        "viz": None if args.no_viz else str(rd / "viz"),
    }
    if args.qa:
        from .qa import run_qa

        summary["qa_report"] = str(run_qa(cfg))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from .audit import run_audit

    out = run_audit(_cfg(args))
    print(f"audit written to {out}")
    return 0


def cmd_stage_a(args: argparse.Namespace) -> int:
    _quiet_da3()
    from .stage_a import run_stage_a

    cfg = _cfg(args)
    limit = args.limit if args.limit is not None else cfg.run.limit
    status = run_stage_a(cfg, force=args.force or cfg.run.force, limit=limit, stems=args.stems)
    print(json.dumps({k: v for k, v in status.items() if k != "focal"}, indent=2))
    print("focal groups:", json.dumps(status["focal"]["groups"], indent=2))
    return 0


def cmd_stage_b(args: argparse.Namespace) -> int:
    from .stage_b import run_stage_b

    cfg = _cfg(args)
    if args.run_name:
        cfg.run.name = args.run_name
    limit = args.limit if args.limit is not None else cfg.run.limit
    status = run_stage_b(cfg, force=args.force or cfg.run.force, sample=args.sample, limit=limit,
                         viz=not args.no_viz, stems=args.stems)
    print(json.dumps(status, indent=2))
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    from .stage_b import image_stems, run_dir
    from .viz import render

    cfg = _cfg(args)
    rd = run_dir(cfg, args.run_name)
    for stem in args.stems or image_stems(rd):
        print(render(cfg, rd, stem))
    return 0


def cmd_qa(args: argparse.Namespace) -> int:
    from .qa import run_qa

    print(run_qa(_cfg(args), args.run_name))
    return 0


def cmd_diag(args: argparse.Namespace) -> int:
    from .diag import run_diag

    print(run_diag(_cfg(args), args.stem, args.run_name, gpu=not args.no_gpu))
    return 0


def cmd_diag_compare(args: argparse.Namespace) -> int:
    from .diag import compare_runs

    print(compare_runs(_cfg(args), args.stem, args.runs, args.baseline_run, args.tag))
    return 0


def cmd_diag_generalize(args: argparse.Namespace) -> int:
    from .diag import generalize

    print(generalize(_cfg(args), args.runs, args.baseline_run, args.tag))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m illum_baseline", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config (default: %(default)s)")

    # Shared by every subcommand. --config may also be given after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("config and paths (precedence: YAML < --set < --input/--output)")
    g.add_argument("--config", default=argparse.SUPPRESS, help="YAML config (same as the global --config)")
    g.add_argument("--input", "-i", default=None, metavar="DIR",
                   help="input folder of TIFFs (overrides paths.data_dir; searched recursively)")
    g.add_argument("--output", "-o", default=None, metavar="DIR",
                   help="output folder (overrides paths.out_dir; must not be inside the input folder)")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, e.g. --set device=cpu or --set detector.k_tau=4 (repeatable)")

    sub = ap.add_subparsers(dest="cmd", required=True)

    inf = sub.add_parser("infer", parents=[common], help="end-to-end: Stage A + Stage B (+ QA) on a folder")
    inf.add_argument("--stems", nargs="+", default=None, help="process only these image stems")
    inf.add_argument("--force", action="store_true", help="recompute images whose outputs exist")
    inf.add_argument("--run-name", default=None, help="Stage B output subdirectory (default: run.name)")
    inf.add_argument("--no-viz", action="store_true", help="skip the diagnostic panels")
    inf.add_argument("--qa", action="store_true", help="also write the dataset QA report")
    _add_overrides(inf)
    inf.set_defaults(func=cmd_infer)

    sub.add_parser("audit", parents=[common], help="dataset audit (M1)").set_defaults(func=cmd_audit)

    sa = sub.add_parser("stage-a", parents=[common], help="preprocess + GeoCalib + DA3, cached (M2)")
    sa.add_argument("--force", action="store_true", help="recompute images whose cache exists")
    sa.add_argument("--limit", type=int, default=None, help="process only the first N files (sorted)")
    sa.add_argument("--stems", nargs="+", default=None, help="process only these image stems")
    sa.set_defaults(func=cmd_stage_a)

    sb = sub.add_parser("stage-b", parents=[common], help="detector + deliverables + manifest + viz (M3/M4)")
    sb.add_argument("--run-name", default=None, help="output subdirectory (default: run.name)")
    sb.add_argument("--force", action="store_true", help="recompute existing outputs / overwrite run params")
    sb.add_argument("--sample", type=int, default=None, help="seeded random subset of N cached images")
    sb.add_argument("--limit", type=int, default=None, help="first N cached images (sorted)")
    sb.add_argument("--no-viz", action="store_true", help="skip panel rendering")
    sb.add_argument("--stems", nargs="+", default=None, help="process only these cached stems")
    _add_overrides(sb)
    sb.set_defaults(func=cmd_stage_b)

    vz = sub.add_parser("viz", parents=[common], help="re-render panels from saved Stage B outputs only")
    vz.add_argument("--run-name", default=None)
    vz.add_argument("stems", nargs="*", help="image stems (default: all in the run)")
    vz.set_defaults(func=cmd_viz)

    qa = sub.add_parser("qa", parents=[common], help="dataset QA report for a Stage B run (M5)")
    qa.add_argument("--run-name", default=None)
    qa.set_defaults(func=cmd_qa)

    dg = sub.add_parser("diag", parents=[common], help="stripe diagnostics for one image (TASK_070_STRIPES T1)")
    dg.add_argument("--stem", required=True)
    dg.add_argument("--run-name", required=True, help="Stage B run whose detector params are used")
    dg.add_argument("--no-gpu", action="store_true", help="skip the DA3 shift test")
    dg.set_defaults(func=cmd_diag)

    dc = sub.add_parser("diag-compare", parents=[common],
                        help="floor/edge/global metrics for variant runs (TASK_070_STRIPES)")
    dc.add_argument("--stem", required=True)
    dc.add_argument("--runs", nargs="+", required=True)
    dc.add_argument("--baseline-run", default="baseline_v0")
    dc.add_argument("--tag", required=True, help="output prefix, e.g. t2")
    dc.set_defaults(func=cmd_diag_compare)

    dz = sub.add_parser("diag-generalize", parents=[common],
                        help="per-image deltas vs a baseline run (TASK_070_STRIPES T5)")
    dz.add_argument("--runs", nargs="+", required=True)
    dz.add_argument("--baseline-run", default="baseline_v0")
    dz.add_argument("--tag", required=True)
    dz.set_defaults(func=cmd_diag_generalize)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
