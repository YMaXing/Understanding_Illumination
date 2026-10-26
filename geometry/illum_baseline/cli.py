"""Command line: python -m illum_baseline.cli {audit|stage-a|stage-b|viz|qa}."""

from __future__ import annotations

import argparse
import dataclasses
import sys

from . import DEFAULT_CONFIG
from .config import DetectorCfg, SkyCfg, load_config, validate

# Detector/sky overrides: --<field> for detector fields, --sky-<field> for sky fields.
OVERRIDES = [("detector", f.name, f"--{f.name.replace('_', '-')}") for f in dataclasses.fields(DetectorCfg)] + [
    ("sky", f.name, f"--sky-{f.name.replace('_', '-')}") for f in dataclasses.fields(SkyCfg)]


def _num_or_null(s: str) -> float | None:
    return None if s.lower() in ("null", "none") else float(s)


def _add_overrides(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("detector/sky overrides (default: config value)")
    for section, name, flag in OVERRIDES:
        default = getattr({"detector": DetectorCfg, "sky": SkyCfg}[section](), name)
        typ = int if isinstance(default, int) else _num_or_null
        g.add_argument(flag, dest=f"ov_{section}_{name}", type=typ, default=argparse.SUPPRESS)


def _apply_overrides(cfg, args: argparse.Namespace):
    for section, name, _ in OVERRIDES:
        key = f"ov_{section}_{name}"
        if hasattr(args, key):
            setattr(getattr(cfg, section), name, getattr(args, key))
    validate(cfg)
    return cfg


def cmd_audit(args: argparse.Namespace) -> int:
    from .audit import run_audit

    cfg = load_config(args.config)
    out = run_audit(cfg)
    print(f"audit written to {out}")
    return 0


def cmd_stage_a(args: argparse.Namespace) -> int:
    import json

    from .stage_a import run_stage_a

    cfg = load_config(args.config)
    limit = args.limit if args.limit is not None else cfg.run.limit
    status = run_stage_a(cfg, force=args.force or cfg.run.force, limit=limit)
    print(json.dumps({k: v for k, v in status.items() if k != "focal"}, indent=2))
    print("focal groups:", json.dumps(status["focal"]["groups"], indent=2))
    return 0


def cmd_stage_b(args: argparse.Namespace) -> int:
    import json

    from .stage_b import run_stage_b

    cfg = _apply_overrides(load_config(args.config), args)
    if args.run_name:
        cfg.run.name = args.run_name
    limit = args.limit if args.limit is not None else cfg.run.limit
    status = run_stage_b(cfg, force=args.force or cfg.run.force, sample=args.sample, limit=limit,
                         viz=not args.no_viz)
    print(json.dumps(status, indent=2))
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    from .stage_b import image_stems, run_dir
    from .viz import render

    cfg = load_config(args.config)
    rd = run_dir(cfg, args.run_name)
    stems = args.stems or image_stems(rd)
    for stem in stems:
        print(render(cfg, rd, stem))
    return 0


def cmd_qa(args: argparse.Namespace) -> int:
    from .qa import run_qa

    cfg = load_config(args.config)
    print(run_qa(cfg, args.run_name))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="illum_baseline")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config (default: %(default)s)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit", help="dataset audit (M1)").set_defaults(func=cmd_audit)
    sa = sub.add_parser("stage-a", help="preprocess + GeoCalib + DA3, cached (M2)")
    sa.add_argument("--force", action="store_true", help="recompute images whose cache exists")
    sa.add_argument("--limit", type=int, default=None, help="process only the first N files (sorted)")
    sa.set_defaults(func=cmd_stage_a)
    sb = sub.add_parser("stage-b", help="detector + deliverables + manifest + viz (M3/M4)")
    sb.add_argument("--run-name", default=None, help="output subdirectory (default: run.name)")
    sb.add_argument("--force", action="store_true", help="recompute existing outputs / overwrite run params")
    sb.add_argument("--sample", type=int, default=None, help="seeded random subset of N cached images")
    sb.add_argument("--limit", type=int, default=None, help="first N cached images (sorted)")
    sb.add_argument("--no-viz", action="store_true", help="skip panel rendering")
    _add_overrides(sb)
    sb.set_defaults(func=cmd_stage_b)
    vz = sub.add_parser("viz", help="re-render panels from saved Stage B outputs only")
    vz.add_argument("--run-name", default=None)
    vz.add_argument("stems", nargs="*", help="image stems (default: all in the run)")
    vz.set_defaults(func=cmd_viz)
    qa = sub.add_parser("qa", help="dataset QA report for a Stage B run (M5)")
    qa.add_argument("--run-name", default=None)
    qa.set_defaults(func=cmd_qa)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
