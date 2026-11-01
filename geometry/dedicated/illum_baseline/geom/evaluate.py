"""Fold evaluation hook (TASK_FOLD_CHANNEL F4): precision and recall of accepted fold candidates against annotated
polylines.

Annotation file: annotations/folds/<stem>.json (project root), upright network-resolution pixel coordinates:

    {"tolerance_px": 2.0,
     "polylines": [{"label": "fold", "points": [[u, v], [u, v], ...]},
                   {"label": "not_fold", "points": [[u, v], ...]}, ...]}

- recall: share of the fold polylines' length (sampled every 0.5 px) within tolerance of an accepted candidate;
- precision: TP / (TP + FP), TP = accepted candidates within tolerance of a fold polyline, FP = accepted candidates
  within tolerance of a not_fold polyline and of no fold polyline (unannotated areas count as neither);
- precision_all: TP / all accepted candidates (treats every unannotated acceptance as a false positive).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

LABELS = ("fold", "not_fold")


def load_annotations(path: Path) -> dict:
    a = json.loads(Path(path).read_text())
    tol = float(a.get("tolerance_px", 2.0))
    lines = a.get("polylines", [])
    for pl in lines:
        if pl.get("label") not in LABELS:
            raise ValueError(f"{path}: polyline label must be one of {LABELS}, got {pl.get('label')!r}")
        if len(pl.get("points", [])) < 2:
            raise ValueError(f"{path}: a polyline needs at least 2 points")
    return {"tolerance_px": tol, "polylines": lines}


def _sample(points: list, step: float = 0.5) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        out.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.concatenate(out)


def evaluate_folds(candidates: pd.DataFrame, ann: dict) -> dict:
    tol = ann["tolerance_px"]
    acc = candidates.loc[candidates["accepted"], ["u0", "v0"]].to_numpy(dtype=np.float64)
    samples = {lab: [_sample(pl["points"]) for pl in ann["polylines"] if pl["label"] == lab] for lab in LABELS}
    S = {lab: (np.concatenate(v) if v else np.zeros((0, 2))) for lab, v in samples.items()}

    def near(pts: np.ndarray, ref: np.ndarray) -> np.ndarray:
        if not len(pts) or not len(ref):
            return np.zeros(len(pts), dtype=bool)
        return cKDTree(ref).query(pts, distance_upper_bound=tol + 1e-9)[0] <= tol

    tp = near(acc, S["fold"])
    fp = near(acc, S["not_fold"]) & ~tp
    covered = near(S["fold"], acc)
    n_tp, n_fp = int(tp.sum()), int(fp.sum())
    return {"n_accepted": int(len(acc)), "tp": n_tp, "fp": n_fp, "tolerance_px": tol,
            "precision": n_tp / (n_tp + n_fp) if n_tp + n_fp else float("nan"),
            "precision_all": n_tp / len(acc) if len(acc) else float("nan"),
            "recall": float(covered.mean()) if len(covered) else float("nan"),
            "fold_length_px": float(0.5 * len(S["fold"])), "n_polylines": len(ann["polylines"])}


def evaluate_run(rd: Path, ann_dir: Path) -> pd.DataFrame:
    """One row per image of run directory rd that has both a fold candidate table and an annotation file."""
    rows = []
    for f in sorted(rd.glob("*_fold_candidates.parquet")):
        stem = f.name[: -len("_fold_candidates.parquet")]
        a = ann_dir / f"{stem}.json"
        if a.exists():
            rows.append({"stem": stem, **evaluate_folds(pd.read_parquet(f), load_annotations(a))})
    return pd.DataFrame(rows)
