"""Fold evaluation hook (TASK_FOLD_CHANNEL F4, classes TASK_FOLD_V2 F9): accepted fold candidates against annotated
polylines.

Annotation file: annotations/folds/<stem>.json (project root), upright network-resolution pixel coordinates:

    {"tolerance_px": 2.0,
     "polylines": [{"label": "fold", "points": [[u, v], [u, v], ...]},
                   {"label": "step", "points": [[u, v], ...]}, ...]}

Labels: fold (a crease / fold the channel should accept), step (a depth step / occlusion edge), limb (a curved
object's occluding contour), texture (no geometric edge), and not_fold (older files: any non-fold edge).

Each accepted candidate (its x0) is assigned to the class of the nearest labelled polyline within the tolerance (ties:
fold first); candidates near no polyline are unlabelled.
- fold: recall = share of the fold polylines' length (sampled every 0.5 px) within tolerance of an accepted candidate;
  precision = TP / (TP + FP), TP = accepted assigned to fold, FP = accepted assigned to any other class (with fold /
  not_fold labels only this is the original definition); precision_all = TP / all accepted.
- per class c: recall_c = share of class c's polyline length within tolerance of an accepted candidate (for step,
  limb, texture: how often the fold channel fires on them); precision_c = accepted assigned to c / accepted assigned
  to any class (the composition of the labelled acceptances).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

LABELS = ("fold", "step", "limb", "texture", "not_fold")


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
    S = {lab: np.concatenate([_sample(pl["points"]) for pl in ann["polylines"] if pl["label"] == lab])
         for lab in LABELS if any(pl["label"] == lab for pl in ann["polylines"])}

    def dist(pts: np.ndarray, ref: np.ndarray) -> np.ndarray:
        if not len(pts):
            return np.zeros(0)
        return cKDTree(ref).query(pts)[0]

    # nearest labelled class within tolerance (ties: the order of LABELS, fold first)
    present = [lab for lab in LABELS if lab in S]
    if len(acc) and present:
        D = np.stack([dist(acc, S[lab]) for lab in present], 1)
        best = np.argmin(D + 1e-9 * np.arange(len(present))[None], 1)
        within = D[np.arange(len(acc)), best] <= tol
        assigned = np.where(within, np.array(present, dtype=object)[best], None)
    else:
        assigned = np.full(len(acc), None, dtype=object)
    n_lab = int(sum(a is not None for a in assigned))
    per_class = {}
    for lab in present:
        cov = dist(S[lab], acc) <= tol if len(acc) else np.zeros(len(S[lab]), dtype=bool)
        n_c = int(sum(a == lab for a in assigned))
        per_class[lab] = {"n_polylines": int(sum(pl["label"] == lab for pl in ann["polylines"])),
                          "length_px": float(0.5 * len(S[lab])), "recall": float(cov.mean()),
                          "accepted_assigned": n_c, "precision": n_c / n_lab if n_lab else float("nan")}
    n_tp = per_class.get("fold", {}).get("accepted_assigned", 0)
    n_fp = n_lab - n_tp
    return {"n_accepted": int(len(acc)), "tp": n_tp, "fp": n_fp, "tolerance_px": tol,
            "precision": n_tp / (n_tp + n_fp) if n_tp + n_fp else float("nan"),
            "precision_all": n_tp / len(acc) if len(acc) else float("nan"),
            "recall": per_class["fold"]["recall"] if "fold" in per_class else float("nan"),
            "fold_length_px": per_class["fold"]["length_px"] if "fold" in per_class else 0.0,
            "n_polylines": len(ann["polylines"]), "per_class": per_class}


def evaluate_run(rd: Path, ann_dir: Path) -> pd.DataFrame:
    """One row per image of run directory rd that has both a fold candidate table and an annotation file; per-class
    results flattened as <class>_recall, <class>_precision, <class>_accepted."""
    rows = []
    for f in sorted(rd.glob("*_fold_candidates.parquet")):
        stem = f.name[: -len("_fold_candidates.parquet")]
        a = ann_dir / f"{stem}.json"
        if a.exists():
            r = evaluate_folds(pd.read_parquet(f), load_annotations(a))
            pc = r.pop("per_class")
            for lab, v in pc.items():
                r |= {f"{lab}_recall": v["recall"], f"{lab}_precision": v["precision"],
                      f"{lab}_accepted": v["accepted_assigned"]}
            rows.append({"stem": stem, **r})
    return pd.DataFrame(rows)
