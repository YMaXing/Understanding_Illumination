"""TASK_GEOM_V2 V5 diagnostic: why the fallback loses fold-candidate RGB-edge coverage (report only; reads RGB).

At d_excl 2, pool_radius 2, band_min 2 (geom_v1_hessian's saved channel weights): the reliable RGB-edge samples of
geom_v1's accepted fold candidates (as in scripts/geom_score.py) covered by min(depth, fold, laplacian) but not by
min(depth, fold, fallback(laplacian)); each lost sample is attributed to the nearest accepted fold candidate (at its
V2b site, within 3 px): its own w_c >= 0.5 (a weak fold that relaxes nothing itself) or < 0.5 (its band misses the
RGB edge); farther than 3 px: a depth edge or none. Run from geometry/dedicated.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path('.').resolve(); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from rgb_edges import luminance, rgb_offset_strength, gradient_median
from scipy.spatial import cKDTree
from illum_baseline.config import load_config
from illum_baseline.stage_b import load_upright
from illum_baseline.geom.fallback import exclusion_zone, apply_fallback
B = {"batch1": ROOT/"baseline_output", "batch2": ROOT/"baseline_output_batch2"}
tot = {"reliable": 0, "lost": 0, "lost_weak_fold": 0, "lost_weak_depth_or_none": 0, "lost_outside_band": 0}
for batch, root in B.items():
    cfg = load_config(root/"stage_b/geom_v1_hessian/config_resolved.yaml")
    for p in sorted((root/"stage_b/geom_v1_hessian").glob("*_fold_candidates.parquet")):
        stem = p.name.replace("_fold_candidates.parquet", "")
        a, _ = load_upright(cfg, root/"stage_b/geom_v1_hessian", stem)
        fc = pd.read_parquet(p); dc = pd.read_parquet(root/f"stage_b/geom_v1_hessian/{stem}_depth_candidates.parquet")
        ref = pd.read_parquet(root/f"stage_b/geom_v1/{stem}_fold_candidates.parquet"); ref = ref[ref.accepted]
        Y = luminance(root/"stage_a", stem)
        x, n = ref[["u0","v0"]].to_numpy(), ref[["e1_u","e1_v"]].to_numpy()
        s, g = rgb_offset_strength(Y, x, n); rel = np.isfinite(s) & (np.abs(s) < 7.99) & (g >= 3*gradient_median(Y))
        pts = x[rel] + s[rel, None]*n[rel]; H, W = a["w_net"].shape
        iu, iv = np.clip(np.rint(pts[:,0]).astype(int),0,W-1), np.clip(np.rint(pts[:,1]).astype(int),0,H-1)
        wd, wf, wl = (a[k].astype(float) for k in ("depth_w_net","fold_w_net","laplacian_w_net"))
        z = exclusion_zone(dc, fc, (H, W), 2)
        old = np.minimum.reduce([wd, wf, wl])[iv, iu] < 0.5
        new = np.minimum.reduce([wd, wf, apply_fallback(wl, z)])[iv, iu] < 0.5
        lost = old & ~new
        tot["reliable"] += int(rel.sum()); tot["lost"] += int(lost.sum())
        fa = fc[fc.accepted]; t = cKDTree(fa[["u_site","v_site"]].to_numpy())
        d, j = t.query(pts[lost]); wc = fa["w_c"].to_numpy()[j]
        weak = (d <= 3) & (wc >= 0.5)
        tot["lost_weak_fold"] += int(weak.sum())
        tot["lost_outside_band"] += int(((d <= 3) & (wc < 0.5)).sum())
        tot["lost_weak_depth_or_none"] += int((d > 3).sum())
print(tot, {k: round(v/tot["lost"],3) for k,v in tot.items() if k.startswith("lost_")})
