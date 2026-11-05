"""Gap-class probe (report only): is there more DA3 signal in the missed-geometry units at 1008 than at 504?

Units (tri-only audit, scripts/geom_g4.py units.csv): the 18 gap_unresolved units; the 25 open step gaps that stop at
the component stage (5 are also gap_unresolved); the narrow-face units 15, 37, 92; controls: the 11 tri_fp units and
the texture ROIs (070 floor, 079 grass, 079 grass sunlit, 083 bollard interior).
Per unit, at 504 (Stage A cache, current parameters) and at 1008 (stage_a/res1008, scripts/probe_gaps_stage_a.py; the
pixel parameters of the measurement x2: Hessian sigma, lobe lengths, jump-fit skip / window / half-width, structure-
tensor scales, rotation windows, link radius; no channel run, no recalibration):
  depth r      |Delta rho* / rho_-| of the depth channel's jump fit (two_sided_fit, gap at x0, Omega samples, no
               contamination check; k_min 11 / 22, m 8 / 16, w_t 4 / 8) at every unit pixel along the depth field's
               normal n;
  rotation     angle between the mean dense normals on s in [2, 6] and [-6, -2] px@504 along the normal structure
               tensor's e1 (windows x2 at 1008);
  noise floor  the same measure at control points shifted onto the adjacent planes (x0 +- D along the measure's own
               direction; D = k_min + m + 2 for r, 8 px@504 for the rotation), both sides;
  S / N        P95 over the unit / P95 over its control points (>= 3: a signal the noise floor does not explain).
Thin structures (component-stage units): depth NMS candidates in the unit (dilated 2 px@504) with r > tau_high = 0.04,
  linked within the link radius (3 px@504); piece lengths in distinct pixels (px@504 = px / 2 at 1008).
Unresolved units: depth candidates with r S/N >= 3 and fold NMS candidates with rotation S/N >= 3 inside the unit.
Missed-geometry estimate: geom_v1_1's core relaxed share per unit with the units that reach S/N >= 3 at 1008 (and the
  stricter set that is below 3 at 504) set to fully relaxed, through scripts/geom_score.py's audit estimator.
Writes baseline_output/diag/geom/gaps1008/gaps.json.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from depth_rois import BOXES, box_mask  # noqa: E402
from fold_null_rois import rois as tex_rois  # noqa: E402
from geom_score import Run, audit, load_labels, unit_fractions  # noqa: E402
from probe_gaps_stage_a import BATCH_ROOT, VARIANT  # noqa: E402
from probe_res1008 import two_sided_rotation  # noqa: E402
from illum_baseline.config import dense_fit_params, detector_params, load_config  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.candidates import depth_candidates, depth_field, fold_candidates, normal_structure_tensor  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.depth import DepthParams, fit_kwargs, relative_jump  # noqa: E402
from illum_baseline.geom.sampling import bilinear  # noqa: E402
from illum_baseline.geom.twosided import two_sided_fit  # noqa: E402

OUT = ROOT / "baseline_output" / "diag" / "geom" / "gaps1008"
MAX_SAMPLES = 4000
TAU_HIGH = 0.04


def depth_params(cfg, c):
    p = DepthParams.from_dict(dataclasses.asdict(cfg.geometry.depth))
    if c == 1:
        return p
    return dataclasses.replace(p, sigma=p.sigma * c, sigma_w_prior=p.sigma_w_prior * c, lobe_half_len=p.lobe_half_len * c,
                               lobe_step=p.lobe_step * c, sigma_w_max=p.sigma_w_max * c, k_min=p.k_min * c, m=p.m * c,
                               w_t=p.w_t * c, m_min=p.m_min * c, link_radius=p.link_radius * c,
                               min_component_px=p.min_component_px * c, null_stride=p.null_stride * c)


class Image:
    def __init__(self, stem, batch, res):
        self.c = c = res // 504
        root = BATCH_ROOT[batch]
        sa = root / "stage_a" if res == 504 else root / "stage_a" / VARIANT
        meta = json.loads((sa / f"{stem}.json").read_text())
        z = np.load(sa / f"{stem}.npz")
        cfg = load_config(root / "stage_b" / "geom_v1_1" / "config_resolved.yaml")
        sky = z["sky_prob"].astype(np.float64) if "sky_prob" in z else None
        self.ctx = GeometryContext(z["D"], sky, Intrinsics.from_stage_a(meta), detector_params(cfg), dense_fit_params(cfg))
        self.p = depth_params(cfg, c)
        self.f = depth_field(self.ctx, self.p)
        fp = cfg.geometry.fold
        _, self.e1, self.e1_valid = normal_structure_tensor(self.ctx, fp.cand_sigma_d * c, fp.cand_sigma_T * c)
        self.nhat = self.ctx.n_hat.astype(np.float64)
        self.nval = self.ctx.normal_valid.astype(bool)
        self.H, self.W = self.ctx.shape
        self.kw = fit_kwargs(self.p, self.ctx)
        self._dc = None
        self._fc = None

    def mask(self, m504):
        return m504 if self.c == 1 else np.kron(m504.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)

    def r_at(self, x0, n):
        res = two_sided_fit(self.ctx, x0, n, self.p.k_min, rms_max=None, **self.kw)
        return np.abs(relative_jump(res))

    def rot_at(self, x0, e1):
        return two_sided_rotation(self.nhat, self.nval, x0, e1, self.c)

    def inside(self, x):
        return (x[:, 0] >= 0) & (x[:, 0] <= self.W - 1) & (x[:, 1] >= 0) & (x[:, 1] <= self.H - 1)

    def measures(self, mk, rng):
        v, u = np.nonzero(mk)
        if len(u) > MAX_SAMPLES:
            sel = rng.choice(len(u), MAX_SAMPLES, replace=False)
            u, v = u[sel], v[sel]
        X = np.stack([u, v], -1).astype(np.float64)
        n, e1 = self.f.n[v, u], self.e1[v, u]
        okn = np.isfinite(n).all(1) & (np.linalg.norm(np.nan_to_num(n), axis=1) > 0)
        oke = np.isfinite(e1).all(1) & self.e1_valid[v, u]
        r = self.r_at(X[okn], n[okn]) if okn.any() else np.array([np.nan])
        rot = self.rot_at(X[oke], e1[oke]) if oke.any() else np.array([np.nan])
        Dr, Dq = float(self.p.k_min + self.p.m + 2), 8.0 * self.c
        rc, qc = [np.array([np.nan])], [np.array([np.nan])]
        for s in (1, -1):
            xr = X[okn] + s * Dr * n[okn]
            ok = self.inside(xr)
            if ok.any():
                rc.append(self.r_at(xr[ok], n[okn][ok]))
            xq = X[oke] + s * Dq * e1[oke]
            ok = self.inside(xq)
            if ok.any():
                qc.append(self.rot_at(xq[ok], e1[oke][ok]))
        rc, qc = np.concatenate(rc), np.concatenate(qc)
        p95 = lambda a: float(np.nanpercentile(a, 95)) if np.isfinite(a).any() else float("nan")  # noqa: E731
        S_r, N_r, S_q, N_q = p95(r), p95(rc), p95(rot), p95(qc)
        return {"S_r": S_r, "N_r": N_r, "SN_r": S_r / N_r if N_r > 0 else float("nan"),
                "S_rot": S_q, "N_rot": N_q, "SN_rot": S_q / N_q if N_q > 0 else float("nan"), "n_samples": len(u)}

    def depth_cands(self):
        if self._dc is None:
            c = depth_candidates(self.ctx, self.p, field=self.f)
            ok = np.isfinite(c.n).all(1) & np.isfinite(c.x0).all(1)
            self._dc = (c.x0[ok], c.n[ok], self.r_at(c.x0[ok], c.n[ok]))
        return self._dc

    def fold_cands(self):
        if self._fc is None:
            fp = FOLD
            cc = fold_candidates(self.ctx, fp["sd"] * self.c, fp["sT"] * self.c, fp["rate"])
            ok = np.isfinite(cc.e1).all(1) & np.isfinite(cc.x0).all(1)
            self._fc = (cc.x0[ok], cc.e1[ok], self.rot_at(cc.x0[ok], cc.e1[ok]))
        return self._fc

    def in_mask(self, x, mk, dil):
        m2 = ndimage.binary_dilation(mk, iterations=dil) if dil else mk
        ui = np.clip(np.rint(x[:, 0]).astype(int), 0, self.W - 1)
        vi = np.clip(np.rint(x[:, 1]).astype(int), 0, self.H - 1)
        return m2[vi, ui]

    def thin(self, mk):
        x0, n, r = self.depth_cands()
        sel = self.in_mask(x0, mk, 2 * self.c) & (r > TAU_HIGH)
        xs = x0[sel]
        if not len(xs):
            return {"n_pieces": 0, "longest_px@504": 0.0, "total_px@504": 0.0}
        pairs = cKDTree(xs).query_pairs(self.p.link_radius, output_type="ndarray")
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xs), len(xs)))
        _, lab = connected_components(g, directed=False)
        pix = np.unique(np.stack([lab, np.rint(xs[:, 1]).astype(int) * self.W + np.rint(xs[:, 0]).astype(int)], -1), axis=0)
        sizes = np.bincount(pix[:, 0]) / self.c
        return {"n_pieces": int(len(sizes)), "longest_px@504": float(sizes.max()), "total_px@504": float(sizes.sum()),
                "pieces_ge_5px@504": int((sizes >= 5).sum())}

    def candidates_sn(self, mk, N_r, N_q):
        x0, _, r = self.depth_cands()
        a = self.in_mask(x0, mk, self.c)
        xf, _, q = self.fold_cands()
        b = self.in_mask(xf, mk, self.c)
        return {"depth_cands": int(a.sum()), "depth_sn3": int((r[a] >= 3 * N_r).sum()) if np.isfinite(N_r) else 0,
                "fold_cands": int(b.sum()), "fold_sn3": int((q[b] >= 3 * N_q).sum()) if np.isfinite(N_q) else 0}


FOLD = {}


def texture_masks(stem):
    if stem == "taiwei_cui_079":
        return tex_rois(stem, (504, 378))
    if stem == "taiwei_cui_083":
        return {"083 bollard interior": box_mask((504, 378), BOXES["083 bollard interior"][2])}
    if stem == "taiwei_cui_070":
        return tex_rois(stem, (504, 378))
    return {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab, draws, strata, totals, masks = load_labels()
    u = pd.read_csv(ROOT / "baseline_output" / "diag" / "geom" / "g4" / "units.csv")
    t = u[u.direction == "tri-only"]
    classes = {"gap_unresolved": set(t.id[t.bucket == "gap_unresolved"]), "component_step": set(t.id[t.stage == "component"]),
               "narrow_face": {15, 37, 92}, "tri_fp": set(t.id[t.bucket == "tri_fp"])}
    ids = set().union(*classes.values())
    info = u.set_index("id")
    cfg0 = load_config(ROOT / "baseline_output" / "stage_b" / "geom_v1_1" / "config_resolved.yaml")
    FOLD.update(sd=cfg0.geometry.fold.cand_sigma_d, sT=cfg0.geometry.fold.cand_sigma_T, rate=cfg0.geometry.fold.cand_rate_factor)
    stems = sorted({info.loc[i, "stem"] for i in ids} | {"taiwei_cui_070", "taiwei_cui_079", "taiwei_cui_083"})
    batch_of = dict(zip(u.stem, u.batch))
    rows, tex = [], []
    for stem in stems:
        for res in (504, 1008):
            im = Image(stem, batch_of[stem], res)
            rng = np.random.default_rng(0)
            for i in sorted(i for i in ids if info.loc[i, "stem"] == stem):
                m504 = np.zeros(504 * 378, bool)
                m504[masks[f"id{i:03d}"]] = True
                m504 = m504.reshape(im.H // im.c, im.W // im.c)
                mk = im.mask(m504)
                r = {"id": i, "stem": stem, "res": res, "area_px@504": int(m504.sum()),
                     "classes": [k for k, v in classes.items() if i in v], **im.measures(mk, rng)}
                if i in classes["component_step"]:
                    r |= {f"thin_{k}": v for k, v in im.thin(mk).items()}
                if i in classes["gap_unresolved"] or i in classes["narrow_face"]:
                    r |= im.candidates_sn(mk, r["N_r"], r["N_rot"])
                rows.append(r)
            for name, m504 in texture_masks(stem).items():
                tex.append({"roi": name, "stem": stem, "res": res, **im.measures(im.mask(m504), rng)})
            print(stem, res, "done", flush=True)
            del im
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "units.csv", index=False)
    pd.DataFrame(tex).to_csv(OUT / "texture.csv", index=False)

    # per class summary
    w = df.pivot_table(index="id", columns="res", values=["SN_r", "SN_rot"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w["SN_max_504"] = w[["SN_r_504", "SN_rot_504"]].max(axis=1)
    w["SN_max_1008"] = w[["SN_r_1008", "SN_rot_1008"]].max(axis=1)
    summ = {}
    for k, s in classes.items():
        g = w.loc[sorted(s)]
        summ[k] = {"n": len(g), **{f"share_SN{m}>=3_{res}": float((g[f"SN_{m}_{res}"] >= 3).mean())
                                   for m in ("r", "rot", "max") for res in (504, 1008)},
                   **{f"median_SN{m}_{res}": float(g[f"SN_{m}_{res}"].median()) for m in ("r", "rot", "max")
                      for res in (504, 1008)}}
    tx = pd.DataFrame(tex)
    summ["texture_rois"] = {f"{r.roi} {r.res}": {"SN_r": r.SN_r, "SN_rot": r.SN_rot} for r in tx.itertuples()}
    thin = df[df.id.isin(classes["component_step"])].pivot_table(index="id", columns="res",
                                                                 values=["thin_longest_px@504", "thin_n_pieces", "thin_pieces_ge_5px@504"])
    summ["thin"] = {f"{a} {b}": float(thin[(a, b)].median()) for a, b in thin.columns}
    summ["thin_units_with_piece_ge_5px@504"] = {res: int((thin[("thin_pieces_ge_5px@504", res)] > 0).sum()) for res in (504, 1008)}
    unres = df[df.id.isin(classes["gap_unresolved"] | classes["narrow_face"])].pivot_table(
        index="id", columns="res", values=["depth_sn3", "fold_sn3"])
    appear = [i for i in unres.index if (unres.loc[i, ("depth_sn3", 504)] + unres.loc[i, ("fold_sn3", 504)] == 0)
              and (unres.loc[i, ("depth_sn3", 1008)] + unres.loc[i, ("fold_sn3", 1008)] > 0)]
    summ["unresolved_candidates"] = {"units_with_sn3_candidate_504": int(((unres[("depth_sn3", 504)] + unres[("fold_sn3", 504)]) > 0).sum()),
                                     "units_with_sn3_candidate_1008": int(((unres[("depth_sn3", 1008)] + unres[("fold_sn3", 1008)]) > 0).sum()),
                                     "appear_at_1008_only": appear}

    # missed-geometry estimate (core w, tri-only audit)
    run = Run("geom_v1_1")
    U = unit_fractions(run, lab, masks)
    base = audit(run, U, draws, strata, totals)["core"]["tri-only"]["missed true geometry | per pixel"]
    gap_ids = set(t.id[t.bucket.isin(["gap_fixable", "gap_unresolved"])])
    est = {"core_missed_per_px_geom_v1_1": base}
    for name, cond in (("SN>=3 at 1008", w.SN_max_1008 >= 3), ("SN<3 at 504 and >=3 at 1008", (w.SN_max_504 < 3) & (w.SN_max_1008 >= 3))):
        flip = sorted(set(w.index[cond]) & gap_ids)
        Um = U.copy()
        Um.loc[Um.id.isin(flip) & (Um.direction == "tri-only"), "frac_core"] = 1.0
        new = audit(run, Um, draws, strata, totals)["core"]["tri-only"]["missed true geometry | per pixel"]
        est[name] = {"units_flipped": flip, "n": len(flip), "core_missed_per_px": new,
                     "reduction": base["share"] - new["share"]}
    allgap = sorted(set(w.index) & gap_ids)
    Um = U.copy()
    Um.loc[Um.id.isin(allgap) & (Um.direction == "tri-only"), "frac_core"] = 1.0
    est["ceiling: every probed gap unit relaxed"] = {"n": len(allgap), "reduction": base["share"] -
                                                     audit(run, Um, draws, strata, totals)["core"]["tri-only"]["missed true geometry | per pixel"]["share"]}
    summ["missed_estimate"] = est
    (OUT / "gaps.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
