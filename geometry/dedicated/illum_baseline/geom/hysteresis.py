"""TASK_GEOM_V2 W2: ridge hysteresis for the Laplacian channel (geometry.laplacian.hysteresis ridge_bridge).

On q' (the scale-ratio-vetoed response before the threshold tau, the detector's q):
  ridge pixels   lambda_1 < 0 and |lambda_1| > |lambda_2| (lambda_1 the most negative eigenvalue of q's Hessian,
                 Gaussian derivatives sigma = 1 px; the second condition keeps the NMS off a ridge's flat direction),
                 q(x) >= q(x +- n) along its eigenvector n (bilinear; non-maximum suppression across the ridge),
                 q > kappa tau;
  linking        8-neighbouring ridge pixels whose tangents agree (|t_i . t_j| >= cos 30 deg);
  seeds          ridge pixels with q > tau (they already relax); seed runs = linked components of seeds;
  continuation   ridge pixels with kappa tau < q <= tau; continuation runs = linked components of them;
  bridge         a continuation run of at most max_gap pixels that is linked to >= 2 distinct seed runs
                 (single_ended, report only: >= 1);
  strength       every pixel of a bridged run takes q_eff = min over the touched seed runs of the median of
                 q - tau over the n_flank seed pixels of that run nearest the contact (seeds and every other pixel
                 keep q_eff = max(q - tau, 0)); pooling and w as the detector (B6, B7).
Pure numpy / scipy, float64, network resolution, upright frame.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .sampling import bilinear

OFFSETS = ((0, 1), (1, -1), (1, 0), (1, 1))   # (dv, du): each 8-neighbour pair once


def ridge_hysteresis(q: np.ndarray, tau: float, kappa: float, max_gap: int = 16, single_ended: bool = False,
                     sigma: float = 1.0, link_deg: float = 30.0, n_flank: int = 3) -> dict:
    q = np.asarray(q, dtype=np.float64)
    H, W = q.shape
    quu = ndimage.gaussian_filter(q, sigma, order=(0, 2))
    qvv = ndimage.gaussian_filter(q, sigma, order=(2, 0))
    quv = ndimage.gaussian_filter(q, sigma, order=(1, 1))
    half = 0.5 * (quu - qvv)
    root = np.hypot(half, quv)
    lam1 = 0.5 * (quu + qvv) - root                     # most negative eigenvalue
    lam2 = 0.5 * (quu + qvv) + root
    theta = 0.5 * np.arctan2(2 * quv, quu - qvv) + 0.5 * np.pi   # its eigenvector angle (u, v)
    nu, nv = np.cos(theta), np.sin(theta)
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    ok = np.ones((H, W), dtype=bool)
    qa, _ = bilinear(q, ok, uu + nu, vv + nv)
    qb, _ = bilinear(q, ok, uu - nu, vv - nv)
    ridge = (lam1 < 0) & (np.abs(lam1) > np.abs(lam2)) & (q >= np.nan_to_num(qa, nan=np.inf)) & (q >= np.nan_to_num(qb, nan=np.inf)) & (q > kappa * tau)
    seed = ridge & (q > tau)
    cont = ridge & ~seed

    idx = -np.ones((H, W), dtype=np.int64)
    rv, ru = np.nonzero(ridge)
    n_r = rv.size
    idx[rv, ru] = np.arange(n_r)
    tu, tv = -nv, nu                                   # tangent
    cos_min = math.cos(math.radians(link_deg))
    src, dst = [], []
    for dv, du in OFFSETS:
        v1, u1 = rv + dv, ru + du
        inside = (v1 >= 0) & (v1 < H) & (u1 >= 0) & (u1 < W)
        i, v1, u1 = np.nonzero(inside)[0], v1[inside], u1[inside]
        j = idx[v1, u1]
        has = j >= 0
        i, j = i[has], j[has]
        agree = np.abs(tu[rv[i], ru[i]] * tu[rv[j], ru[j]] + tv[rv[i], ru[i]] * tv[rv[j], ru[j]]) >= cos_min
        src.append(i[agree])
        dst.append(j[agree])
    src = np.concatenate(src) if src else np.zeros(0, int)
    dst = np.concatenate(dst) if dst else np.zeros(0, int)
    is_seed = seed[rv, ru]

    def comps(mask_nodes: np.ndarray) -> np.ndarray:
        """Component label per ridge node within the subgraph of mask_nodes (-1 outside)."""
        e = mask_nodes[src] & mask_nodes[dst]
        g = coo_matrix((np.ones(int(e.sum())), (src[e], dst[e])), shape=(n_r, n_r))
        _, lab = connected_components(g, directed=False)
        out = -np.ones(n_r, dtype=np.int64)
        if mask_nodes.any():
            out[mask_nodes] = np.unique(lab[mask_nodes], return_inverse=True)[1]
        return out

    seed_run = comps(is_seed) if n_r else np.zeros(0, int)
    cont_run = comps(~is_seed) if n_r else np.zeros(0, int)
    n_cr = int(cont_run.max()) + 1 if n_r and (cont_run >= 0).any() else 0
    # contacts: links between a continuation pixel and a seed pixel
    c1 = (~is_seed[src]) & is_seed[dst]
    c2 = is_seed[src] & (~is_seed[dst])
    cont_node = np.concatenate([src[c1], dst[c2]])
    seed_node = np.concatenate([dst[c1], src[c2]])
    size = np.bincount(cont_run[cont_run >= 0], minlength=n_cr) if n_cr else np.zeros(0, int)
    contact = pd.DataFrame({"run": cont_run[cont_node], "seed_run": seed_run[seed_node], "seed_node": seed_node})
    n_touch = contact.groupby("run").seed_run.nunique().reindex(range(n_cr), fill_value=0).to_numpy() if n_cr \
        else np.zeros(0, int)
    need = 1 if single_ended else 2
    bridged_run = (n_touch >= need) & (size <= max_gap)

    # flank strength: per (run, seed run) contact, the median q - tau over the n_flank seed pixels of that seed run
    # nearest the contact pixel(s)
    q_c = np.full(n_cr, np.nan)
    if n_cr and bridged_run.any():
        qn = q[rv, ru] - tau
        seed_nodes_by_run = pd.Series(np.nonzero(is_seed)[0]).groupby(seed_run[is_seed]).apply(np.asarray)
        flank = {}
        for (run, sr), g in contact[bridged_run[contact.run.to_numpy()]].groupby(["run", "seed_run"]):
            nodes = seed_nodes_by_run[sr]
            cu, cv = ru[g.seed_node.to_numpy()], rv[g.seed_node.to_numpy()]
            d = np.min(np.hypot(ru[nodes][:, None] - cu[None], rv[nodes][:, None] - cv[None]), axis=1)
            near = nodes[np.argsort(d, kind="stable")[:n_flank]]
            flank.setdefault(run, []).append(float(np.median(qn[near])))
        for run, vals in flank.items():
            q_c[run] = max(0.0, min(vals))
    q_cont = np.zeros((H, W))
    bridged = np.zeros((H, W), dtype=bool)
    if n_cr:
        node_b = (cont_run >= 0) & bridged_run[np.maximum(cont_run, 0)]
        bridged[rv[node_b], ru[node_b]] = True
        q_cont[rv[node_b], ru[node_b]] = q_c[cont_run[node_b]]
    runs = pd.DataFrame({"run": np.arange(n_cr), "n_px": size, "n_seed_runs": n_touch, "bridged": bridged_run,
                         "q_c": q_c})
    run_map = -np.ones((H, W), dtype=np.int64)
    if n_r:
        run_map[rv, ru] = cont_run
    return {"ridge": ridge, "seed": seed, "cont": cont, "bridged": bridged, "q_cont": q_cont, "runs": runs,
            "run_map": run_map, "lambda1": lam1}
