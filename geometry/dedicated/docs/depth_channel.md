# Depth-edge channel: design (draft, D0)

Status 2026-11-01: **design draft** (TASK_DEPTH_CHANNEL D0), with D1 (field) and D2 (candidates, calibration) implemented, D3 (jump, gates, magnitude, weights, `DepthChannel`) and D4 (registry, config, Stage B saving); sections 7–9 record what D2–D4 changed. The math is settled in
`TASK_DEPTH_CHANNEL.md` (D-K, D-1 … D-7) and only referenced here; this draft fixes the module map, the data flow, the
use of the shared two-sided fit (and the one change to it), the context's laziness for `[depth]`, and where the
evaluation classes and ROIs come from. It is finalized in D6 in the style of `docs/fold_channel.md`.

Conventions as in `docs/geometry_channels.md`: network resolution, upright frame, float64, pure numpy/scipy, no RGB,
no new DA3 pass. ρ_f is `GeometryContext.rho_f` (the tri_veto4 trilateral filter of ρ = 1/D, the same field the fold
channel uses); S = `ctx.S`, Ω = `ctx.omega`, ρ_floor = `ctx.rho_floor`; f_n = (K.f_x + K.f_y)/2 from
`Intrinsics.from_stage_a` (equal to the Stage A JSON's `focal.f_n` in per-image mode, tested in F1).

## 1. Module map

| module | contents | status |
|---|---|---|
| `geom/candidates.py` | `depth_kernels(sigma)` (g0, g1, g2 on the detector's construction; g1 = −x g0/σ² normalized to return exactly 1 on f(u) = u); `depth_field(ctx, p) -> DepthField` (ρ_σ, the Hessian ρ_uu, ρ_vv, ρ_uv, the detrended gradient ∇Dd = ∇ρ_σ − ∇ρ_{σ_L}, its direction n, q = nᵀHn / max(ρ_σ, ρ_floor), ŝ_q, the valid mask); `depth_candidates(ctx, p) -> DepthCandidates` (D-3: zero crossings with subpixel x₀, merging, polarity, lobes, two-sided height, sharpness, optional symmetry, σ_w,c and k_c) — replaces the stub | D1, D2 |
| `geom/twosided.py` | `gap_mode` (`min_span` default, `at_x0`), always-recorded x₀ gap and its SE; `usable=` (the sample mask; default Ω ∧ valid normals as now); a constant-fit helper for thin structures | D3 (section 3) |
| `geom/depth.py` | `DepthParams`, `DepthChannel` (name `"depth"`): candidates → two-sided fits at k_c (`gap_mode="at_x0"`, `fallback=False`, `usable=Ω`) with the m = 4 and constant-fit retries → sign check → gates (relative jump with hysteresis on 8-connected contours, significance with λ̂_Δρ from a stride-4 null) → contour-smoothed ρ̂± → ΔZ (sky, lower bound, focal bracket) → w_c, rasterization, strength | D3 |
| `geom/channels.py` | `"depth": DepthChannel` in the lazy registry | D4 |
| `config.py` | `DepthCfg` (field for field `DepthParams`, tested), `geometry.depth`, validation, nested `--set` | D4 |
| `stage_b.py` | nothing new: `_channel_params` already returns `asdict(cfg.geometry.<name>)`; saving, tables and JSON follow from `ChannelResult` | D4 |
| `viz_depth.py`, `diag_depth.py`, `cli.py` | depth sheet `stage_b/<run>/viz_depth/<stem>.png`, profile tool `diag-depth`, `viz` renders it when the run has the depth channel | D5 |
| `scripts/calibrate_depth.py` | σ_w,max (P95 of σ_w,c at unambiguous occlusions), λ̂_Δρ per k on synthetic noise and on the real null → `geom/data/depth_calibration.json` | D2 |
| `geom/synthetic.py` | step scenes against background slopes (both polarities), poles 4–20 px wide, blurred steps (DA3-like smear), cylinders (limb bias), steps to sky and to a very distant background | D1–D3 |
| configs | `configs/candidate_depth_only.yaml` (`[depth]`, run `depth_v1`), `configs/candidate_depth_fold.yaml` (`[depth, fold]` with the fold_v2 settings, run `depth_fold_v1`) | D4 |

## 2. Data flow (one image)

```
GeometryContext(D, sky_prob, K, detector Params, DenseFitParams)
  disparity: S, Ω, ρ, ρ_floor          (shared, cheap)
  rho_f                                (trilateral, shared with fold)
  [dense, normals: NOT touched by the depth channel]
        │
        ▼
depth_field (D-K, D-1, D-2): ρ_σ = G_σ * ρ_f; Hessian (g2·g0, g0·g2, g1·g1); ∇Dd at σ and σ_L = 4 σ_tot;
                              n = ∇Dd/|∇Dd| (|∇Dd| ≥ k_g median); q = nᵀHn / max(ρ_σ, ρ_floor); ŝ_q
        ▼
depth_candidates (D-3): zero crossings of q → x₀, n(x₀) → polarity q(x₀ − n/2) > 0 > q(x₀ + n/2) → lobes on
                        s ∈ [−8, 8] → min(|q_p|, |q_m|) > 3 ŝ_q → s_lo ≤ s_sep ≤ s_hi → σ_w,c, k_c = max(11, ⌈2σ_w,c⌉+1)
        ▼
DepthChannel (D-4 … D-7): two_sided_fit(x₀, e₁ = n, k_c, m = 8, w_t = 4, gap_mode="at_x0", usable=Ω, fallback=False)
                          → retries m = 4, constant fit; else Δρ_doublet (flag thin)
                          → sign (Δρ* > 0, since side + is the near side) → r = Δρ*/ρ̂₋ with hysteresis (τ_low 0.02,
                          τ_high 0.04, ≥ 5 px) and significance |Δρ*|/(SE sqrt(λ̂_Δρ)) > 3.29
                          → running median of ρ̂₊, ρ̂₋ along contours → ΔZ = (f_n/300)(1/ρ̂₋ − 1/ρ̂₊), sky → inf,
                          lower bound when ρ̂₋ − z SE₋ ≤ 0 → w_c = max(0.05, ΔZ₀/(ΔZ + ΔZ₀)) → min-filters over radii b_c
        ▼
ChannelResult("depth", w_net = w_depth, strength = r, candidates, arrays {dZ, q_over_s, edge_mask}, scalars)
        ▼
combine(min) with the other channels; Stage B saving as for every channel
```

## 3. The shared two-sided fit

**Fields the depth channel reads** (`TwoSidedResult`, `SideResult` for sides p = +n, near; m = −n, far):

| field | use |
|---|---|
| `p.alpha`, `p.beta`, `p.c`, `p.cov` and the same for `m` | the plane values and their variances at x₀ (ρ̂±(x₀), SE±) |
| `p.planar`, `m.planar`, `p.m_used`, `p.excl`, `p.rms_rel` (and m) | which retry produced the fit; contamination diagnostics |
| `d_rho`, `d_rho_se` | Δρ* and SE(Δρ*) with `gap_mode="at_x0"` |
| `rho_p_gap`, `rho_m_gap` | ρ̂₊(x₀), ρ̂₋(x₀) with `gap_mode="at_x0"` (the gap point is x₀) |
| `planes_cross`, `s_gap` | record (always computed) |
| new `d_rho_min_span` | the min-span gap, recorded next to the at-x₀ one |
| `valid` | both sides usable |

The depth channel does not use `dtheta`, `T` or the normals beyond what `two_sided_fit` computes per window (cheap:
N2 from the window planes and K).

**Changes to `twosided.py`** (all with defaults that keep the current behaviour):

1. `gap_mode: str = "min_span"`. `min_span` runs the current code unchanged. `at_x0` sets `s_gap = 0`,
   `d_rho = ρ̂₊(x₀) − ρ̂₋(x₀)` (= the existing `d_rho_x0`), `d_rho_se`, `rho_*_gap` and `dZ_rel` at x₀. In both modes
   the result also carries `d_rho_x0` (already there), a new `d_rho_x0_se`, and a new `d_rho_min_span` (the min-span
   value), so both gaps are always recorded; `planes_cross` is always computed.
2. `usable: np.ndarray | None = None`: the sample mask. None means Ω ∧ `ctx.normal_valid` (the current rule); the depth
   channel passes Ω. Reason: `ctx.normal_valid` would build the dense normal fit (section 4).
3. `one_sided_constant(ctx, x0, e1, k, m, w_t, sign, usable)`: the thin-structure constant fit (β = 0, α the sample
   mean, SE = s/√n), a new function; `_side` is not changed. The depth channel drives the retries itself: m = 8 with
   the contamination check (which already retries m/2 = 4 for contaminated windows), then m = 4 for windows invalid by
   exclusion, then the constant fit, then the doublet estimate.

**Why fold stays bit-for-bit.** With the defaults (`gap_mode="min_span"`, `usable=None`) `two_sided_fit` executes the
same statements in the same order as now; the new fields are computed after, from values the function already has,
and do not feed back. The fold table is built from explicitly named `TwoSidedResult` fields and from
`SideResult.side_fields()`; new fields go only into `TwoSidedResult` (no new `SideResult` field, which
`side_fields()` would add to the fold table). The repro tests (`fold_only`, `tri_veto4_fold`, `fold_v2`,
`tri_veto4_fold_v2`, plus `baseline_v0`, both `tri_veto4`, geometry `[laplacian]`) enforce this: D4 adds `fold_v2`
and `tri_veto4_fold_v2` to `test_baseline_repro.py`.

## 4. Context laziness for `[depth]`

`GeometryContext` computes each field on first access (`functools.cached_property`). The depth channel reads only
`disparity` (S, Ω, ρ, ρ_floor), `rho_f`, `K` and `sky_prob`; it must never touch `dense`, `normals`, `n_hat`,
`Sigma_n`, `sigma_theta` or `normal_valid`.

- **Trap found in D0:** `two_sided_fit` reads `ctx.normal_valid` for the sample mask and, with `fallback=True`,
  `ctx.dense`, `ctx.n_hat`, `ctx.Sigma_n`. The depth channel therefore calls it with `usable=ctx.omega` and
  `fallback=False` (section 3).
- `normals_computed` in the JSON is `"_normals" in ctx.__dict__`, so it stays accurate: false for `[depth]`, true for
  `[depth, fold]` (the fold channel builds the normals; channel order does not matter).
- D4 test: after a `[depth]` run neither `dense` nor `_normals` is in `ctx.__dict__`, no `n_hat` is saved, and the
  JSON says `normals_computed: false`.
- Cost: `[depth]` skips the about 5 s dense fit; the trilateral filter (about 0.4 s) is shared with fold.

## 5. Evaluation classes and ROIs

- **Classes.** `eval-fold` reads `annotations/folds/<stem>.json` with polylines labelled fold, step, limb, texture
  (`geom/evaluate.py`). No image is annotated yet. *Lean:* until annotations exist, D5's "share of w_depth < 0.5 per
  class" is computed on the pinned ROIs and line targets below, each tagged with a class; a small
  `evaluate_weight(w, ann)` in `geom/evaluate.py` (share of polyline samples / area pixels below 0.5, per class) serves
  both the pinned set now and the annotations later.
- **Existing ROIs and targets** (`scripts/fold_null_rois.py`): 070 floor (T1 mask; texture/plane), 079 grass and 079
  sunlit grass (texture), 083 bollard surface (u 175–205, v 260–330), 079 box fold rows 230/254 (u 180–215), and the
  line targets: narrow-face folds (079, 076, 090 plinth edges; 080 side face edges) and the creases 076 façade ledge
  and 088 rock ridge (fold class).
- **New ROIs this task needs** (proposed, `scripts/depth_d0_rois.py`; overlays `baseline_output/diag/depth_v1/d0/`):

  | ROI | class | definition (upright network px) |
  |---|---|---|
  | 079 box silhouette left | step | (118.5, 128.0) → (112.9, 248.0) |
  | 079 box silhouette top-back L | step | (122.8, 111.8) → (232.9, 89.5) |
  | 079 box silhouette top-back R | step | (237.4, 95.8) → (312.3, 146.1) |
  | 079 box lit/shaded fold (rows 225–260 check) | fold | (201.5, 222.8) → (189.3, 263.1) |
  | 070 sky boundary | step | 2-px band around the boundary of the tri_veto4 sky mask, image border excluded (1766 px) |
  | 083 bollard limb L | limb | (149.9, 215.6) → (160.0, 311.8) |
  | 083 bollard limb R | limb | (230.5, 215.1) → (224.1, 311.9) |
  | 083 bollard base rim | fold | (169.0, 324.5) → (188.0, 332.0) → (210.4, 326.7) |

  Silhouettes and limbs are snapped onto the maximum |d log D / ds|, folds onto the maximum normal-angle slope. The
  snapped lit/shaded fold sits 3–4 px right of the RGB shading boundary and within 1–2 px of the accepted fold
  candidates: DA3's boundary offset; the depth position is the one the channel sees.

## 6. Flags and leans (borderline choices)

1. **Sample mask** `usable=Ω` for depth windows (not Ω ∧ valid normals): needed for laziness; depth windows at
   k ≥ 11 lie beyond the 1-px invalid-normal band of their own step anyway.
2. **Thin-structure retries** driven by `depth.py` with a new constant-fit helper, `_side` unchanged.
3. **Contamination check for depth windows:** the fold formula max(3ε̂, 1.25 b_tri(k, m), Q_0.95), with ε̂ and Q_0.95 from
   the depth channel's own stride-4 null (the same sample as λ̂_Δρ); at k ≥ 11, b_tri ≈ 6e-6 is negligible.
4. **Sky fraction of the far window** ("≥ 50% sky (prob ≥ 0.5)"): measured on the context's sky mask S (sky prob ≥ 0.5
   after the B1 morphology), consistent with Ω, rather than on the raw probability.
5. **Contours for hysteresis and smoothing:** 8-connected components of the candidates rounded to pixels; for the
   running median, candidates are chained along the edge (neighbours within 1.5 px, ordered along the tangent n⊥);
   pixels with more than two neighbours are junctions and break the chain.
6. **n at x₀:** bilinear interpolation of the n field (both components), renormalized; the sign of n is the
   gradient's (toward larger ρ, the near side), not canonicalized like the fold channel's e₁.
7. **Null for λ̂_Δρ:** stride-4 grid over valid Ω (|∇Dd| above the k_g floor not required: the null needs ordinary
   pixels), each pixel's own n, x₀ = the pixel centre, k = 11, `gap_mode="at_x0"`; λ̂ = median((Δρ*/SE)²)/0.455,
   clamped to [0.5, 1e4].
8. **Evaluation classes:** pinned ROIs as a stand-in until annotations exist (section 5).
9. **ΔZ₀ = 0.10 m** and `symmetry: off`, as the task leans; D3 reports ΔZ₀ ∈ {0.05, 0.10, 0.20} m.
10. **The `[depth, fold]` configuration** takes the fold_v2 settings (capped veto, multiplicative gate); the channels
    order is `[depth, fold]` as specified; the order only matters for saving, not for results.

## 7. D2 record (candidates, calibration)

- **Gradient floor** (replaces k_g × median |∇Dd|): g_n = |∇Dd| / max(ρ_σ, ρ_floor), σ̂_g = median(g_n over Ω) / 1.1774
  (Rayleigh median), valid where g_n > 3 σ̂_g. A crossing is kept when g_n, linearly interpolated between its two
  pixels, is above the floor. No recall loss on synthetic steps with r ≥ 5 % (edge coverage 1.000 with and without);
  on the real images 24–38 % of Ω is valid (g_n is far from Rayleigh there: real slopes and edges, not noise).
- **ŝ_q sample:** Ω at least R = ⌈4σ⌉ px from the sky and the border, *not* the gradient-valid pixels (with the new
  floor those are mostly edges).
- **q on the sky:** not zeroed. ρ = 0 there (the far limit), so a sky boundary is a doublet like any step; D-6 decides
  what it means (ΔZ = inf, sky_near).
- **D-3 implementation:** sign changes of q between 4-neighbour pixel pairs with at least one pixel in Ω; merged within
  0.5 px (single linkage, mean position); n at x₀ bilinear and renormalized; lobes are the runs of one sign *contiguous
  with x₀* along ±n (step 0.25 px, to ±8 px), peaks parabola-refined.
- **Centring stage (added):** |s_p + s_m| ≤ s_sep/3. Beyond a step's core the detrended n flips sign
  (−∇ρ_{σ_L} dominates) and the trilateral filter leaves small wiggles 5–7 px out; a crossing there borrows the step's
  own lobe ~5 px away (|s_p + s_m|/s_sep ≈ 0.54) and, with the prior smear bound, passed D-3 as a parallel ghost line
  (hundreds per step). Real unambiguous occlusions: max 0.29. Residual noise doublets 6–10 px from large noisy steps
  carry the flipped n, so their two-sided jump is negative (D-4 `sign` rejects them; checked in D2).
- **σ_w,max = 0.83 px** (P95 of σ_w,c over 576 unambiguous occlusions on the 10 images; P50 0, P99 1.05), so
  s_hi = 3.0 px. Real DA3 edges at network resolution are sharper than an ideal pixel-sampled step (s_sep 2.45 vs
  2.72 at P50): DA3 slightly over-sharpens.
- **λ̂_Δρ:** synthetic noise ≈ 98–106 (median-based; MC mean 106–119), flat in k ∈ {11, 13, 16}; real images 660–1391.
- **ρ_f vs ρ (070 floor ROI):** ρ_f leaves 0 candidates in the floor ROI after the height test, raw ρ 193 after all of
  D-3. ρ_f stays the default.
- **Shared fit:** `two_sided_fit(..., gap_mode, usable)` and the new `d_rho_x0_se`, `d_rho_min_span` fields (section 3).
- Scripts: `scripts/calibrate_depth.py` → `geom/data/depth_calibration.json`; `scripts/depth_d2_report.py` →
  `baseline_output/diag/depth_v1/d2/` (counts, ablation, DA3-vs-RGB alignment).

## 8. D3 record (jump, gates, magnitude, weights)

Changes to the D2 parameters (decided after D2): `centre_tol` 0.4; σ_w,max = P99 (now 1.02 px, see below).

- **Lobes on h.** q = h/ρ_σ with h = nᵀHn; ρ_σ varies across a doublet, which drags the far lobe outward for large
  jumps (far peak at −1.76 σ_tot·px for r = 1, −2.1 for r = 5, −3.3 for r = 100; s_sep then exceeds s_hi and
  steps in front of distant backgrounds failed sharpness). Lobe positions are now found on h (peaks at ∓σ_tot for any
  r); heights stay q at those peaks; the doublet magnitude is 1.03 |h|_lobe s_sep². Recalibrated σ_w,max: P99 1.02 px
  (P95 0.77).
- **Nearest peak.** Inside a thin structure the near lobes of both edges form one run; the lobe is the nearest local
  peak reaching ½ of the run maximum (8-px poles: s_sep 8.0 → 2.9).
- **Jump (D-4).** Per side: plane at k_c, m = 8 with the contamination check (internal retry m = 4) → m = 4 → constant
  fit → doublet. Obstacles: D-3 candidates whose own doublet estimate passes τ_low (noise doublets ≤ 1e-3);
  a side whose window lies beyond one uses the constant fit on the clear span. Fit modes and `d_next_±` recorded.
- **Gates (D-5).** sky_near → sign (Δρ* > 0) → significance (|Δρ*|/(SE√λ̂) > 3.29; doublet: min lobe/ŝ_q) → r > τ_low →
  components (link radius 3 px, ≥ 5 distinct pixels, one member r > τ_high). Sky edges pass with r = inf.
- **Magnitude (D-6).** Plane-aware running median along mutual-nearest-neighbour chains; ΔZ = (f_n/300)(1/ρ̂₋ − 1/ρ̂₊);
  lower bound with √λ̂-inflated SEs (inf when unbounded); focal bracket recorded.
- **Weight (D-7).** w_c = max(0.05, ΔZ₀/(ΔZ + ΔZ₀)); bands max(2, ⌈σ_w,c⌉ + 1); strength r (inf at sky edges);
  arrays `dZ` (float32, NaN off the bands, inf at sky), `q_over_s` (float16, clipped), `edge_mask`.

**Tests 6–14** (`tests/test_depth_channel.py`): single-step recall 1.000 (r 5–200 %, smear ≤ 0.8, against slope);
box silhouette recall 0.910; noise-free ΔZ error ≤ 1.3e-14, smear ≤ 3 px ≤ 0.9 %; scale/focal invariance exact; sky
inf / w_min and sky_near; horizon lower bound flagged (unbounded); poles 6–20 px detected (6 px doublet, ΔZ error
15 %; 8–12 px constant fit; 16–20 px planes; 4 px not detected: both lobes merge into one peak); limb bias +7 to +32 %
(within the cylinder's own depth change over k px, 10–44 %); λ̂_Δρ −11 % vs the Monte-Carlo inflation.

**Real images** (`scripts/depth_d3_report.py` → `baseline_output/diag/depth_v1/d3/d3_report.json`): 0.56–0.79 s per
image after the context, normals never built. Significance, not τ_low, binds for 73–100 % of candidates (r_gate P50
0.035–0.34; P90/P10 7–51). 090: a chain-link fence (DA3 renders the mesh as depth blobs) gives 1714 thin accepted
candidates in small rings (w < 0.5 on 7.9 % of the image).

## 9. D4 record: integration and saved outputs

**Registry and config.** `CHANNELS["depth"] = DepthChannel`. `geometry.depth` (`DepthCfg`) mirrors `DepthParams` field
for field (tested), with validation and nested `--set` (e.g. `--set geometry.depth.null=additive`). The gate null is
`geometry.depth.null`: `multiplicative` (default; λ̂_Δρ) or `additive` (λ_n SE² + σ_geo,ρ²); both significance columns
are always recorded. In YAML the key is written quoted, `"null": multiplicative` (a bare `null` key is YAML's null);
PyYAML quotes it when it writes `config_resolved.yaml`. Configs: `configs/candidate_depth_only.yaml` (`[depth]`, run
`depth_v1`), `configs/candidate_depth_fold.yaml` (`[depth, fold]` with the fold_v2 settings, run `depth_fold_v1`);
`configs/default.yaml` is unchanged (geometry off).

**Records.** The JSON `params.geometry.depth` block is written only for runs that include the depth channel, so every
earlier geometry run reproduces its record bit for bit (`fold_only`, `tri_veto4_fold`, `fold_v2`,
`tri_veto4_fold_v2` in `tests/test_baseline_repro.py`). Channel arrays keep float32 / float16 / bool dtypes; float64
arrays are stored as float16 as before (fold's `r_n` unchanged).

**Saved outputs** (`stage_b/<run>/`, arrays in the stored frame at network resolution):

| key / file | content |
|---|---|
| `depth_w_net` (float16) | w_depth ∈ [w_min, 1] |
| `depth_strength` (float16) | r = ΔZ/Z_near at rasterized pixels (max over overlapping bands), 0 elsewhere. **inf at sky edges** (and where the far side is unbounded): any future fusion on strengths must treat inf as saturated |
| `depth_dZ` (float32) | ΔZ in metres at rasterized pixels (max over bands), NaN elsewhere, inf at sky edges / unbounded far sides |
| `depth_q_over_s` (float16) | signed q/ŝ_q, clipped to ±6.5e4 |
| `depth_edge_mask` (bool) | the accepted candidates' rounded pixels |
| `<stem>_depth_candidates.parquet` | every D-3 candidate (merged crossings above the gradient floor): position, n, lobes (q at the h peaks), s_sep, centring, σ_w,c, k_c; for D-3 survivors the jump (fit modes per side, d_next±, ρ̂±, SE±, Δρ*, its SE, the min-span gap, planarity / m_used / rms / exclusion of the m = 8 fit), the doublet and its ratio, `signif` (multiplicative) and `signif_add`, r, r_gate, sky shares, flags (`sky_edge`, `sky_near`, `thin`, `planes_cross`, `r_lower_bound`, `lower_bound`), component id / size / max r, smoothed ρ̂±, ΔZ with its focal bracket, w_c, band, `accepted` and `reason` ∈ {polarity, height, sharpness, centring, symmetry, sky_near, sign, signif, rel, component, accepted} |
| `<stem>_depth_null.parquet` | the null grid: u, v, Δρ*, SE, planar± |
| JSON `channels.depth` | ŝ_q, σ̂_g, λ̂_Δρ (raw and clamped), σ_geo,ρ, `null`, σ_w,max, s_lo, s_hi, counts per stage and per fit mode, r_gate P10/P50/P90, the share where significance binds, focal f_n and σ_f/f, shares w < 0.5 / 0.9, params, context and runtime |
| manifest | `depth_*` columns (counts, ŝ_q, λ̂, σ_geo,ρ, null, σ_w,max, r_gate P50, share binding, shares of w, runtime) for depth runs only |

A `[depth]` run never builds the dense fit or the normals: no `n_hat` / `sigma_theta` / `normal_valid` arrays,
`normals_computed: false` (tested). `[depth, fold]` saves the elementwise min of the two channels' w (tested).
