# Fold channel (draft, F2)

Status 2026-10-31: **draft**. Implemented in `illum_baseline/geom/` (F1, F2); not yet wired into Stage B (F3). The full
write-up (math with derivations, J bound, results) follows in F5. This draft records the decisions taken in F2 and
the data-derived constants, with where they come from.

Notation as in `TASK_FOLD_CHANNEL.md` and `docs/geometry_channels.md`: ρ_f is the trilateral-filtered disparity,
R_s = 9 px its support, k the window skip, m the window length, w_t the window half-width.

## 1. Pipeline as implemented

| step | where | notes |
|---|---|---|
| N1 dense edge-aware plane fit | `planefit.dense_fit` | tangent-plane gradient g from 8 half-disk side windows, then one refit with g = β (section 3) |
| N2 normals, Σ_n, σ_θ | `normals.normals` | |
| F2.1 candidates | `candidates.fold_candidates` | structure tensor of n̂, NMS along e₁, subpixel x₀; e₁ angle in (−90°, 90°], side "+" is +e₁ |
| F2.2 rounding width, k₁ | `candidates.rounding_width` | k₁ = max(k_min, ⌈width/2⌉ + k_margin) |
| F2.3 two-sided fits | `twosided.two_sided_fit` | contamination check (section 2), fallback to the dense normal at k₁ |
| F2.4 Δθ, F2.5 gate | `fold.FoldChannel` | two null models (section 4); default **multiplicative** |
| F2.6 veto | `fold.FoldChannel` | k₂ windows: no contamination check, only the 20% exclusion rule |
| F2.7 J, focal bracket | `fold.FoldChannel` | |
| F2.8 weight, rasterization | `fold.FoldChannel` | |
| Δρ* (record) | `twosided.two_sided_fit` | smallest plane gap over s ∈ [−k, k]; `planes_cross` |

**Arm A (k_min = 6) is the fold default.** Arm B (k_min = 11) is recorded as the planned depth-channel default
(`docs/geometry_channels.md` section 5).

## 2. Window contamination check

A one-sided window fit is accepted when

```
rms(residual)/α  ≤  max( planarity_factor · ε̂,  b_tri_margin · b_tri(k, m),  Q_0.95(null-window rms) )
```

- ε̂: median rms/α over the null-grid windows (k = k_min, each pixel's own e₁, no check in that pass).
- Q_0.95: the 95th percentile of the same null-window rms, per image.
- b_tri(k, m): the rms/α that the trilateral filter's bias alone puts into a window at skip k next to a feature.
  ρ_f is biased within R_s = 9 px of a feature (one-sided weights, contaminated gradient), and k_min = 6 lies inside
  that zone.
- Failing windows are retried with m/2; below m_min = 4 they are invalid and fall back to the dense normal (k₁ only).

### b_tri(k, m)

Measured by `scripts/calibrate_fold.py` on 58 noise-free single-feature scenes (folds 15–120° at 4 orientations,
convex and concave; occlusion steps of depth ratio 1.1–3 with three tilt pairs at 2 orientations). Only windows of
the channel's own candidates whose samples (all bilinear corners) lie on one surface and ≥ R_s + 1 px inside the
border count; b_tri = max over those, made non-increasing in k up to k = 16. Stored in
`illum_baseline/geom/data/b_tri.json` for the default trilateral (σ_s 3, σ_r 0.01, σ_g 3); `b_tri()` refuses other
filter settings.

| k | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|
| b_tri(k, m = 8) | 8.4e-5 | 8.4e-5 | 8.4e-5 | 1.2e-5 | 6.3e-6 | 5.6e-6 |
| b_tri(k, m = 4) | 1.0e-4 | 1.0e-4 | 1.0e-4 | 1.5e-5 | 7.9e-6 | 6.6e-6 |

- Beyond R_s the filter is exact; the residual ~5e-6 at k ≥ 10 comes from candidates whose x₀ sits a few px off the
  feature or whose e₁ is misoriented, so their windows still touch the bias zone. That is real contamination at the
  channel's own candidates, so it stays in the envelope.
- **`b_tri_margin = 1.25`.** With noise the window rms is about sqrt(b² + ε²); for b ≥ 3ε that is at most 1.054 b, so
  without a margin a bias-limited window just next to a feature would fail the check by a few percent. 1.25 covers
  that and the finite calibration set (the worst case at k = 6 is a 120° convex fold; other geometries can be a
  little worse). Checked on a held-out noise-free fold (45°, φ = 60°): every window at k ≥ 6 passes
  (`test_b_tri_bounds_held_out_fold`).
- The table must be regenerated when N1, the candidates or the trilateral parameters change (it was regenerated
  after the N1 fix: b_tri(6, 8) went from 4.4e-5 to 8.4e-5 because candidates now sit closer to the feature).

## 3. N1 gradient initialisation

The Gaussian-derivative gradient (σ_g = 3, support 12 px) sees a depth step, so the range weights rejected the
pixel's own side as well and the dense fit became invalid in a band along every step (Σ W = 3–5 within ±3 px of a
1.5× step, below min_weight_sum = 5). Fold candidates at silhouettes then sat 3–5 px off the edge.

Fix (geometry context only; the trilateral filter of tri_veto4 is untouched):

1. g₀ from 8 half-disk side windows (radius R_n, every 45°; spatial Gaussian σ_n, sky excluded; LS plane of
   ρ_f(x+d) − ρ_f(x) on (1, d)); the half-disk with the smallest relative residual Σ w r² / (Σ w ρ_f(x)²) among
   those with Σ w ≥ min_weight_sum. Gaussian gradient where none qualifies.
2. Weighted N1 fit with g₀; 3. once more with g = β of step 2 (`refit = 1`).

Result: Σ W ≥ 35 beyond 1 px of a step (only the pixel straddling the step drops); planes stay exact (≤ 4e-15).
`DenseFitParams(g_init="gaussian", refit=0)` reproduces the F1 fit. Cost: about 5 s per image on the CPU (was 1.5 s).

## 4. Gate: null models

T = δᵀ (Σ₊ + Σ₋)⁻¹ δ in the tangent basis at n̄. Two null models are computed for every candidate; `null_model` picks
the one that gates.

| | multiplicative (task spec, **default**) | additive |
|---|---|---|
| statistic | T | T_add = δᵀ (λ_n (Σ₊ + Σ₋) + σ_geo² I)⁻¹ δ |
| per-image calibration | λ̂ = median(T_null) / (2 ln 2), clamped to [0.5, 1e4] | σ_geo solved so that median(T_add, null) = 2 ln 2 |
| accept | T > λ̂ · 2 ln(1/p_fa) | T_add > 2 ln(1/p_fa) |

On real DA3 depth λ̂ = 625–1480: the window covariances describe noise, but real surfaces bend by σ_geo ≈ 0.9–6.3°
over the 20-px window span. A multiplicative λ̂ scales each candidate's own σ_θ, so the effective angle threshold
Δθ_gate = sqrt(13.8 λ̂) σ spreads by P90/P10 = 8–24 within an image (median 13–133°). The additive model separates
noise (λ_n Σ) from geometric roughness (σ_geo²): median Δθ_gate 5–41°, P90/P10 2.9–9.4.

**Default decision (rule: additive only if its share of w_fold < 0.5 is ≤ multiplicative + 0.02 in every
spot-check ROI):**

| ROI (network px, upright) | multiplicative | additive |
|---|---|---|
| 070 floor (T1 mask) | 0.000 | 0.000 |
| 079 grass: u 5–110 × v 5–110 and u 5–60 × v 360–495 | 0.000 | 0.009 |
| 079 grass, sunlit: u 110–290 × v 5–85 | 0.009 | **0.054** |
| 083 bollard: u 175–205 × v 260–330 | 0.059 | 0.059 |

The sunlit grass fails the rule, so the default stays **multiplicative**; the additive statistic is computed and
recorded for every candidate (`T_add`, `gate_add`, `dtheta_gate_add`). A single σ_geo per image is set by the
smoother surfaces and is too permissive on rough grass: see "local σ_geo" (open items).

The 083 bollard pixels with w_fold < 0.5 are all on the bollard's base contour where it meets the ground (130 px from
28 candidates, Δθ 94–98°): rim 130, limb 0, interior 0 (`baseline_output/diag/fold_f2/083_bollard_crop.png`).

### λ_n per k

Noise-only inflation of T: Monte-Carlo E[T]/2 over the null sample on noisy synthetic planes (three planes, two
seeds, i.i.d. relative noise on ρ before the trilateral filter), `illum_baseline/geom/data/lambda_n.json`. The
inflation comes from the noise correlation the trilateral filter and bilinear sampling introduce; it barely depends on
the noise level.

| k_min | noise 2e-4 | 5e-4 | 1e-3 | used |
|---|---|---|---|---|
| 6 (fold default) | 94.3 | 93.2 | 86.6 | **93.2** (`FoldParams.lambda_n`) |
| 11 (planned depth channel) | 123.0 | 121.2 | 111.6 | 121.2 |

λ_n is tied to k_min: a run with another k_min must pass the matching λ_n (the default is the k_min = 6 value).

## 5. Other F2 decisions

- **Veto (F2.6).** The k₂ windows get no contamination check (curved surfaces fail it, and an invalid k₂ window
  meant `no_veto`, i.e. acceptance). Cost: folds next to faces narrower than about 2k₂ + m are vetoed (synthetic
  18-px faces: all; 079 box region: 41 gate-passed candidates under the multiplicative gate, 64 under the additive
  one). The 64 (additive) are plotted in `baseline_output/diag/fold_f2/079_narrow_face_veto.png` with their adjacent
  face widths in the `.csv`: 58 along the plinth edge (one face 7–14 px), 5 near the box's top-right corner (16–34 px),
  1 with 6 px on both sides; none has both faces ≥ 2k₂ + m (which would be a bug).
- **Δρ*.** Smallest |ρ̂₊ − ρ̂₋| over s ∈ [−k₁, k₁] (affine, closed form); 0 with `planes_cross` when the planes
  intersect in that span. Record only (`depth_jump`); does not affect w_fold.
- **Test 1.** Δθ is exact (≤ 1e-9) at k = R_s + 2 = 11; at the channel's k₁ = 6–8 the trilateral bias leaves
  ≤ 0.154° (asserted ≤ 0.2°).

## 6. Open items

- **Local σ_geo.** σ_geo is one value per image; roughness varies across an image (grass vs boxes).
- **Narrow-face veto.** Real edges next to narrow faces are vetoed (079: the plinth edge). A veto that tolerates a
  second fold inside the k₂ window (e.g. compare against the nearer face only) is open.
- **Silhouette `depth_jump` recall at k_min = 6** is 75% on the synthetic box scene: misses sit a median 4.9 px from
  the true edge (detected ones 0.1 px), and k = 6 windows next to the step often fail the contamination check. The
  fold test now asserts only "no `depth_jump` on folds away from narrow faces" (0 of 1436 single-fold candidates)
  and reports recall; the strict recall requirement (≥ 90%, k ≥ 11) is in the depth-channel test plan
  (`docs/geometry_channels.md` section 5).
- **Highest fold weight share:** taiwei_cui_075 has share w_fold < 0.5 = 0.135 under the additive null (0.069
  multiplicative); to be examined in F4.
- **Narrow faces between two folds** produce `depth_jump` on fold candidates (box corner: the + window skips the
  narrow x-face, so the two fitted planes do not meet within ±k₁).
- Runtime: the N1 fix makes the context about 5 s per image on the CPU; the GPU port follows after F5.
