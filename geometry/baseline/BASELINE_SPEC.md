# Baseline geometric-edge weights for an illumination smoothness loss — implementation spec

## How to work

- Implement milestone by milestone (M0–M5). After each milestone, **stop**, report, and wait for my go-ahead.
- Report tersely: what was done, test results, open issues.
- Decisions in this spec are settled; apply them without asking. Where the data or a library forces a choice this spec does not cover, **flag it explicitly with your lean**. Do not resolve it silently.
- Never guess third-party APIs. Read the installed source of `depth_anything_3` and `geocalib`, and name the file/function each call relies on.
- Keep the math (`detector.py`) free of torch and network imports so it is unit-testable.
- If a test assertion fails for a principled reason, report it. Do not loosen it silently.

## Background

Goal: a per-pixel weight `w ∈ [w_min, 1]` that down-weights an illumination-map smoothness loss at geometric features (depth discontinuities **and** creases) **without classifying them**.

- DA3METRIC-LARGE outputs canonical depth `D`. Metric depth is `Z = f_n · D / 300`, where `f_n` is the focal length in pixels **at the resolution DA3 processed**.
- Disparity `ρ = 1/D` is affine in pixel coordinates on any plane (pinhole camera). So `∇²ρ = 0` on planes of **any** slant, while steps (doublet response) and creases (one-signed bump) respond.
- The detector uses `q = |∇²ρ_σ| / ρ_σ`. Normalizing by `ρ` makes `q` invariant to global depth scale, and hence to focal length, and matches DA3's roughly relative noise.
- Consequence: **the weights do not depend on GeoCalib**. GeoCalib only provides `f` for the saved metric depth and QA.
- This is a deliberate baseline. No doublet/polarity tests, no jump estimation, no ΔZ weighting, no normal model.

## Data and frames

- `data_dir`: linear raw `.tif` images, W×H = 1023×768 or 768×1023. Exact format unknown until the M1 audit.
- `out_dir`: all outputs. Never write into `data_dir`. Never modify the TIFFs.
- **Stored frame**: the pixel array as stored in the TIFF. **Upright frame**: after applying the TIFF Orientation tag (274).
- All network and detector work happens in the upright frame. **Final deliverables are saved in the stored frame**, so they align with the linear TIFF and the illumination targets. Visualizations are upright.
- Pixel-center convention throughout: principal point `((W−1)/2, (H−1)/2)`. Resizes use half-pixel alignment.

## Environment

- Python ≥ 3.10, PyTorch with CUDA if available (fall back to CPU and warn).
- Depth Anything 3: `git clone https://github.com/ByteDance-Seed/Depth-Anything-3` then `pip install -e .`
- GeoCalib: `pip install -e "git+https://github.com/cvg/GeoCalib#egg=geocalib"`
- Also: `numpy scipy opencv-python-headless tifffile imagecodecs matplotlib pandas pyarrow pyyaml tqdm pytest`
- Write `requirements.txt` and pin the versions you actually installed.

## Repo layout

```
illum_baseline/
  __init__.py
  config.py        # dataclasses + YAML load/validate; resolved config snapshot
  io_tiff.py       # TIFF load, tag reading, orientation to_upright/to_stored
  preprocess.py    # linear -> sRGB uint8 network input
  geometry.py      # network size, scale factors, focal rescaling
  calib.py         # GeoCalib wrapper
  depth.py         # DA3 wrapper
  stage_a.py       # orchestration + cache
  detector.py      # pure numpy (float64): kernels, disparity, LoG, q, noise, weights
  outputs.py       # upsampling, frame transforms, pair_weights() helper, saving
  manifest.py
  viz.py
  cli.py           # audit | stage-a | stage-b | viz | qa
configs/default.yaml
docs/api_notes.md
tests/
```

## Config — `configs/default.yaml`

```yaml
paths:
  data_dir: /mnt/f/NEU_Seattle/CS_7980 Research Capstone/Project/codebase/geometry/data     # set before M1; ask me if null
  out_dir: /mnt/f/NEU_Seattle/CS_7980 Research Capstone/Project/codebase/geometry/baseline_output
  glob: "**/*.tif*"
device: cuda
seed: 0

preprocess:               # proposed by the M1 audit, confirmed by me
  layout: auto            # rgb | mosaic | auto (fail loudly if ambiguous)
  cfa_pattern: null       # RGGB | BGGR | GRBG | GBRG, if mosaic
  black_level: null       # scalar or per-channel
  white_level: null
  white_balance: grayworld   # metadata | grayworld | none (network copy only)
  exposure_percentile: 99.0
  exposure_target: 0.9
  exposure_gain_max: 64.0
  respect_orientation_tag: true

network:
  long_side: 504
  multiple: 14
  resize_interp: area

geocalib:
  weights: pinhole
  camera_model: pinhole

focal:
  mode: per_image         # per_image | pooled
  group_key: null         # e.g. [Make, Model, FocalLength]; null = one group

da3:
  model_id: depth-anything/DA3METRIC-LARGE
  process_res: 504
  process_res_method: upper_bound_resize   # must be a no-op on our pre-resized input (verify in M0)
  batch_size: 8
  canonical_focal: 300.0

sky:
  threshold: 0.5
  open_close_radius: 1
  min_area_px: 50

detector:
  sigma: 1.25             # px at network resolution
  sigma_w_prior: 1.0      # DA3 boundary smear prior, px
  truncate: 4.0
  rho_floor_frac: 0.5
  rho_floor_percentile: 1.0
  k_tau: 3.0
  k_eps: 3.0
  w_min: 0.05
  s_hat_min: 1.0e-6
  s_hat_max: null

run:
  name: baseline_v0
  limit: null
  force: false
```

---

## M0 — Environment and API verification

Install everything, verify the GPU, and read the library source to confirm the following. Record each with file/function references in `docs/api_notes.md`.

**DA3**
- How to load DA3METRIC-LARGE.
- The `inference()` signature, and which input types it accepts (uint8 HxWx3 numpy? PIL? paths?).
- The `Prediction` field names and shapes for depth and sky. Check `depth_anything_3/specs.py`. Is sky a probability, a score, or logits?
- Exactly what `upper_bound_resize` does to a 504×378 or 378×504 input. It must be a no-op. If it is not, find the setting that is.
- Whether outputs come back at the processed resolution.
- Whether a confidence map is returned.

**GeoCalib**
- The `calibrate()` input format: expected `[3,H,W]` float RGB in `[0,1]`.
- Confirm that the returned camera is in **input-image** pixel coordinates (look for the undo-scale/crop step).
- Where `fx`, `fy`, `focal_uncertainty`, `vfov`, roll/pitch and their uncertainties live.
- The `shared_intrinsics=True` batch mode.

**Deliverables**
- `docs/api_notes.md`.
- A smoke script that runs both models on one image, any JPEG is fine.

---

## M1 — Dataset audit

Command: `python -m illum_baseline.cli audit`

Sample all files if there are ≤ 500, else a seeded random 200. Record per file:
- dtype, shape, channel count.
- min/max and the 0.1/99.9 percentiles per channel.
- All TIFF tags: BitsPerSample, SamplesPerPixel, Photometric, Orientation, Make/Model, FocalLength, any CFA/DNG tags (CFAPattern, BlackLevel, WhiteLevel, AsShotNeutral), and Compression.
- Fraction of pixels at the max value (saturation).
- Count of each W×H and each Orientation value.

**Outputs**
- `out_dir/audit/audit.csv`.
- `out_dir/audit/audit_report.md` with a **proposed `preprocess:` and `focal.group_key` config**, each item with evidence. Flag ambiguities (e.g. RGB vs mosaic, black level, whether stored arrays are already upright).
- Thumbnails of 8 images after a provisional sRGB conversion: 4 landscape, 4 portrait.

**STOP.** I confirm the preprocessing config before M2.

---

## M2 — Stage A (inference, cached)

Command: `python -m illum_baseline.cli stage-a`. Resumable: skip images whose cache exists unless `--force`.

### A1. Preprocess (network copy only; the linear data is untouched)

1. Load into the stored frame. Convert to float32, subtract the black level, divide by `(white − black)`, clip to `[0, 1]`.
2. If mosaic: demosaic with the confirmed CFA pattern. OpenCV's Bayer code names are offset relative to common naming, so the demosaic test must verify the pattern mapping.
3. White balance per config: gains from metadata, or gray-world over unsaturated pixels.
4. Exposure: `Y = 0.2126R + 0.7152G + 0.0722B`. Gain `= min(target / P_pct(Y over unsaturated pixels), gain_max)`. Apply the gain and clip to `[0, 1]`.
5. sRGB OETF: `12.92x` for `x ≤ 0.0031308`, else `1.055·x^(1/2.4) − 0.055`. Convert to uint8.
6. `to_upright` per the Orientation tag. Implement all 8 tag values with an exact inverse `to_stored`.
7. Record: levels, WB gains, exposure gain, saturation fraction, orientation tag.

### A2. Network size

- Upright size `(W_up, H_up)`. Long side → 504. Short side → `14·round(short·504/long/14)`. This gives 1023×768 → 504×378 and 768×1023 → 378×504.
- Resize with `cv2.INTER_AREA`.
- Record `s_x = W_n/W_up` and `s_y = H_n/H_up`. These differ by about 0.1%; keep both.

### A3. GeoCalib

- Run on the **full-resolution upright sRGB** image with weights/model `pinhole`.
- Store `fx, fy` (upright full-res pixels), `focal_uncertainty`, `vfov`, roll, pitch, and their uncertainties.
- Per group (`focal.group_key`, or a single group), compute the pooled `f` as the median of per-image `f`. Optionally cross-check with `shared_intrinsics=True` on a seeded subset of ≤ 16 images per group.
- Store both per-image and pooled `f`. `focal.mode` selects the active one.
- Active focal at network resolution: `f_n = (fx·s_x + fy·s_y) / 2`.

### A4. DA3METRIC-LARGE

- Input: the upright network-resolution uint8 RGB image. Batch by shape.
- Use `process_res` and `process_res_method` per the M0 findings, and **assert** the processed shape equals the input shape. DA3 must not resize again.
- Get the canonical depth `D` (float32, H_n×W_n, upright) and the sky probability. If the sky output is logits or a score, convert per the source.
- Do not convert to metric in the cache. `Z` is computed on demand.

### Cache (upright frame)

- `out_dir/stage_a/<stem>.npz`: `D` float32, `sky_prob` float16.
- `out_dir/stage_a/<stem>.json`: all A1–A4 metadata, model IDs, package versions, git hash.
- `out_dir/stage_a/srgb/<stem>.png`: the upright full-resolution sRGB image, for visualization.

**Run on 5 images** (both orientations) and report a cache summary.

---

## M3 — Stage B detector and weights

`detector.py`: pure numpy, **float64**, network resolution, upright frame. One function:
`compute_weights(D, sky_prob, params) -> dict` of arrays and scalars.

### B1. Sky mask

- `S0 = sky_prob ≥ threshold`.
- Morphological open, then close, with `disk(open_close_radius)`.
- Remove 8-connected sky components smaller than `min_area_px`. Do **not** fill holes: holes in the sky are real objects.
- Record `sky_frac`. If DA3 provides no sky output, `S` is empty.

### B2. Disparity

- `ρ = 1/D` on `~S`, and `ρ = 0` on `S`.
- Non-finite or `D ≤ 0` on `~S`: clamp `D` to `1e-6·median(D[~S])`, and record `n_bad_depth`.

### B3. Smoothing and Laplacian

Use **custom separable kernels and odd-reflect padding**.

**Why not the obvious tools.** `scipy.ndimage.gaussian_laplace` uses a truncated sampled kernel whose sum is not zero. That leaks about `1e-3/σ²·ρ` into `L`. Even-reflect padding creates kinks on slanted planes. Both make `L ≠ 0` on planes.

- Kernel support: `R = ceil(truncate·σ)`, `x ∈ [−R, R]`.
- `g0 = exp(−x²/2σ²)`, then `g0 /= g0.sum()`.
- `g2 = (x²/σ⁴ − 1/σ²)·g0`. Then `g2 −= g2.sum()·g0` so `Σg2 = 0`. Then `g2 /= (Σ x²·g2) / 2` so `Σx²g2 = 2`.
- Pad `ρ` by `R` with `np.pad(mode="reflect", reflect_type="odd")`, which preserves affine functions exactly. Then use valid-mode separable convolution:
  - `ρ_σ = (g0 ⊗ g0) ∗ ρ`
  - `L = (g2 along x, g0 along y) ∗ ρ + (g0 along x, g2 along y) ∗ ρ`

### B4. Normalized response

- `ρ_floor = rho_floor_frac · P_{rho_floor_percentile}(ρ[~S])`.
- `q = |L| / max(ρ_σ, ρ_floor)`. This stays finite next to the sky, and the sky boundary still gets a large `q`, which is intended.
- Set `q = 0` on `erode(S, disk(r))`, the sky interior. The boundary band keeps its response.

### B5. Noise scale

- `valid = ~dilate(S, disk(r))` and at least `R` px from the image border.
- `ŝ = median(q[valid]) / 0.6745`. Because `q` is already an absolute value, its median is a robust std estimate.
- Clamp to `[s_hat_min, s_hat_max]`.

### B6. Threshold, then pool

The order matters. Pooling first raises the noise floor and invalidates `τ`.

- `τ = k_tau·ŝ` and `ε = k_eps·ŝ`.
- `q_eff = max(q − τ, 0)`.
- `r = ceil(sqrt(σ² + sigma_w_prior²)) + 1`. With the defaults, `r = 3` (a 29-px disk).
- `q_pool = grey_dilation(q_eff, footprint=disk(r))`, isotropic.

### B7. Weight

- `w = max(w_min, ε / (q_pool + ε))`.
- `w = 1` wherever `q_pool = 0`: planes and the sky interior.

### B8. Outputs

- Upsample `w` in the upright frame to `(W_up, H_up)` with `cv2.INTER_LINEAR` (half-pixel centers). Clip to `[w_min, 1]`. Then `to_stored`.
- Network-resolution arrays: apply `to_stored` only. **Never differentiate an upsampled map.**
- Provide `pair_weights(w) -> (w_x [H, W−1], w_y [H−1, W])`, the elementwise min of horizontal and vertical neighbors. It is computed at load time and not stored.

---

## M4 — Saving, manifest, visualization

Command: `python -m illum_baseline.cli stage-b --run-name ...`. CLI overrides for detector params are allowed. Each run writes to its own directory, so parameter variants coexist. Also write a resolved config snapshot.

### Per image: `out_dir/stage_b/<run>/<stem>.npz` (stored frame)

| key | resolution | dtype | notes |
|---|---|---|---|
| `w` | original | float16 | **the deliverable** |
| `w_net` | network | float16 | |
| `q` | network | float16 | for re-thresholding |
| `L_over_rho` | network | float16 | signed `L / max(ρ_σ, ρ_floor)` |
| `D` | network | float32 | source of everything; never float16 |
| `sky_mask` | network | bool | |

### Per image: `<stem>.json`

- Detector scalars: `ŝ, τ, ε, ρ_floor, r, R, sky_frac, n_bad_depth, frac_w_lt_0.5, frac_w_lt_0.9`.
- Focal: `f_active` (upright full-res px), `f_n`, focal mode, per-image and pooled `f`, GeoCalib uncertainties.
- Geometry: `s_x, s_y`, orientation tag, frame notes.
- Provenance: all params and the Stage A metadata reference.

### `manifest.parquet` (CSV fallback)

One row per image with the scalars above, plus boolean flags:
- `sky_frac > 0.5`
- GeoCalib `focal_uncertainty / f > 0.15`
- `ŝ` outside the dataset P2–P98
- `n_bad_depth > 0`
- `frac_w_lt_0.5 > ` dataset P98

### Visualization: `out_dir/stage_b/<run>/viz/<stem>.png`

Upright frame, 2×4 panels:
1. sRGB with the sky-mask contour.
2. `log10 Z` in metres, using the active `f`, turbo colormap, sky gray, with a colorbar.
3. `L_over_rho / ŝ`, signed, RdBu_r, fixed symmetric limits ±20. Steps show as ± doublets, creases as one-signed bands.
4. `log10(q/ŝ)` clipped to `[−1, 3]`, with contours at `k_tau` and `k_tau + k_eps`.
5. `w` on `[w_min, 1]`, grayscale.
6. sRGB at original resolution overlaid in red with `alpha = (1 − w) / (1 − w_min)`.
7. Histogram of `q/ŝ` (log y) with lines at `1`, `k_tau`, and `k_tau + k_eps`.
8. A text box with the key scalars.

Plot in units of `ŝ` with fixed limits so panels are comparable across images. Add a `viz` command that re-renders from saved outputs only.

**Run M3+M4 on 20 seeded images** (both orientations). Report the paths of 4 representative panels.

---

## M5 — Full run and QA

- Full Stage A + B with resume and a progress bar.
- `qa` command writes `qa_report.md`:
  - histograms of `ŝ`, `sky_frac`, `f`, `focal_uncertainty/f`, `frac_w_lt_0.5`;
  - pooled vs per-image `f` per group;
  - the flagged-image list;
  - suggested dataset-level `s_hat_min`/`s_hat_max` (P2/P98).

---

## Tests (pytest)

1. **sRGB OETF**: `0→0`, `0.0031308→0.04045`, `1→1`, monotone.
2. **Orientation**: for tags 1–8, on random non-square arrays with and without channels, `to_stored(to_upright(x)) == x`. Document the convention and check tag 6 against a reference rotation.
3. **Network size**: (1023,768)→(504,378) and (768,1023)→(378,504). Both dims are multiples of 14.
4. **Kernels**: `Σg0 = 1`, `|Σg2| < 1e-12`, `Σx²g2 = 2` (to 1e-12), odd moments 0.
5. **Planes exact**: build a synthetic K at network resolution (f = 400 px, principal point at the image center per the convention above). For ≥ 5 random visible planes (`n·r > 0.05` everywhere), `D = (d/(n·r))·300/f`. Assert `max q < 1e-9` over the **whole** image, borders included.
6. **Features respond**: ray-cast a floor (camera 1.5 m high) plus an axis-aligned box 2–4 m away, and record a surface-ID image. Feature pixels are surface-ID boundaries. Plane-interior pixels are those farther than `R+1` px from any boundary. Noise-free, assert:
   - `max q(plane interior) < 1e-9`;
   - `median q(feature) > 1e-4`;
   - `≥ 95%` of feature pixels have `q > 1e-6`.
7. **Noise estimator**: one plane plus i.i.d. Gaussian noise on `ρ` with std `1e-3·ρ` (fixed seed). Assert `ŝ` is within 10% of `1e-3·‖k_LoG‖₂`, where `k_LoG = outer(g0,g2) + outer(g2,g0)`. Also **report, do not assert**, the fraction of pixels with `w < 1` for `k_tau ∈ {3, 4, 5}`.
8. **Scale invariance**: with noise, `w(D)` and `w(c·D)` for `c ∈ {0.37, 2.9}` agree to `1e-9`, with the `ŝ` clamp inactive.
9. **Threshold-before-pool**: an isolated spike with `q < τ` gives `w ≡ 1`. Check against a reference implementation.
10. **Upsample alignment**: a 1-px vertical line at network column `c` upsamples to an intensity centroid at `(c + 0.5)/s_x − 0.5`, ±0.1 px.
11. **Pair weights**: correct shapes and min semantics.
12. **Demosaic** (only if the data is a mosaic): mosaic a smooth color image with each CFA pattern, demosaic, and check the colors are recovered. This catches pattern-name offsets.
13. **Integration** (marked `slow`/`gpu`): Stage A+B on 3 real images. The outputs exist and have the right shapes and frames, and `w ∈ [w_min, 1]`.

---

## Do not

- Feed linear (non-gamma) images to DA3 or GeoCalib, or modify the TIFFs.
- Let DA3 resize internally, or differentiate any upsampled map.
- Use `scipy.ndimage.gaussian_laplace`, even-reflect padding, or float16 in any derivative computation.
- Add doublet/polarity tests, jump estimation, ΔZ weighting, or normal estimation. Those are later tiers.
- Store metric `Z` or pair weights. Both are derivable from the saved data.
