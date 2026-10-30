# API notes (M0)

Source read from the installed packages:

- **DA3**: editable install of `../DA3/Depth-Anything-3` (commit `3d835ec`), package `depth_anything_3`. Paths below are relative to `src/depth_anything_3/`.
- **GeoCalib**: `git+https://github.com/cvg/GeoCalib` @ `1dedb29f94fc11b896c5f572005978aeebccd67f`, package `geocalib`.

Empirical confirmations come from `scripts/smoke_m0.py`, see [Smoke results](#smoke-results).

---

## Depth Anything 3 (DA3METRIC-LARGE)

### Loading

- `api.py: DepthAnything3(nn.Module, PyTorchModelHubMixin)`. Load with `DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(device)`.
- The Hub `config.json` holds `{"model_name": "da3metric-large", ...}`. The mixin passes it to `DepthAnything3.__init__(model_name)`, which builds the net from `configs/da3metric-large.yaml` (via `registry.py: MODEL_REGISTRY`, `cfg.py: load_config/create_object`) and loads `model.safetensors`.
- Architecture (`configs/da3metric-large.yaml`):
  - `DepthAnything3Net` with a DinoV2 `vitl` backbone, `alt_start: -1`, and a `DPT` head with `output_dim: 1`.
  - There is no `cam_dec`, `cam_enc` or GS head.
- **Views are independent in the network.** `model/dinov2/vision_transformer.py: _get_intermediate_layers_not_chunked` only uses `"global"` (cross-view) attention when `alt_start != -1`. With `-1`, every block is `process_attention(..., "local")`, so attention stays within each view. `model/dpt.py: DPT.forward` reshapes to `(B*S, N, C)` and processes each view separately.
- **The only cross-image coupling** is in `model/da3.py: DepthAnything3Net._process_mono_sky_estimation`. See [Sky](#sky).

### `inference()` signature

`api.py: DepthAnything3.inference(image, extrinsics=None, intrinsics=None, align_to_input_ext_scale=True, infer_gs=False, use_ray_pose=False, ref_view_strategy="saddle_balanced", render_exts=None, render_ixts=None, render_hw=None, process_res=504, process_res_method="upper_bound_resize", export_dir=None, export_format="mini_npz", export_feat_layers=None, conf_thresh_percentile=40.0, num_max_points=1_000_000, show_cameras=True, feat_vis_fps=15, export_kwargs={}) -> Prediction`

- `image: list[np.ndarray | PIL.Image | str]`. Loading happens in `utils/io/input_processor.py: InputProcessor._load_image`:
  - `str`: `Image.open(path).convert("RGB")`.
  - `np.ndarray`: `Image.fromarray(arr).convert("RGB")`. The docstring says "Assume HxWxC uint8/RGB". A uint8 HxWx3 RGB array is passed through losslessly.
  - `PIL.Image`: `.convert("RGB")`.
- **The whole list is one multi-view sample**, not a batch. `InputProcessor.__call__` stacks to `(N,3,H,W)`, and `api.py: _prepare_model_inputs` adds `[None]`, giving `(B=1, N, 3, H, W)`.
  - "Batching" our images means passing N images in one call.
  - Images of different sizes in one call are **center-cropped to the smallest** (`InputProcessor._unify_batch_shapes`, which only logs a warning). Stage A must therefore batch by identical shape, as the spec says.
- Normalization: `T.ToTensor()` then ImageNet mean/std (`InputProcessor.NORMALIZE`).
- Precision: `api.py: DepthAnything3.forward` runs the backbone under `torch.autocast` with bf16 if supported, else fp16. `DepthAnything3Net.forward` runs the head with `autocast(enabled=False)`, i.e. fp32.
- For a single-image list, the passed `export_dir=None` means no export. `ref_view_strategy` is irrelevant (`alt_start=-1`).

### `upper_bound_resize` on 504×378 / 378×504

`InputProcessor._process_one` handles resizing:

1. `_resize_image` calls `_resize_longest_side(img, 504)`. **This returns `img` unchanged when `max(w, h) == 504`.**
2. `process_res_method.endswith("resize")` calls `_make_divisible_by_resize(img, 14)`. **This returns `img` unchanged when both dims are already multiples of 14.** 504 = 36·14 and 378 = 27·14.

So with `process_res=504` and `process_res_method="upper_bound_resize"`, our pre-resized input is not resampled at all. The only operations are the uint8 → PIL → `ToTensor` round trip (lossless) and normalization. The spec setting is correct; no change needed.

Stage A should still assert `tuple(prediction.processed_images.shape[1:3]) == (H_n, W_n)` and `prediction.depth.shape[1:] == (H_n, W_n)`.

### Output resolution

`model/dpt.py: DPT._forward_impl` upsamples to `h_out = (H//14)·14 / down_ratio` with `down_ratio = 1` (the default, not overridden in the yaml). Depth and sky therefore come back at exactly the processed resolution, `(N, H_n, W_n)`. No un-resize back to the original size happens anywhere in `inference()`.

### `Prediction` (`specs.py`)

```python
@dataclass
class Prediction:
    depth: np.ndarray            # N, H, W
    is_metric: int
    sky: np.ndarray | None       # N, H, W
    conf: np.ndarray | None      # N, H, W
    extrinsics / intrinsics: np.ndarray | None
    processed_images: np.ndarray | None   # N, H, W, 3 uint8 (de-normalized)
    gaussians, aux, scale_factor
```

Filled by `utils/io/output_processor.py: OutputProcessor.__call__`:

- `depth`: `model_output["depth"].squeeze(0).squeeze(-1)`, shape `(N, H, W)`, float32.
  - It is `exp(logits)` (`DPT(activation="exp")`), i.e. the **canonical depth `D`**.
  - Per `README.md` line 235: `metric_depth = focal * net_output / 300`, with focal in pixels at the processed resolution.
- `conf`: from `model_output["depth_conf"]`. **This is `None` for DA3METRIC-LARGE.** The head has `output_dim: 1`, so `DPT.has_conf = output_dim > 1 = False` and no confidence channel is produced.
- `sky`: **already thresholded**. `OutputProcessor._extract_sky` returns `sky.squeeze(0).cpu().numpy() >= 0.5`, a **bool** mask. See below.
- `is_metric`: `getattr(model_output, "is_metric", 0)`. Nothing sets it for this model, so it is 0. Ignore it.
- `extrinsics`/`intrinsics`: `None` (no camera decoder).

### Sky

- Head: `model/dpt.py: DPT.__init__(..., use_sky_head=True, sky_activation="relu")`, not overridden in the yaml. `DPT._apply_sky_activation` applies `torch.relu`.
  - **The raw sky output is a non-negative ReLU score, not a sigmoid probability and not logits.** It is unbounded above by construction. Its empirical range is in the smoke results.
  - DA3 itself treats it like a probability: it thresholds at 0.3 (`utils/alignment.py: compute_sky_mask`, used in `_process_mono_sky_estimation`) and at 0.5 (`OutputProcessor._extract_sky`).
- **Getting the continuous score**: `Prediction.sky` is bool, so Stage A must read the raw output. The supported decomposition of `inference()`, using DA3's own methods in `api.py`, is:
  ```python
  imgs_cpu, _, _ = model._preprocess_inputs(images, None, None, process_res, process_res_method)
  imgs, _, _     = model._prepare_model_inputs(imgs_cpu, None, None)
  raw            = model._run_model_forward(imgs, None, None, [], False, False, "saddle_balanced")
  raw["depth"]   # (1, N, H, W) float32, canonical D (after sky fill, see below)
  raw["sky"]     # (1, N, H, W) float32, ReLU score
  ```
  These are the same calls `inference()` makes, minus `OutputProcessor`, alignment (no-op without extrinsics) and export.
- **DA3 overwrites depth in the sky region.** `model/da3.py: DepthAnything3Net._process_mono_sky_estimation` runs inside `forward`, so it also affects `Prediction.depth`:
  - `non_sky = sky < 0.3`.
  - `non_sky_max = quantile(depth[non_sky], 0.99)`. This is computed **jointly over all N images in the call**, on a random subsample of 100 000 pixels (`torch.randint`) when there are more.
  - Then `depth[~non_sky] = non_sky_max`.
  - It is skipped if either set has ≤ 10 pixels.
  - Consequences:
    1. The sky depth fill value depends on batch composition and on the torch RNG. Non-sky pixels are unaffected.
    2. Pixels with `0.3 ≤ sky < 0.5` get the fill depth but are **not** in our `sky.threshold = 0.5` mask. They carry an artificial depth step into the detector.

### Confidence

None for DA3METRIC-LARGE (see above).

---

## GeoCalib

### Loading

`geocalib/extractor.py: GeoCalib(weights="pinhole")`. This downloads `https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar` via `torch.hub.load_state_dict_from_url` into `{torch.hub.get_dir()}/geocalib/pinhole.tar`, then calls `.to(device)`.

### `calibrate()` input

`extractor.py: GeoCalib.calibrate(img, camera_model="pinhole", priors=None, shared_intrinsics=False, camera_R_rig=None)`

- `img: torch.Tensor`, shape `(3,H,W)`, `(1,3,H,W)` or `(B,3,H,W)`. The docstring says "Assumes input image is in range [0, 1] and in RGB format". So the input is **float RGB in [0,1], `[3,H,W]`**, on the model device. `extractor.py: load_image` / `utils.py: load_image` produce exactly this from a file.
- Internally, `utils.py: ImagePreprocessor({"resize": 320, "edge_divisible_by": 32})`:
  - Resizes the **short side to 320** with kornia bilinear and antialiasing.
  - `scales = [w'/w, h'/h]` is one shape-`[2]` vector for the whole batch, so all images in a batch must share H, W.
  - Center-crops each dim down to a multiple of 32 (`fit_features_to_multiple(..., crop=True)`), recorded as `crop_pad` (negative).

### Camera returned in input-image pixels

`extractor.py: GeoCalib._post_process` calls `camera.undo_scale_crop(img_data)` (`camera.py: BaseCamera.undo_scale_crop`):

1. `self.crop(-crop_pad)` restores the size and shifts `c` back by half the crop. `f` is unchanged.
2. `.scale(1/scales)` multiplies `size`, `f` and `c` by `1/s`.

So `camera.f`, `camera.c` and `camera.size` are in **input-image pixel coordinates**. `focal_uncertainty` is also rescaled: `out["focal_uncertainty"] * inverse_scales[1]`, i.e. by `1/s_y`.

Principal-point convention: GeoCalib uses `c = (w/2, h/2)` (`camera.py: BaseCamera.from_dict` defaults), a corner-origin convention, not our `(W−1)/2`. We only consume `f`, so this has no effect. Do not mix GeoCalib's `c` into our K.

### Where the outputs live

`calibrate()` returns a dict with:

| quantity | expression | units / frame |
|---|---|---|
| `fx, fy` | `out["camera"].f[b]` (`camera.py: BaseCamera.f`, `_data[..., 2:4]`) | px, input image |
| `cx, cy` | `out["camera"].c[b]` | px, input image, corner convention |
| size | `out["camera"].size[b]`, (w, h) | px |
| `vfov` | `out["camera"].vfov[b]` (`focal2fov(fy, h)`) | rad |
| roll, pitch | `out["gravity"].rp[b]` (`gravity.py: Gravity.roll/pitch`) | rad |
| `focal_uncertainty` | `out["focal_uncertainty"][b]` | px, input image |
| `vfov_uncertainty` | `out["vfov_uncertainty"][b]` | rad |
| `roll_uncertainty`, `pitch_uncertainty`, `gravity_uncertainty` | `out[...]` | rad |
| `covariance` | `out["covariance"]` | the inverse LM Hessian in (roll, pitch, f) at the **320-px** resolution |
| perspective fields | `up_field`, `latitude_field`, `up_confidence`, `latitude_confidence` | upsampled to input size |

Details:

- Uncertainties come from `lm_optimizer.py: LMOptimizer.estimate_uncertainty`. It builds the Hessian with `as_rpf=True` (`log_focal=False`), so focal variance is in px² at the processed resolution.
- `focal_uncertainty = sqrt(focal_var)/2`. The `/2` is GeoCalib's own convention; we store it as returned.
- For pinhole, a single focal is optimized and `fx/fy` is preserved (`camera.py: update_focal`), so `fx == fy` in practice.

### Stochastic forward pass (found in M5)

- `geocalib/modules.py: NMF2D._build_bases` draws `torch.rand((B*S, D, R))` bases on every forward pass, inside the `LightHamHead` decoders. GeoCalib results therefore depend on the torch RNG state.
- Repeating the same image gives about ±1% spread in f (e.g. 079: 743.6–750.8 px), and this happens on CPU too.
- `cudnn.deterministic` and disabling TF32 do not help.
- `illum_baseline/calib.py` calls `torch.manual_seed(seed)` before each `calibrate()`. f is then bit-identical across calls and processes, and independent of processing order (`stage_a_version` 2).
- Averaging over several seeds would reduce this variance. It is not done; the spread is small next to the reported `focal_uncertainty` (4–24%).

### `shared_intrinsics=True`

- `calibrate(batch[B,3,H,W], shared_intrinsics=True)` sets `LMOptimizer.shared_intrinsics`.
- `lm_optimizer.py: setup_system_shared_intrinsics` builds one joint system: per-image gravity (2B params) plus **one shared focal**, Hessian `(1, 2B+1, 2B+1)`. The update `delta_f` is broadcast to all images, so `out["camera"].f` is identical across the batch.
- All images must have the same size, because `scales` is shared.
- ⚠ **The per-image uncertainties are wrong in this mode.** `estimate_uncertainty` falls into the non-rig `else` branch, which indexes the joint Cov as if it were per-image:
  - `roll_var = Cov[..., 0, 0]`, `pitch_var = Cov[..., 1, 1]` (image 0 only).
  - `focal_var = Cov[..., 2, 2]`, which is **image 1's roll variance**, not the focal.
  - Do not use `focal_uncertainty`/`vfov_uncertainty` from shared mode. If needed, compute the shared focal σ as `sqrt(out["covariance"][0, -1, -1]) / 2 / s_y`.
  - We only use shared mode as an optional cross-check of the pooled `f`, so this is not blocking.

---

## Environment

- uv project `geometry/pyproject.toml` with its own `.venv`. It is standalone, not a member of the root uv workspace, because DA3 pins `numpy<2` while the root venv has numpy 2.5.
- Always run as `env -u PYTHONPATH uv run ...`, because ROS Jazzy is on `PYTHONPATH`.
- Python 3.12.11, torch 2.14.1+cu130, CUDA available on an RTX 4090 Laptop GPU (16 GB), bf16 supported.
- Pins: `requirements.txt` (from `uv pip freeze`) and `uv.lock`.
- Install quirks:
  - `addict` is imported by `depth_anything_3/model/da3.py` but not declared in DA3's `pyproject.toml`. It is added explicitly.
  - `pyarrow<26`: pyarrow 26 raises `ImportError: pyarrow requires NumPy 2.0` at import. 25.0.1 works with numpy 1.26.4.
  - DA3 depends on `opencv-python`, the spec asks for `opencv-python-headless`, and both are installed (4.11.0.86). They share the `cv2` package directory, but the versions are identical, so this is harmless.
  - xFormers 0.0.35 warns that its C++/CUDA extensions were built for torch 2.10, so they are unavailable. **This does not affect DA3METRIC-LARGE**:
    - `model/dinov2/dinov2.py` uses `ffn_layer="mlp"` for `vitl`. Only `vitg` uses the xFormers `SwiGLU`.
    - Attention uses `F.scaled_dot_product_attention` (`model/dinov2/layers/attention.py`).
- Weights:
  - HF cache `~/.cache/huggingface/hub/models--depth-anything--DA3METRIC-LARGE`. A second, lowercase `...da3metric-large` copy from earlier work also exists; the cache key follows the repo-id casing.
  - GeoCalib: `~/.cache/torch/hub/geocalib/pinhole.tar`.

## Smoke results

`env -u PYTHONPATH uv run python scripts/smoke_m0.py [IMAGE]`. Default image: `DA3/Depth-Anything-3/assets/examples/SOH/000.png` (1208×680, outdoor, about 33% sky). The JSON is at `baseline_output/m0_smoke/smoke_m0.json`. The run gives 0 FAIL.

| check | result |
|---|---|
| `upper_bound_resize` no-op | Input 504×280 (sample) and random 504×378 / 378×504: processed tensor is the same shape, and max \|diff\| vs our own ImageNet normalization is **0**. |
| Output resolution | `Prediction.depth` is `(1, H_n, W_n)` float32. |
| conf | `None` |
| `Prediction.sky` | bool, and exactly `raw_sky >= 0.5` |
| raw sky score | Range [0, 1.21]. **14.8% of pixels are > 1**, so it is a ReLU score, not a probability. 0.4% of pixels fall in [0.3, 0.5). |
| sky depth fill | Every pixel with raw sky ≥ 0.3 has one constant depth (13.77). This is `_process_mono_sky_estimation`. |
| identical views in one call | Bit-identical outputs, so there is no cross-view coupling. |
| N=2 vs N=1 call (same image) | Non-sky relative depth difference: median 7.2e-4, p99 2.4e-3, **max 4.8e-2**. Repeating N=1 is bit-identical. |
| same, in fp32 (backbone without autocast) | median 2.6e-5, max 1.2e-3 |
| bf16 vs fp32 (N=1) | median 3.8e-3, p99 5.6e-3, max 2.5e-2 |
| GeoCalib `camera.size` | Equals input (W, H) to float32 precision, so the camera is in input-image pixels. |
| GeoCalib single | fx = fy = 1210.4 px, vfov 31.4°, roll −0.33°, pitch −0.36°, **focal_uncertainty 316 px (26%)**, vfov_uncertainty 0.19 rad |
| GeoCalib half-size input | f ratio 0.524 (expected ≈ 0.5; within the reported uncertainty). This also confirms input-pixel units. |
| GeoCalib `shared_intrinsics=True` (img + h-flip) | Shared f = 1212.4 for both images. Covariance `(1, 5, 5)`. Returned `focal_uncertainty` = 0.0209, which is ≈ a roll σ and **confirms the indexing bug above**. `vfov_uncertainty` = 0. |

### Flags (choices the spec does not cover). Lean in **bold**.

**Status: all leans below were confirmed by the user on 2026-10-26.** `da3.batch_size` is set to 1 in `configs/default.yaml`.

1. **Sky is a ReLU score, not a probability.** The spec's `sky_prob` with `threshold: 0.5` still makes sense, because DA3 itself thresholds this score at 0.5. **Lean: store the raw score clipped to [0, 1] as `sky_prob` float16 and record the unclipped max in the JSON.** Clipping is lossless for any threshold ≤ 1.
2. **Raw sky must come from the forward pass**, since `Prediction.sky` is bool. **Lean: use the `_preprocess_inputs` → `_prepare_model_inputs` → `_run_model_forward` decomposition above**, i.e. DA3's own private methods, and assert shapes. The alternative, monkeypatching `OutputProcessor._extract_sky`, is more fragile.
3. **DA3 fills sky depth (sky ≥ 0.3) with a constant**, and that constant depends on the batch. Our mask uses 0.5, so pixels in [0.3, 0.5) carry a fake depth step. **Lean: keep `sky.threshold: 0.5` as specified and record the [0.3, 0.5) fraction per image in the Stage A JSON**, so M3/M4 can see whether it matters. Alternative: set the threshold to 0.3 to match DA3.
4. **Batch composition changes depth at the ~1e-3 level, up to ~5% at isolated pixels**, through bf16 kernel numerics. This is not semantic coupling, but the detector's LoG of ρ is exactly sensitive to that noise scale (test 7 uses 1e-3·ρ). **Lean: run DA3 with batch size 1** (`da3.batch_size: 1`). Results are then reproducible and independent of how images are grouped, and the cost is small (504×378 on a 4090). Keep DA3's default bf16 autocast, since it is the model's own inference mode. Alternative: keep `batch_size: 8` and fix the batch order with the seed.
5. **GeoCalib shared-mode uncertainties are wrong.** **Lean: use shared mode only for the pooled-f cross-check and do not store its uncertainties.** If needed, derive σ_f from `covariance[0, -1, -1]`.
6. **GeoCalib per-image `focal_uncertainty` can be large** (26% on the sample). This affects metric depth and QA only, not the weights. No action needed; the spec's `focal_uncertainty / f > 0.15` flag covers it.
