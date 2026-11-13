"""Predict ISD maps with the ViT-12 linear model and compute log chromaticity images.

Usage: python run_isd_vit.py <image_dir> <out_dir> [--weights vit_12_linear.pth] [--anchor 10.4]
"""
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from log_chroma_utils import LogChromaUtils
from map_ViT_models import ViT_Patch2Patch_ver3

EXTS = {".tif", ".tiff"}


class ViTISD(ViT_Patch2Patch_ver3):
    """ver3 with _pad_to_multiple always returning (x, pad, (H, W))."""

    def _pad_to_multiple(self, x, m):
        _, _, h, w = x.shape
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, (0, pw, 0, ph), (h, w)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights, device):
    model = ViTISD(img_size=512, patch_size=16, in_ch=3, out_ch=3, embed_dim=768, depth=12, heads=8, dropout=0)
    ckpt = torch.load(weights, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return model.to(device).eval()


def predict_isd(model, rgb01, device):
    x = torch.from_numpy(rgb01.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    out = out[0].permute(1, 2, 0).float().cpu().numpy()
    isd = out / np.maximum(np.linalg.norm(out, axis=2, keepdims=True), 1e-8)
    g = isd.reshape(-1, 3).mean(axis=0)
    return isd.astype(np.float32), g / max(np.linalg.norm(g), 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--weights", default="vit_12_linear.pth")
    ap.add_argument("--anchor", type=float, default=10.4)
    args = ap.parse_args()

    files = sorted(p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"no TIFF images in {args.image_dir}")
    out = Path(args.out_dir)
    for sub in ("isd_npy", "isd_viz", "logchrom_npy", "logchrom_viz"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"device: {device}")
    model = load_model(args.weights, device)

    rows = []
    for i, path in enumerate(files, 1):
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16 or raw.ndim != 3:
            print(f"[{i}/{len(files)}] {path.name}: not a 16 bit color TIFF, skipping")
            continue
        rgb16 = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
        rgb01 = rgb16.astype(np.float32) / 65535.0

        isd, isd_global = predict_isd(model, rgb01, device)
        res = LogChromaUtils.get_log_chroma(image=rgb16, sr_map=np.clip(isd, 0, 1),
                                            anchor=args.anchor, normalize_sr_map=True)

        np.save(out / "isd_npy" / f"{path.stem}.npy", isd)
        np.save(out / "logchrom_npy" / f"{path.stem}.npy", res["projected_log_rgb"])
        isd_viz = (np.clip(isd, 0, 1) * 255 + 0.5).astype(np.uint8)
        cv2.imwrite(str(out / "isd_viz" / f"{path.stem}.png"), cv2.cvtColor(isd_viz, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "logchrom_viz" / f"{path.stem}.png"), cv2.cvtColor(res["chroma_8bit"], cv2.COLOR_RGB2BGR))
        rows.append({"image": path.stem, "isd_r": round(float(isd_global[0]), 4),
                     "isd_g": round(float(isd_global[1]), 4), "isd_b": round(float(isd_global[2]), 4)})
        print(f"[{i}/{len(files)}] {path.name}: global ISD {np.round(isd_global, 3)}")

    with open(out / "global_isd.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["image", "isd_r", "isd_g", "isd_b"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"done, results in {out}")


if __name__ == "__main__":
    main()
