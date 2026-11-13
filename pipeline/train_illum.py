"""Train a U-Net to predict log illumination with the self supervised losses in losses.py.

The network sees the log RGB image and predicts log illumination l (3 channels, natural log units).
Reflectance follows from the image: r = log I - l, so I = R x L exactly.

Illumination model (--illum):
  isd   (default) the network predicts ONE number a per pixel, and l = a * ISD(pixel). Illumination
        can then only change color along the ISD, as the bi-illuminant model says, so material colors
        cannot be absorbed into the illumination. Needs --isd-dir (isd_output/isd_npy).
  free  the network predicts all 3 channels of l freely (the earlier behavior)

Losses (no ground truth needed):
  IlluminationSmoothnessLoss  weights from build_smoothness_weights.py (smoothness_weights/npy)
  MaterialConstancyLoss       regions from cluster_materials.py       (materials/regions)
  MaterialBoundaryLoss        regions + smoothness weights
  PairConsistencyLoss         automatic lit and dark pairs across shadow edges (needs --labels-dir,
                              edge_labels/labels): the reflectance is equal on both sides

Evaluation (if --annotations is given), on every annotated image:
  for each pair of annotated points from the same region, the difference in log reflectance r
  (lower is better), compared with the difference in the log image itself (no decomposition).
  Reported for all same region pairs and for pairs that cross a shadow (brightness differs > 2x).

Outputs in <out_dir>:
  best.pth / last.pth          model weights (best = lowest validation loss)
  log.csv                      losses per epoch
  pred/<stem>_illum.png        predicted illumination (display)
  pred/<stem>_refl.png         predicted reflectance (display)
  pred/<stem>_illum.npy        predicted log illumination, float16
  eval.csv                     reflectance consistency on annotated images

Usage (from the project folder, with losses.py in the same folder as this script):
  python scripts/train_illum.py images smoothness_weights/npy materials/regions runs/isd \
      --isd-dir isd_output/isd_npy --labels-dir edge_labels/labels --annotations annotations_sanvidha
"""
import argparse
import csv
import os
import random
import sys
import time
from itertools import combinations
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent))
from losses import IlluminationSmoothnessLoss, MaterialConstancyLoss, MaterialBoundaryLoss, PairConsistencyLoss

LOG_MAX = 11.1  # log(65535) is about 11.09; inputs are divided by this to lie in [0, 1]


# ----------------------------------------------------------------------------------------------
# Model: ResNet18 encoder U-Net (same layout as ResNet18UNet in models_unet.py, without SE blocks)
# ----------------------------------------------------------------------------------------------
class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class IllumUNet(nn.Module):
    def __init__(self, pretrained=True, out_ch=3):
        super().__init__()
        try:
            r = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        except Exception as e:  # no internet: start from scratch
            print(f"could not load ImageNet weights ({e}); training from scratch")
            r = resnet18(weights=None)
        self.inc = nn.Sequential(r.conv1, r.bn1, r.relu)   # 64,  H/2
        self.pool = r.maxpool
        self.e1, self.e2, self.e3, self.e4 = r.layer1, r.layer2, r.layer3, r.layer4  # 64 H/4 .. 512 H/32
        self.u4 = Up(512, 256, 256)
        self.u3 = Up(256, 128, 128)
        self.u2 = Up(128, 64, 64)
        self.u1 = Up(64, 64, 32)
        self.u0 = Up(32, 0, 16)
        self.out = nn.Conv2d(16, out_ch, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.e1(self.pool(x0))
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        d = self.u4(x4, x3)
        d = self.u3(d, x2)
        d = self.u2(d, x1)
        d = self.u1(d, x0)
        d = self.u0(d)
        return self.out(d)


def to_log_illum(out, isd):
    """Network output -> log illumination. isd is None for the free model, else (B, 3, H, W) unit ISD."""
    return out if isd is None else out * isd


def predict_log_illum(model, log_img, device, isd=None):
    """log_img: (H, W, 3) natural log image; isd: (H, W, 3) or None. Returns (H, W, 3) log illumination.

    The losses only use differences, so the absolute level of the predicted illumination is arbitrary;
    only its changes across the image matter.
    """
    h, w = log_img.shape[:2]
    x = torch.from_numpy(log_img / LOG_MAX).permute(2, 0, 1)[None].float().to(device)
    ph, pw = (-h) % 32, (-w) % 32
    x = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
    with torch.no_grad():
        out = model(x)[:, :, :h, :w]
        isd_t = None if isd is None else torch.from_numpy(isd.astype(np.float32)).permute(2, 0, 1)[None].to(device)
        li = to_log_illum(out, isd_t)
    return li[0].permute(1, 2, 0).float().cpu().numpy()


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def make_pairs(log_img, regions, labels, dist=10, max_pairs=4000, min_ratio=1.3, seed=0):
    """Automatic lit and dark pairs across shadow edges.

    For shadow edge pixels (label 1), step `dist` pixels along the brightness gradient to the lit side
    and against it to the dark side. Keep the pair if both points are in the same material region, neither
    is a geometry edge, and the lit side is at least `min_ratio` times brighter.
    Returns (N, 4) int32: y_lit, x_lit, y_dark, x_dark.
    """
    h, w = labels.shape
    lum = cv2.GaussianBlur(log_img.mean(axis=2), (0, 0), 2.0)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    ys, xs = np.nonzero(labels == 1)
    if len(ys) == 0:
        return np.zeros((0, 4), np.int32)
    rng = np.random.default_rng(seed)
    if len(ys) > 4 * max_pairs:
        sel = rng.choice(len(ys), 4 * max_pairs, replace=False)
        ys, xs = ys[sel], xs[sel]
    n = np.hypot(gx[ys, xs], gy[ys, xs]) + 1e-8
    ny, nx = gy[ys, xs] / n, gx[ys, xs] / n
    yl, xl = np.rint(ys + dist * ny).astype(int), np.rint(xs + dist * nx).astype(int)
    yd, xd = np.rint(ys - dist * ny).astype(int), np.rint(xs - dist * nx).astype(int)
    ok = (yl >= 0) & (yl < h) & (xl >= 0) & (xl < w) & (yd >= 0) & (yd < h) & (xd >= 0) & (xd < w)
    yl, xl, yd, xd = yl[ok], xl[ok], yd[ok], xd[ok]
    keep = ((regions[yl, xl] == regions[yd, xd]) & (regions[yl, xl] > 0)
            & (labels[yl, xl] != 3) & (labels[yd, xd] != 3)
            & (lum[yl, xl] - lum[yd, xd] > np.log(min_ratio)))
    p = np.stack([yl[keep], xl[keep], yd[keep], xd[keep]], axis=1).astype(np.int32)
    if len(p) > max_pairs:
        p = p[rng.choice(len(p), max_pairs, replace=False)]
    return p


def load_item(image_dir, weights_dir, regions_dir, stem, isd_dir=None, labels_dir=None, pair_dist=10):
    p = next((image_dir / f"{stem}{e}" for e in (".tif", ".tiff") if (image_dir / f"{stem}{e}").exists()), None)
    wp, rp = weights_dir / f"{stem}.npy", regions_dir / f"{stem}.npy"
    if p is None or not wp.exists() or not rp.exists():
        return None
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
    log_img = np.log(np.maximum(rgb, 1.0)).astype(np.float32)
    item = {"stem": stem, "log": log_img, "w": np.load(wp).astype(np.float32), "reg": np.load(rp).astype(np.int64),
            "isd": None, "pairs": np.zeros((0, 4), np.int32)}
    if isd_dir is not None:
        ip = isd_dir / f"{stem}.npy"
        if not ip.exists():
            return None
        isd = np.load(ip).astype(np.float32)
        item["isd"] = (isd / np.maximum(np.linalg.norm(isd, axis=2, keepdims=True), 1e-8)).astype(np.float16)
    if labels_dir is not None and (labels_dir / f"{stem}.npy").exists():
        item["pairs"] = make_pairs(log_img, item["reg"], np.load(labels_dir / f"{stem}.npy"), dist=pair_dist)
    return item


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, items, crop, length, train=True, max_pairs=512):
        self.items, self.crop, self.length, self.train, self.max_pairs = items, crop, length, train, max_pairs

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = random if self.train else random.Random(i)
        it = self.items[rng.randrange(len(self.items))] if self.train else self.items[i % len(self.items)]
        h, w = it["log"].shape[:2]
        c = min(self.crop, h, w) // 32 * 32
        y0, x0 = rng.randint(0, h - c), rng.randint(0, w - c)
        if not self.train:
            y0, x0 = (h - c) // 2, (w - c) // 2
        sl = (slice(y0, y0 + c), slice(x0, x0 + c))
        log_img, wt, reg = it["log"][sl], it["w"][sl], it["reg"][sl]
        isd = it["isd"][sl].astype(np.float32) if it["isd"] is not None else np.zeros((c, c, 3), np.float32)

        # pairs fully inside the crop, shifted to crop coordinates
        p = it["pairs"].astype(np.int64)
        if len(p):
            inside = ((p[:, 0] >= y0) & (p[:, 0] < y0 + c) & (p[:, 2] >= y0) & (p[:, 2] < y0 + c)
                      & (p[:, 1] >= x0) & (p[:, 1] < x0 + c) & (p[:, 3] >= x0) & (p[:, 3] < x0 + c))
            p = p[inside] - np.array([y0, x0, y0, x0])
        flip = self.train and rng.random() < 0.5
        if flip:
            log_img, wt, reg, isd = log_img[:, ::-1], wt[:, ::-1], reg[:, ::-1], isd[:, ::-1]
            if len(p):
                p[:, 1], p[:, 3] = c - 1 - p[:, 1], c - 1 - p[:, 3]
        pairs = np.zeros((self.max_pairs, 4), np.int64)
        mask = np.zeros(self.max_pairs, np.float32)
        if len(p):
            if len(p) > self.max_pairs:
                p = p[np.array(rng.sample(range(len(p)), self.max_pairs))]
            pairs[:len(p)], mask[:len(p)] = p, 1.0
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        return t(log_img).permute(2, 0, 1), t(wt), t(reg), t(isd).permute(2, 0, 1), t(pairs), t(mask)


# ----------------------------------------------------------------------------------------------
# Evaluation against point annotations
# ----------------------------------------------------------------------------------------------
def patch_mean(a, x, y, r=2):
    return a[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1].reshape(-1, a.shape[2]).mean(axis=0)


def evaluate_pairs(log_img, log_illum, csv_path):
    """Pairwise checks on the annotated points (5 x 5 patch means, log units, L2 over RGB).

    img_all / refl_all       same region pairs: difference in the image vs in the reflectance (lower is better)
    img_cross / refl_cross   same region pairs that cross a shadow (brightness differs > 2x)
    img_diff / refl_diff     different region pairs: should STAY large; guards against a flat reflectance
    metric2                  for shadow crossing pairs: the reflectance of the dark point should equal the
                             image color of the lit point. Reflectance has an arbitrary global level, so it is
                             aligned first by the median offset between reflectance and image at lit points.
    """
    refl = log_img - log_illum
    pts = []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if row["label"] == "specular":
                continue
            x, y = int(row["x"]), int(row["y"])
            if 0 <= y < log_img.shape[0] and 0 <= x < log_img.shape[1]:
                pts.append((int(row["region_id"]), patch_mean(log_img, x, y), patch_mean(refl, x, y)))
    res = {k: [] for k in ("img_all", "refl_all", "img_cross", "refl_cross", "img_diff", "refl_diff")}
    lit_dark = []
    for (ra, ia, fa), (rb, ib, fb) in combinations(pts, 2):
        di, dr = np.linalg.norm(ia - ib), np.linalg.norm(fa - fb)
        if ra != rb:
            res["img_diff"].append(di)
            res["refl_diff"].append(dr)
            continue
        res["img_all"].append(di)
        res["refl_all"].append(dr)
        if abs(ia.mean() - ib.mean()) > np.log(2):  # crosses a shadow
            res["img_cross"].append(di)
            res["refl_cross"].append(dr)
            lit, dark = ((ia, fa), (ib, fb)) if ia.mean() > ib.mean() else ((ib, fb), (ia, fa))
            lit_dark.append((lit, dark))
    out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in res.items()}
    if lit_dark:
        offset = np.median(np.stack([l[1] - l[0] for l, _ in lit_dark]), axis=0)  # refl - image at lit points
        out["metric2"] = float(np.mean([np.linalg.norm(d[1] - offset - l[0]) for l, d in lit_dark]))
    else:
        out["metric2"] = float("nan")
    out["cross_pairs"] = len(lit_dark)
    return out


def to_display(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    y = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) ** (1 / 2.2)
    return cv2.cvtColor((y * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("weights_dir", help="smoothness_weights/npy")
    ap.add_argument("regions_dir", help="materials/regions")
    ap.add_argument("out_dir")
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=50, help="training steps per epoch")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam-smooth", type=float, default=1.0)
    ap.add_argument("--lam-const", type=float, default=1.0)
    ap.add_argument("--lam-bound", type=float, default=1.0)
    ap.add_argument("--lam-pair", type=float, default=1.0)
    ap.add_argument("--illum", choices=["isd", "free"], default="isd")
    ap.add_argument("--isd-dir", default=None, help="isd_output/isd_npy (required for --illum isd)")
    ap.add_argument("--labels-dir", default=None, help="edge_labels/labels, for the automatic lit and dark pairs")
    ap.add_argument("--pair-dist", type=int, default=10, help="pixels from the shadow edge to each pair point")
    ap.add_argument("--val", type=int, default=5, help="images held out for the validation loss")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true", help="skip training; load best.pth from out_dir and evaluate")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    image_dir, out = Path(args.image_dir), Path(args.out_dir)
    (out / "pred").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    stems = sorted(p.stem for p in Path(args.regions_dir).glob("*.npy"))
    if args.illum == "isd" and not args.isd_dir:
        raise SystemExit("--illum isd needs --isd-dir isd_output/isd_npy")
    isd_dir = Path(args.isd_dir) if args.isd_dir else None
    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    items = [it for s in stems if (it := load_item(image_dir, Path(args.weights_dir), Path(args.regions_dir), s,
                                                   isd_dir if args.illum == "isd" else None, labels_dir, args.pair_dist))]
    if labels_dir is not None:
        print(f"automatic lit and dark pairs: {sum(len(it['pairs']) for it in items)} over {len(items)} images")
    if len(items) <= args.val:
        raise SystemExit(f"only {len(items)} complete images found; need more than --val {args.val}")
    ann = Path(args.annotations) if args.annotations else None
    def has_points(p):  # the annotation tool writes a header only CSV for images visited but not annotated
        if not p.exists():
            return False
        with open(p) as fh:
            return sum(1 for _ in csv.DictReader(fh)) >= 2
    annotated = {it["stem"] for it in items if ann and has_points(ann / f"{it['stem']}.csv")}
    # keep annotated images out of training, so the evaluation is on unseen images
    pool = [it for it in items if it["stem"] not in annotated]
    random.shuffle(pool)
    val_items, train_items = pool[:args.val], pool[args.val:]
    print(f"{len(train_items)} training images, {len(val_items)} validation images, {len(annotated)} annotated test images")

    train_dl = torch.utils.data.DataLoader(CropDataset(train_items, args.crop, args.steps * args.batch), batch_size=args.batch, shuffle=False)
    val_dl = torch.utils.data.DataLoader(CropDataset(val_items, args.crop, len(val_items), train=False), batch_size=args.batch)

    model = IllumUNet(pretrained=not args.no_pretrained, out_ch=1 if args.illum == "isd" else 3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    L_s, L_c, L_b, L_p = IlluminationSmoothnessLoss(), MaterialConstancyLoss(), MaterialBoundaryLoss(), PairConsistencyLoss()

    def step(batch):
        log_img, wt, reg, isd, pairs, pmask = (t.to(device) for t in batch)
        log_illum = to_log_illum(model(log_img / LOG_MAX), isd if args.illum == "isd" else None)
        ls, lc, lb = L_s(log_illum, wt), L_c(log_illum, log_img, reg), L_b(log_illum, reg, wt)
        lp = L_p(log_illum, log_img, pairs, pmask) if args.lam_pair > 0 else log_illum.new_zeros(())
        total = args.lam_smooth * ls + args.lam_const * lc + args.lam_bound * lb + args.lam_pair * lp
        return total, (ls.item(), lc.item(), lb.item(), lp.item())

    best = float("inf")
    with open(out / "log.csv", "a" if args.eval_only else "w", newline="") as fh:
        wr = csv.writer(fh)
        if not args.eval_only:
            wr.writerow(["epoch", "train_total", "smooth", "constancy", "boundary", "pair", "val_total", "seconds"])
        for ep in range(1, 0 if args.eval_only else args.epochs + 1):
            t0 = time.time()
            model.train()
            parts, totals = np.zeros(4), []
            for batch in train_dl:
                loss, p = step(batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                totals.append(loss.item())
                parts += p
            sched.step()
            model.eval()
            with torch.no_grad():
                val = float(np.mean([step(b)[0].item() for b in val_dl]))
            parts /= max(len(totals), 1)
            wr.writerow([ep, round(np.mean(totals), 5), *np.round(parts, 5), round(val, 5), round(time.time() - t0, 1)])
            fh.flush()
            print(f"epoch {ep:3d}  train {np.mean(totals):.4f} (smooth {parts[0]:.4f}  const {parts[1]:.4f}  bound {parts[2]:.4f}  pair {parts[3]:.4f})"
                  f"  val {val:.4f}  {time.time() - t0:.0f}s")
            if val < best:
                best = val
                torch.save({"model_state_dict": model.state_dict(), "epoch": ep}, out / "best.pth")
    if not args.eval_only:
        torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs}, out / "last.pth")

    # predictions for every image with the best weights, and evaluation on the annotated ones
    model.load_state_dict(torch.load(out / "best.pth", map_location=device)["model_state_dict"])
    model.eval()
    rows = []
    for it in items:
        li = predict_log_illum(model, it["log"], device, it["isd"] if args.illum == "isd" else None)
        np.save(out / "pred" / f"{it['stem']}_illum.npy", li.astype(np.float16))
        cv2.imwrite(str(out / "pred" / f"{it['stem']}_illum.png"), to_display(li))
        cv2.imwrite(str(out / "pred" / f"{it['stem']}_refl.png"), to_display(it["log"] - li))
        if it["stem"] in annotated:
            r = evaluate_pairs(it["log"], li, ann / f"{it['stem']}.csv")
            rows.append({"image": it["stem"], **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}})
            print(f"{it['stem']}: same material {r['img_all']:.3f} -> {r['refl_all']:.3f} | across shadows "
                  f"{r['img_cross']:.3f} -> {r['refl_cross']:.3f} ({r['cross_pairs']} pairs) | different materials "
                  f"{r['img_diff']:.3f} -> {r['refl_diff']:.3f} | metric 2: {r['metric2']:.3f}")
    if rows:
        with open(out / "eval.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    print(f"done, results in {out}")


if __name__ == "__main__":
    main()
