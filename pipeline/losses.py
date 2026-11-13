"""Loss functions for training the illumination network.

Convention: the network predicts log illumination  l = log L  (B, C, H, W).
Log reflectance then follows from the image:      r = log I - l
so I = R x L holds exactly, and predicting illumination or reflectance is the same thing.

Implemented so far:
  IlluminationSmoothnessLoss  illumination changes smoothly, except at shadow and geometry edges
  MaterialConstancyLoss       within a material region, all change is illumination (constant reflectance)
  MaterialBoundaryLoss        at material boundaries away from shadow and geometry edges, all change is
                              material (illumination continuous across the boundary)
  PairConsistencyLoss         for lit and dark pairs on the same material across a shadow edge, the
                              reflectance is equal (all of the change is illumination)
"""
import torch
import torch.nn as nn


class IlluminationSmoothnessLoss(nn.Module):
    """Weighted smoothness of log illumination.

        loss = sum_p w(p) * rho(grad l(p)) / sum_p w(p)

    w comes from build_smoothness_weights.py: 1 where illumination must be smooth, small at shadow
    and geometry edges where it may change. For a difference between two neighboring pixels the
    smaller of their two weights is used, so an edge releases the pair on either side of it.

    penalty:
      "l2"         squared differences; strongly prefers smooth illumination
      "charbonnier" sqrt(d^2 + eps^2); behaves like L1 for large differences, so it tolerates
                   a few sharp changes the weight map missed
    """

    def __init__(self, penalty="charbonnier", eps=1e-3):
        super().__init__()
        if penalty not in ("l2", "charbonnier"):
            raise ValueError("penalty must be 'l2' or 'charbonnier'")
        self.penalty = penalty
        self.eps = eps

    def _rho(self, d):
        sq = d.pow(2).sum(dim=1, keepdim=True)  # sum over color channels
        if self.penalty == "l2":
            return sq
        return torch.sqrt(sq + self.eps ** 2) - self.eps

    def forward(self, log_illum, weight):
        """log_illum: (B, C, H, W); weight: (B, 1, H, W) or (B, H, W), values in [0, 1]."""
        if weight.dim() == 3:
            weight = weight.unsqueeze(1)
        weight = weight.to(log_illum.dtype)

        dx = log_illum[..., :, 1:] - log_illum[..., :, :-1]
        dy = log_illum[..., 1:, :] - log_illum[..., :-1, :]
        wx = torch.minimum(weight[..., :, 1:], weight[..., :, :-1])
        wy = torch.minimum(weight[..., 1:, :], weight[..., :-1, :])

        num = (wx * self._rho(dx)).sum() + (wy * self._rho(dy)).sum()
        den = wx.sum() + wy.sum()
        return num / den.clamp_min(1.0)


def _charbonnier(d, eps):
    """Robust penalty on a (C, ...) difference: sqrt(sum_c d^2 + eps^2) - eps."""
    return torch.sqrt(d.pow(2).sum(dim=0) + eps ** 2) - eps


class MaterialConstancyLoss(nn.Module):
    """Within each material region, reflectance should be constant.

    r = log I - l. For every region (from cluster_materials.py, regions/<stem>.npy; 0 = invalid, ignored),
    penalize how far each pixel's r is from the region's mean r. The mean is computed from the current
    prediction, so the loss pushes all variation inside a region into the illumination.
    Uses a robust penalty, because real materials have some texture.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, log_illum, log_image, regions):
        """log_illum, log_image: (B, C, H, W); regions: (B, H, W) integer region ids."""
        r = log_image - log_illum
        B, C = r.shape[:2]
        total = r.new_zeros(())
        count = 0
        for b in range(B):
            reg = regions[b].reshape(-1).long()
            rb = r[b].reshape(C, -1)
            keep = reg > 0
            reg, rb = reg[keep], rb[:, keep]
            if reg.numel() == 0:
                continue
            K = int(reg.max().item()) + 1
            sums = rb.new_zeros(C, K).index_add_(1, reg, rb)
            cnt = rb.new_zeros(K).index_add_(0, reg, torch.ones_like(reg, dtype=rb.dtype))
            mean = sums / cnt.clamp_min(1.0)
            total = total + _charbonnier(rb - mean[:, reg], self.eps).sum()
            count += reg.numel()
        return total / max(count, 1)


class MaterialBoundaryLoss(nn.Module):
    """At a material boundary that is not also a shadow or geometry edge, illumination is continuous.

    For every pair of neighboring pixels in different material regions (both valid), where the
    smoothness weight of both pixels is above `release` (i.e. no shadow or geometry edge there),
    penalize the difference in log illumination. All of the change at such a boundary must then be
    explained by reflectance.
    """

    def __init__(self, release=0.5, eps=1e-3):
        super().__init__()
        self.release = release
        self.eps = eps

    def forward(self, log_illum, regions, weight):
        """log_illum: (B, C, H, W); regions: (B, H, W); weight: (B, H, W) or (B, 1, H, W) smoothness weights."""
        if weight.dim() == 4:
            weight = weight[:, 0]
        reg = regions.long()
        total = log_illum.new_zeros(())
        count = log_illum.new_zeros(())
        for a, b, la, lb, wa, wb in (
            (reg[:, :, 1:], reg[:, :, :-1], log_illum[..., :, 1:], log_illum[..., :, :-1], weight[:, :, 1:], weight[:, :, :-1]),
            (reg[:, 1:, :], reg[:, :-1, :], log_illum[..., 1:, :], log_illum[..., :-1, :], weight[:, 1:, :], weight[:, :-1, :]),
        ):
            pair = (a != b) & (a > 0) & (b > 0) & (torch.minimum(wa, wb) > self.release)
            d = (la - lb).permute(1, 0, 2, 3)              # (C, B, H, W')
            pen = _charbonnier(d, self.eps)                # (B, H, W')
            total = total + (pen * pair).sum()
            count = count + pair.sum()
        return total / count.clamp_min(1.0)


class PairConsistencyLoss(nn.Module):
    """Lit and dark pairs: the same material on both sides of a shadow edge has the same reflectance.

        r_lit = r_dark   <=>   (log I_lit - log I_dark) = (l_lit - l_dark)

    so the whole change across the pair must be explained by the illumination.
    pairs: (B, K, 4) integer pixel coordinates (y_lit, x_lit, y_dark, x_dark); mask: (B, K), 1 for real pairs.
    This is metric 1 used as a training loss; with the global level aligned it also gives metric 2.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, log_illum, log_image, pairs, mask):
        B, C, H, W = log_illum.shape
        refl = (log_image - log_illum).flatten(2)                      # (B, C, H*W)
        idx_lit = (pairs[..., 0] * W + pairs[..., 1]).long()            # (B, K)
        idx_dark = (pairs[..., 2] * W + pairs[..., 3]).long()
        r_lit = refl.gather(2, idx_lit.unsqueeze(1).expand(B, C, -1))   # (B, C, K)
        r_dark = refl.gather(2, idx_dark.unsqueeze(1).expand(B, C, -1))
        d = (r_lit - r_dark).permute(1, 0, 2)                          # (C, B, K)
        pen = _charbonnier(d, self.eps)                                # (B, K)
        m = mask.to(pen.dtype)
        return (pen * m).sum() / m.sum().clamp_min(1.0)


if __name__ == "__main__":
    # quick self check: a step in illumination costs a lot when the weight is 1 there, little when released
    torch.manual_seed(0)
    h, w = 32, 32
    illum = torch.zeros(1, 3, h, w)
    illum[..., :, 16:] = 1.0                      # a sharp illumination change at column 16
    w_all = torch.ones(1, 1, h, w)
    w_rel = w_all.clone()
    w_rel[..., :, 15:17] = 0.05                   # the edge is released
    loss = IlluminationSmoothnessLoss()
    print("step, smoothness enforced everywhere:", round(loss(illum, w_all).item(), 4))
    print("step, edge released:                 ", round(loss(illum, w_rel).item(), 4))
    print("flat illumination:                   ", round(loss(torch.zeros(1, 3, h, w), w_all).item(), 4))

    # material losses: two regions, left and right; the image steps up by 1.0 at the boundary
    regions = torch.ones(1, h, w, dtype=torch.long)
    regions[:, :, 16:] = 2
    log_image = torch.zeros(1, 3, h, w)
    log_image[..., :, 16:] = 1.0
    const, bound = MaterialConstancyLoss(), MaterialBoundaryLoss()
    good = torch.zeros(1, 3, h, w)                # flat illumination: the step is reflectance
    bad = log_image.clone()                       # illumination takes the step: wrong at a material boundary
    print("constancy, flat illumination (good): ", round(const(good, log_image, regions).item(), 4))
    print("boundary,  flat illumination (good): ", round(bound(good, regions, w_all[:, 0]).item(), 4))
    print("boundary,  illumination steps (bad): ", round(bound(bad, regions, w_all[:, 0]).item(), 4))
    noisy = good + 0.2 * torch.randn(1, 3, h, w)  # reflectance varies inside regions
    print("constancy, noisy illumination (bad): ", round(const(noisy, log_image, regions).item(), 4))

    # pair loss: the image is 1.0 darker in a shadow on the left half's bottom; one lit and dark pair across it
    img2 = torch.zeros(1, 3, h, w)
    img2[..., 16:, :] -= 1.0
    pairs = torch.tensor([[[10, 5, 20, 5]]])
    pmask = torch.ones(1, 1)
    pair = PairConsistencyLoss()
    shadow_in_illum = img2.clone()                # illumination carries the shadow (good)
    print("pair, shadow in illumination (good): ", round(pair(shadow_in_illum, img2, pairs, pmask).item(), 4))
    print("pair, flat illumination (bad):       ", round(pair(torch.zeros(1, 3, h, w), img2, pairs, pmask).item(), 4))
