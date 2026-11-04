"""Pinhole intrinsics at network resolution, upright frame, OpenCV camera axes (X right, Y down, Z forward)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    f_x: float
    f_y: float
    c_x: float
    c_y: float
    W: int
    H: int
    sigma_f_x: float = float("nan")  # focal uncertainty at network res (px); diagnostics only, never in a gate
    sigma_f_y: float = float("nan")

    @classmethod
    def centered(cls, f_x: float, f_y: float, W: int, H: int, sigma_f_x: float = float("nan"),
                 sigma_f_y: float = float("nan")) -> Intrinsics:
        """Principal point at the image centre ((W-1)/2, (H-1)/2), the pixel-centre convention GeoCalib assumes."""
        return cls(float(f_x), float(f_y), (W - 1) / 2, (H - 1) / 2, int(W), int(H), float(sigma_f_x),
                   float(sigma_f_y))

    @classmethod
    def from_stage_a(cls, meta_a: dict[str, Any]) -> Intrinsics:
        """Active focal from a Stage A JSON: f_x = fx s_x, f_y = fy s_y (fx, fy in upright full-res px).

        per_image: GeoCalib's fx, fy; pooled: f_pooled for both (as stage_a.update_focal sets f_active).
        sigma_f is GeoCalib's focal_uncertainty rescaled the same way.
        """
        net, gc, fo = meta_a["network"], meta_a["geocalib"], meta_a["focal"]
        if fo["mode"] == "per_image":
            fx, fy = gc["fx"], gc["fy"]
        elif fo["mode"] == "pooled":
            fx = fy = fo["f_pooled"]
        else:
            raise ValueError(f"unknown focal mode {fo['mode']!r}")
        unc = gc["focal_uncertainty"]
        return cls.centered(fx * net["s_x"], fy * net["s_y"], net["W_n"], net["H_n"],
                            unc * net["s_x"], unc * net["s_y"])

    def scaled(self, kappa: float) -> Intrinsics:
        """The same camera with f -> kappa f (principal point unchanged)."""
        return dataclasses.replace(self, f_x=kappa * self.f_x, f_y=kappa * self.f_y,
                                   sigma_f_x=kappa * self.sigma_f_x, sigma_f_y=kappa * self.sigma_f_y)

    def pixel_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """(u, v) float64 grids of shape (H, W): u is the column, v the row."""
        return np.meshgrid(np.arange(self.W, dtype=np.float64), np.arange(self.H, dtype=np.float64))

    def ray(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """r = K^-1 (u, v, 1) = ((u - c_x)/f_x, (v - c_y)/f_y, 1), shape (..., 3)."""
        u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
        return np.stack([(u - self.c_x) / self.f_x, (v - self.c_y) / self.f_y, np.ones_like(u)], axis=-1)

    def rays(self) -> np.ndarray:
        """Rays at every pixel, shape (H, W, 3)."""
        return self.ray(*self.pixel_grid())
