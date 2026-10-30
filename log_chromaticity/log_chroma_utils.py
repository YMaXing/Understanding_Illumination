from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import cv2


class LogChromaUtils:
    """
    Namespace for log-chromaticity projection operations.

    Assumptions
    -----------
    - Input image is a linear RGB image representing 16-bit intensity data.
    - Accepted input formats:
        * uint16 in [0, 65535]
        * float32/float64 in [0, 1]
    - SR / ISD map is expected to be shape (H, W, 3).
    - By default, SR / ISD vectors are unit-normalized internally before projection.
    """

    LOG_16BIT_MAX: float = float(np.log(65535.0))

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
        """
        Convert a 3-channel BGR image/map to RGB.
        Leaves non-3-channel arrays unchanged.
        """
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
        """
        Convert a 3-channel RGB image/map to BGR.
        Leaves non-3-channel arrays unchanged.
        """
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    @staticmethod
    def convert_16bit_to_8bit(image: np.ndarray) -> np.ndarray:
        """
        Convert uint16 or float [0,1] image to uint8 for visualization.
        """
        if image.dtype == np.uint16:
            img01 = image.astype(np.float32) / 65535.0
        else:
            img01 = np.clip(image.astype(np.float32), 0.0, 1.0)

        return (img01 * 255.0 + 0.5).astype(np.uint8)
    
    @staticmethod
    def to_float01(image: np.ndarray) -> np.ndarray:
        """
        Convert supported linear RGB image formats to float32 in [0, 1].
        """
        LogChromaUtils.validate_linear16_image(image)

        if image.dtype == np.uint16:
            out = image.astype(np.float32) / 65535.0
        else:
            out = image.astype(np.float32)

        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def float01_to_uint16(image: np.ndarray) -> np.ndarray:
        """
        Convert float image in [0,1] to uint16.
        """
        image = np.clip(image.astype(np.float32), 0.0, 1.0)
        return (image * 65535.0 + 0.5).astype(np.uint16)
    
    @staticmethod
    def load_tiff(
        path: str | Path,
        expected_shape_hw: tuple[int, int] | None = None,
        as_float: bool = True,
        as_rgb: bool = True
    ) -> np.ndarray:
        """
        Load a 16-bit TIFF image or SR/ISD map using OpenCV.

        Parameters
        ----------
        path : str | Path
            Path to the TIFF file.

        expected_shape_hw : tuple[int, int] | None, optional
            Expected spatial shape `(H, W)`. If provided, the loaded image must
            match this shape. This is primarily used when loading SR / ISD maps
            to ensure they match the dimensions of the corresponding image.

        as_float : bool, default=True
            If True, convert the image to `float32` in the range `[0, 1]`.
            If False, the original dtype (typically `uint16`) is preserved.

        as_rgb : bool, default=True
            If True, convert the image from OpenCV's default BGR channel order
            to RGB.

        Notes
        -----
        OpenCV loads images in **BGR channel order** by default. For standard
        RGB images this function typically converts the data to RGB.

        In this project, SR / ISD map TIFF files were written such that the
        raw OpenCV load already aligns with the intended channel semantics
        for projection. Therefore these maps should typically be loaded with
        `as_rgb=False` to avoid an unnecessary channel swap.

        Returns
        -------
        np.ndarray
            Image array of shape `(H, W, 3)` with either `uint16` or `float32`
            dtype depending on the `as_float` setting.
        """
        img_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read TIFF: {path}")
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image, got {img_bgr.shape}")
        if expected_shape_hw is not None and img_bgr.shape[:2] != expected_shape_hw:
            raise ValueError(
                f"Shape {img_bgr.shape[:2]} does not match expected shape {expected_shape_hw}"
            )
        if as_float:
            if img_bgr.dtype == np.uint16:
                img_bgr = img_bgr.astype(np.float32)
            img_bgr /= 65535.0
        if as_rgb:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_rgb
        return img_bgr

    # ------------------------------------------------------------------
    # Validation / format helpers
    # ------------------------------------------------------------------
    @staticmethod
    def validate_linear16_image(image: np.ndarray, name: str = "image") -> None:
        """
        Validate that the input looks like a linear 16-bit RGB image.

        Warns instead of failing for suspicious dtype/range issues unless shape is invalid.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(image)}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"{name} must have shape (H, W, 3). Got {image.shape}."
            )

        if image.dtype == np.uint16:
            return

        if np.issubdtype(image.dtype, np.floating):
            min_val = float(np.min(image))
            max_val = float(np.max(image))

            if min_val < 0.0 or max_val > 1.0:
                warnings.warn(
                    f"{name} is float dtype but has range [{min_val:.6f}, {max_val:.6f}]. "
                    "Expected float image in [0, 1] for linear 16-bit representation.",
                    stacklevel=2,
                )
            return

        warnings.warn(
            f"{name} has dtype {image.dtype}. Expected uint16 in [0,65535] "
            "or float in [0,1]. Attempting to continue.",
            stacklevel=2,
        )

    @staticmethod
    def validate_sr_map(sr_map: np.ndarray, image_shape: tuple[int, int]) -> None:
        """
        Validate sr map shape and range.
        """
        if not isinstance(sr_map, np.ndarray):
            raise TypeError(f"sr_map must be a numpy array, got {type(sr_map)}")

        if sr_map.ndim != 3 or sr_map.shape[2] != 3:
            raise ValueError(f"sr_map must have shape (H, W, 3). Got {sr_map.shape}")

        if sr_map.shape[:2] != image_shape:
            raise ValueError(
                f"sr_map shape {sr_map.shape[:2]} does not match image shape {image_shape}"
            )
        sr_max = np.max(sr_map)
        sr_min = np.min(sr_map)
        if sr_max > 1.0 or sr_min < 0.0:
            raise ValueError(
                f"sr_map values out of range [0,1]. "
                f"min={sr_min:.6f}, max={sr_max:.6f}"
            )

    # ------------------------------------------------------------------
    # Log / linear conversions
    # ------------------------------------------------------------------
    @staticmethod
    def convert_img_to_log_space(image: np.ndarray) -> np.ndarray:
        """
        Convert a linear 16-bit RGB image to log space.

        Input
        -----
        - uint16 in [0,65535], or
        - float in [0,1]

        Output
        ------
        float32 log image in natural log space, shape (H, W, 3)

        Zero masking is used: zero-valued pixels remain zero.
        """
        img01 = LogChromaUtils.to_float01(image)
        img16 = img01 * 65535.0

        log_img = np.zeros_like(img16, dtype=np.float32)
        mask = img16 > 0
        log_img[mask] = np.log(img16[mask])
        return log_img

    @staticmethod
    def log_to_linear(log_img: np.ndarray) -> np.ndarray:
        """
        Convert natural-log RGB values back to linear float image in [0,1].

        Zero masking is preserved.
        """
        x = log_img.astype(np.float32)

        lin16 = np.zeros_like(x, dtype=np.float32)
        mask = x > 0
        lin16[mask] = np.exp(x[mask])

        lin01 = np.clip(lin16 / 65535.0, 0.0, 1.0)
        return lin01

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def per_pixel_normalize(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Normalize each pixel vector independently.

        Uses sum normalization:
            rgb / (r + g + b)
        """
        image = image.astype(np.float32)
        denom = np.sum(image, axis=2, keepdims=True)
        return image / np.maximum(denom, eps)

    @staticmethod
    def normalize_vectors(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Unit-normalize a vector field of shape (H, W, 3).
        """
        n = np.linalg.norm(v, axis=2, keepdims=True)
        return v / np.maximum(n, eps)

    # ------------------------------------------------------------------
    # Core projection
    # ------------------------------------------------------------------
    @staticmethod
    def project_log_rgb_to_plane(
        image: np.ndarray,
        sr_map: np.ndarray,
        *,
        anchor: float = 10.4,
        normalize_sr_map: bool = True,
    ) -> np.ndarray:
        """
        Project each pixel's log RGB vector onto the plane orthogonal to the
        per-pixel SR / ISD vector.

        Parameters
        ----------
        image : np.ndarray
            Linear RGB image in uint16 [0,65535] or float [0,1].
        sr_map : np.ndarray
            Per-pixel SR / ISD vector map, shape (H, W, 3).
        anchor : float
            Log-space anchor / shift value.
        normalize_sr_map : bool
            If True, SR vectors are unit-normalized before projection.

        Returns
        -------
        projected_log_rgb : np.ndarray
            Log-space RGB image after projection, shape (H, W, 3).
        """
        LogChromaUtils.validate_linear16_image(image)
        LogChromaUtils.validate_sr_map(sr_map, image.shape[:2])

        log_img = LogChromaUtils.convert_img_to_log_space(image)
        shifted_log_rgb = log_img - anchor

        sr = sr_map.astype(np.float32)
        if normalize_sr_map:
            sr = LogChromaUtils.normalize_vectors(sr)

        # Per-pixel dot product: (v · n)
        dot_product_map = np.einsum("ijk,ijk->ij", shifted_log_rgb, sr)

        # Expand to (H, W, 1) for broadcasting
        dot_product_reshaped = dot_product_map[:, :, np.newaxis]

        # Vector projection: (v · n)n
        projection = dot_product_reshaped * sr

        # Plane projection: v - (v · n)n
        projected_rgb = shifted_log_rgb - projection

        # Shift back to original log coordinate frame
        projected_rgb += anchor
        return projected_rgb.astype(np.float32)

    @staticmethod
    def get_log_chroma(
        image: np.ndarray,
        sr_map: np.ndarray,
        *,
        anchor: float = 10.4,
        normalize_sr_map: bool = True,
        eps: float = 1e-8,
    ) -> dict[str, np.ndarray]:
        """
        Full log-chromaticity pipeline.

        Steps
        -----
        1. Convert image to log RGB
        2. Project onto plane orthogonal to SR / ISD
        3. Convert back to linear RGB
        4. Per-pixel chromaticity normalization
        5. Create 8-bit preview image

        Returns
        -------
        dict with keys:
            projected_log_rgb
            linear_chroma
            chroma_uint16
            chroma_8bit
        """
        projected_log_rgb = LogChromaUtils.project_log_rgb_to_plane(
            image=image,
            sr_map=sr_map,
            anchor=anchor,
            normalize_sr_map=normalize_sr_map,
        )

        linear_chroma = LogChromaUtils.log_to_linear(projected_log_rgb)
        linear_chroma = LogChromaUtils.per_pixel_normalize(linear_chroma, eps=eps)

        chroma_uint16 = LogChromaUtils.float01_to_uint16(linear_chroma)
        chroma_8bit = LogChromaUtils.convert_16bit_to_8bit(chroma_uint16)

        return {
            "projected_log_rgb": projected_log_rgb,
            "linear_chroma": linear_chroma,
            "chroma_uint16": chroma_uint16,
            "chroma_8bit": chroma_8bit,
        }
    
    # ------------------------------------------------------------------
    # Writing Results
    # ------------------------------------------------------------------

    @staticmethod
    def save_preview_png(
        path: str | Path,
        image: np.ndarray,
        *,
        assume_rgb: bool = True,
    ) -> None:
        """
        Save an 8-bit preview PNG.
        """

        if image.ndim == 3 and image.shape[2] == 3 and assume_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        ok = cv2.imwrite(str(path), image)
        if not ok:
            raise IOError(f"Failed to write preview PNG to {path}")
        else:
            print(f"Saved log-chroma png preview to: {path}")


def main():

    image_path = Path("data/submission_images/maddineni_poojit_045.tif")
    sr_map_path = Path("data/sr_maps/maddineni_poojit_045_isd.tiff")

    img_linear_rgb = LogChromaUtils.load_tiff(
                                path=image_path,
                                expected_shape_hw=None,
                                as_float=False,
                                as_rgb=True
                                )
    print("Image stats")
    print("min:", img_linear_rgb.min())
    print("max:", img_linear_rgb.max())
    img_shape = img_linear_rgb.shape[:2]
    print("shape", img_shape)
    print()

    sr_map_bgr = LogChromaUtils.load_tiff(
                            path=sr_map_path,
                            expected_shape_hw=img_shape,
                            as_float=True,
                            as_rgb=False
                            )
    print("SR map stats")
    print("min:", sr_map_bgr.min(axis=(0, 1)))
    print("max:", sr_map_bgr.max(axis=(0, 1)))
    print("shape", sr_map_bgr.shape[:2])
    print()


    results = LogChromaUtils.get_log_chroma(
                                image=img_linear_rgb,
                                sr_map=sr_map_bgr,
                                anchor=10.4,
                                normalize_sr_map=True,
                                eps=1e-8,
                                )
    
    linear_chroma = results["linear_chroma"]
    print("Linear chroma stats")
    print("min:", linear_chroma.min(axis=(0, 1)))
    print("max:", linear_chroma.max(axis=(0, 1)))
    print("shape", linear_chroma.shape[:2])
    print()

    
    chroma_8bit = results["chroma_8bit"]
    print("PNG chroma stats")
    print("min:", chroma_8bit.min(axis=(0, 1)))
    print("max:", chroma_8bit.max(axis=(0, 1)))
    print("shape", chroma_8bit.shape[:2])
    print()

    png_path = f"{image_path.stem}_8bit_chroma.png"
    chroma_8bit_bgr = cv2.cvtColor(chroma_8bit, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(png_path), chroma_8bit_bgr)
    if not ok:
        raise IOError(f"Failed to write preview PNG to {png_path}")
    else:
        print(f"Saved log-chroma png preview to: {png_path}")

if __name__ == "__main__":
    main()