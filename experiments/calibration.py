"""
calibration.py
──────────────
Manages the affine mapping between camera pixel space and the MakeMeAHanzi
coordinate space (900 × 900 SVG units, Y increasing upward).

Usage
─────
    calib = Calibration()

    # User pinches two opposite corners of their intended drawing area:
    calib.add_corner((px1, py1))
    calib.add_corner((px2, py2))

    if calib.ready:
        # Convert a hanzi-space point to screen pixels:
        sx, sy = calib.hanzi_to_screen(450, 450)

        # Convert a screen point back to hanzi space:
        hx, hy = calib.screen_to_hanzi(sx, sy)

        # Scale a hanzi-space length (e.g. stroke arc-length) to pixels:
        px_len = calib.hanzi_len_to_px(some_hanzi_length)

        # Scale a pixel length back to hanzi units:
        hz_len = calib.px_len_to_hanzi(some_pixel_length)

Coordinate systems
──────────────────
    Hanzi  space : x ∈ [0, 900], y ∈ [0, 900], y increases UPWARD  (SVG)
    Screen space : x ∈ [0, W],   y ∈ [0, H],   y increases DOWNWARD (OpenCV)

The transform is:

    screen_x = origin_x + scale * hanzi_x
    screen_y = origin_y - scale * hanzi_y      ← Y flip

where `scale` (pixels / hanzi-unit) is computed from the bounding box the
user draws, and `origin_x / origin_y` centre the 900-unit square inside it.

A single uniform scale is used for both axes so the character is never
distorted — the mapping is similarity (scale + translate + Y-flip), not
a full affine warp.
"""

from __future__ import annotations

import numpy as np


# The MakeMeAHanzi SVG canvas is always 900 × 900 units.
HANZI_CANVAS = 900.0


class Calibration:
    """
    Similarity transform:  hanzi-space  <-->  pixel-space.

    Attributes (read-only after ready == True)
    ──────────────────────────────────────────
    ready       : bool   – True once both corners have been registered.
    scale       : float  – pixels per hanzi unit (uniform, both axes).
    inv_scale   : float  – hanzi units per pixel  (1 / scale).
    corners     : list   – the two raw pixel corners supplied by the user.
    """

    def __init__(self) -> None:
        self.corners: list[tuple[int, int]] = []
        self.ready     = False
        self.scale     = 1.0      # px / hanzi-unit
        self.inv_scale = 1.0      # hanzi-unit / px
        self._origin_x = 0.0     # pixel x that corresponds to hanzi x = 0
        self._origin_y = 0.0     # pixel y that corresponds to hanzi y = 0

    # ── Corner registration ───────────────────────────────────────────────

    def add_corner(self, pt: tuple[int, int]) -> None:
        """
        Register one corner of the user's drawing bounding box.
        Call twice with opposite corners; the transform is built on the
        second call and `ready` becomes True.
        """
        self.corners.append(pt)
        if len(self.corners) == 2:
            self._build()

    def reset(self) -> None:
        """Clear calibration so the user can re-draw the bounding box."""
        self.__init__()

    # ── Transform build ───────────────────────────────────────────────────

    def _build(self) -> None:
        """
        Derive scale and origin from the two registered corner pixels.

        Scale
        ─────
        The hanzi canvas is HANZI_CANVAS units square.  We fit it inside
        the pixel bounding box while preserving aspect ratio (uniform scale):

            scale = min(box_pixel_width, box_pixel_height) / HANZI_CANVAS

        This means:

            1 hanzi unit  =  scale pixels
            1 pixel       =  1/scale hanzi units

        Origin
        ──────
        We centre the 900-unit square inside the drawn box, so neither
        axis is flush with the box edge if the box is not square.

            origin_x = centre_x - scale * HANZI_CANVAS / 2
            origin_y = centre_y + scale * HANZI_CANVAS / 2
                                        ↑ positive because Y is flipped
        """
        (x0, y0), (x1, y1) = self.corners
        px_min, px_max = min(x0, x1), max(x0, x1)
        py_min, py_max = min(y0, y1), max(y0, y1)

        box_w = max(px_max - px_min, 1)
        box_h = max(py_max - py_min, 1)

        # ── The key ratio: pixels per hanzi unit ──────────────────────────
        self.scale     = min(box_w, box_h) / HANZI_CANVAS
        self.inv_scale = 1.0 / self.scale

        cx = (px_min + px_max) / 2.0
        cy = (py_min + py_max) / 2.0

        # pixel position of the hanzi origin (0, 0)
        # hanzi (0,0) is bottom-left → screen bottom-centre of the box
        self._origin_x = cx - self.scale * HANZI_CANVAS / 2.0
        self._origin_y = cy + self.scale * HANZI_CANVAS / 2.0

        self.ready = True

    # ── Point transforms ──────────────────────────────────────────────────

    def hanzi_to_screen(self, hx: float, hy: float) -> tuple[int, int]:
        """
        Map a single hanzi-space point (hx, hy) to pixel coordinates.

        screen_x = origin_x + scale * hx
        screen_y = origin_y - scale * hy   (Y flip)
        """
        sx = self._origin_x + self.scale * hx
        sy = self._origin_y - self.scale * hy
        return int(round(sx)), int(round(sy))

    def screen_to_hanzi(self, sx: float, sy: float) -> tuple[float, float]:
        """
        Inverse map: pixel (sx, sy) → hanzi-space (hx, hy).

        hx = (screen_x - origin_x) / scale
        hy = (origin_y - screen_y) / scale   (Y flip)
        """
        hx = (sx - self._origin_x) * self.inv_scale
        hy = (self._origin_y - sy) * self.inv_scale
        return hx, hy

    # ── Array transforms (vectorised, no Python loop) ─────────────────────

    def hanzi_to_screen_array(self, pts: np.ndarray) -> np.ndarray:
        """
        Vectorised hanzi → screen.

        Parameters
        ──────────
        pts : ndarray, shape (N, 2), columns [hanzi_x, hanzi_y]

        Returns
        ───────
        ndarray, shape (N, 2), dtype float32, columns [screen_x, screen_y]
        """
        pts = np.asarray(pts, dtype=np.float32)
        sx  = self._origin_x + self.scale * pts[:, 0]
        sy  = self._origin_y - self.scale * pts[:, 1]
        return np.stack([sx, sy], axis=1)

    def screen_to_hanzi_array(self, pts: np.ndarray) -> np.ndarray:
        """
        Vectorised screen → hanzi.

        Parameters
        ──────────
        pts : ndarray, shape (N, 2), columns [screen_x, screen_y]

        Returns
        ───────
        ndarray, shape (N, 2), dtype float32, columns [hanzi_x, hanzi_y]
        """
        pts = np.asarray(pts, dtype=np.float32)
        hx  = (pts[:, 0] - self._origin_x) * self.inv_scale
        hy  = (self._origin_y - pts[:, 1]) * self.inv_scale
        return np.stack([hx, hy], axis=1)

    # ── Length / distance scaling ─────────────────────────────────────────

    def hanzi_len_to_px(self, hanzi_length: float) -> float:
        """
        Scale a length (or distance) from hanzi units to pixels.

        Because the transform is uniform (same scale on both axes), lengths
        scale linearly:  pixel_length = scale * hanzi_length
        """
        return self.scale * hanzi_length

    def px_len_to_hanzi(self, pixel_length: float) -> float:
        """
        Scale a length (or distance) from pixels to hanzi units.

        hanzi_length = pixel_length / scale  =  pixel_length * inv_scale
        """
        return self.inv_scale * pixel_length

    # ── Bounding box helpers ──────────────────────────────────────────────

    def screen_bbox(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Return (top-left, bottom-right) in pixels of the calibrated
        900 × 900 hanzi canvas projected onto the screen.
        """
        top_left     = self.hanzi_to_screen(0,           HANZI_CANVAS)
        bottom_right = self.hanzi_to_screen(HANZI_CANVAS, 0)
        return top_left, bottom_right

    # ── Diagnostics ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if not self.ready:
            return f"Calibration(ready=False, corners={self.corners})"
        tl, br = self.screen_bbox()
        return (
            f"Calibration("
            f"scale={self.scale:.4f} px/unit, "
            f"inv_scale={self.inv_scale:.4f} unit/px, "
            f"bbox_tl={tl}, bbox_br={br})"
        )