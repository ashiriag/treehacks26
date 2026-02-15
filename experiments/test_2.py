#!/usr/bin/env python3
"""
Hanzi Stroke Order Trainer  –  v2  (calibrated + curve-aware)
═══════════════════════════════════════════════════════════════

Workflow
────────
1. CALIBRATE  – pinch two corners of a bounding box on screen.
               This gives us the affine map  hanzi-space → pixel-space.
2. DRAW       – draw each stroke with two fingers pinched together.
               At every point the algorithm finds the nearest location
               on the reference median (projected into pixel space) and
               compares the user's local direction against the reference
               tangent at that location.
3. FEEDBACK   – each drawn segment is coloured green/red live, and a
               small arrow at the fingertip shows expected vs actual dir.
4. Press C to restart, ESC to quit.

Coordinate systems
──────────────────
• Hanzi  space : x ∈ [0, 900], y ∈ [0, 900], y increases UPWARD  (SVG)
• Screen space : x ∈ [0, W],   y ∈ [0, H],   y increases DOWNWARD (OpenCV)

The calibration transform handles the Y flip automatically.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import os
import sys
import json
import urllib.request

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_INDEX = 0
INDEX_TIP    = 8
MIDDLE_TIP   = 12

TWO_FINGER_THRESH_NORM = 0.06   # normalised distance → pen-down
MOVE_THRESHOLD_PX      = 4      # min px movement to record a new point
LINE_THICKNESS         = 4

TARGET_CHAR    = "七"
HANZI_DATA_DIR = "../makemeahanzi"

# Direction tolerance (degrees).  Applied to the instantaneous tangent
# comparison – no hardcoded per-stroke values anywhere.
MAX_DIR_ANGLE_DEG = 40

# How many trailing points to use for the user's "instantaneous" direction
INSTANT_DIR_WINDOW = 8

# Don't show feedback until the user has drawn at least this far (px)
MIN_DRAWN_LEN_PX = 25

# Ghost overlay alpha (0-1)
GHOST_ALPHA = 0.35

# Colours  (BGR)
COL_OK      = (30,  210,  30)
COL_WRONG   = (30,   30, 220)
COL_GHOST   = (180, 180,  60)
COL_ARROW_E = (255, 200,   0)   # expected direction arrow
COL_ARROW_U = (255, 255, 255)   # user direction arrow
COL_BBOX    = ( 60, 220, 220)

# ─────────────────────────────────────────────────────────────────────────────
# MakeMeAHanzi loader
# ─────────────────────────────────────────────────────────────────────────────

def load_hanzi_json(char: str, data_dir: str) -> dict:
    path = os.path.join(data_dir, "graphics.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"graphics.txt not found in {data_dir}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("character") == char:
                return data
    raise ValueError(f"'{char}' not found in graphics.txt")

# ─────────────────────────────────────────────────────────────────────────────
# Calibration  –  hanzi-space  <->  pixel-space
# ─────────────────────────────────────────────────────────────────────────────

class Calibration:
    """
    Stores a simple scale+translate+y-flip mapping between the fixed
    900x900 hanzi coordinate space and pixel screen space.

    The user draws a bounding box by pinching two corners.  We fit the
    hanzi 0-900 square into that box.
    """

    HANZI_SIZE = 900.0  # hanzi coordinates span [0, 900]

    def __init__(self):
        self.corners: list[tuple[int, int]] = []  # up to 2 pinch points
        self.ready   = False
        self._px0 = 0.0
        self._py0 = 0.0
        self._sx  = 1.0
        self._sy  = 1.0

    def add_corner(self, pt: tuple[int, int]):
        self.corners.append(pt)
        if len(self.corners) == 2:
            self._build()

    def _build(self):
        (x0, y0), (x1, y1) = self.corners
        px_min, px_max = min(x0, x1), max(x0, x1)
        py_min, py_max = min(y0, y1), max(y0, y1)

        box_w = max(px_max - px_min, 1)
        box_h = max(py_max - py_min, 1)

        # uniform scale so the whole 900-unit square fits in the box
        s = min(box_w, box_h) / self.HANZI_SIZE

        cx = (px_min + px_max) / 2.0
        cy = (py_min + py_max) / 2.0

        self._sx  =  s
        self._sy  = -s    # negative: Y flip (hanzi Y up, screen Y down)
        self._px0 = cx - s * self.HANZI_SIZE / 2.0
        # hanzi y=0 maps to screen bottom, hanzi y=900 maps to screen top
        self._py0 = cy + s * self.HANZI_SIZE / 2.0

        self.ready = True

    def to_screen(self, hx: float, hy: float) -> tuple[int, int]:
        px = self._px0 + self._sx * hx
        py = self._py0 + self._sy * hy
        return int(round(px)), int(round(py))

    def to_screen_array(self, pts: np.ndarray) -> np.ndarray:
        """pts shape (N, 2) in hanzi space -> screen coords (N, 2)."""
        px = self._px0 + self._sx * pts[:, 0]
        py = self._py0 + self._sy * pts[:, 1]
        return np.stack([px, py], axis=1).astype(np.float32)

    def screen_bbox(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return (top-left, bottom-right) of the calibrated box in pixels."""
        tl = self.to_screen(0,              self.HANZI_SIZE)
        br = self.to_screen(self.HANZI_SIZE, 0)
        return tl, br

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def polyline_length(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum())


def resample_uniform(pts: np.ndarray, n: int = 64) -> np.ndarray:
    """Resample polyline to n uniformly-spaced points by arc-length."""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return np.repeat(pts[:1], n, axis=0)
    seg   = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s     = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    t   = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float32)
    j   = 0
    for i, ti in enumerate(t):
        while j < len(s) - 2 and s[j + 1] < ti:
            j += 1
        t0, t1 = s[j], s[j + 1]
        p0, p1 = pts[j], pts[j + 1]
        a      = 0.0 if (t1 - t0) < 1e-6 else (ti - t0) / (t1 - t0)
        out[i] = (1.0 - a) * p0 + a * p1
    return out


def unit_vec(v: np.ndarray):
    n = np.linalg.norm(v)
    return None if n < 1e-8 else v / n


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    un = unit_vec(np.asarray(u, dtype=np.float64))
    vn = unit_vec(np.asarray(v, dtype=np.float64))
    if un is None or vn is None:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(un, vn), -1.0, 1.0))))


def dominant_direction(pts: np.ndarray, window: int = None):
    """
    Length-weighted average tangent direction of pts (or its last `window` points).
    Returns a unit vector or None.  Operates purely on direction – scale-agnostic.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if window is not None:
        pts = pts[-window:]
    if len(pts) < 2:
        return None
    r   = resample_uniform(pts, n=min(32, max(len(pts), 2)))
    v   = r[1:] - r[:-1]
    w   = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    avg = v.sum(axis=0)   # equivalent to length-weighted sum of unit vecs * lengths
    return unit_vec(avg)

# ─────────────────────────────────────────────────────────────────────────────
# Reference stroke in screen space + nearest-point query
# ─────────────────────────────────────────────────────────────────────────────

class RefStroke:
    """
    A single reference median stroke projected into pixel space.

    We keep a densely-resampled version (N_DENSE points) so that the
    nearest-point lookup is a simple argmin over a Euclidean distance array.
    """

    N_DENSE = 128

    def __init__(self, hanzi_pts: np.ndarray, calib: Calibration):
        screen = calib.to_screen_array(hanzi_pts)
        self.dense: np.ndarray    = resample_uniform(screen, n=self.N_DENSE)
        self.tangents: np.ndarray = self._build_tangents(self.dense)

    @staticmethod
    def _build_tangents(pts: np.ndarray) -> np.ndarray:
        """Central-difference tangents, each normalised to unit length."""
        n   = len(pts)
        out = np.zeros_like(pts)
        for i in range(n):
            lo = max(i - 1, 0)
            hi = min(i + 1, n - 1)
            v  = pts[hi] - pts[lo]
            nv = np.linalg.norm(v)
            out[i] = v / nv if nv > 1e-8 else np.array([1.0, 0.0])
        return out

    def nearest(self, px: float, py: float) -> tuple[int, float]:
        """
        Return (dense index, normalised t in [0,1]) of the closest point
        on the reference stroke to screen point (px, py).
        """
        pt  = np.array([px, py], dtype=np.float32)
        d   = np.linalg.norm(self.dense - pt, axis=1)
        idx = int(np.argmin(d))
        t   = idx / max(self.N_DENSE - 1, 1)
        return idx, t

    def tangent_at(self, idx: int) -> np.ndarray:
        return self.tangents[idx]

    def draw_ghost(self, overlay: np.ndarray):
        pts_int = self.dense.astype(np.int32)
        for i in range(len(pts_int) - 1):
            cv2.line(overlay,
                     tuple(pts_int[i]), tuple(pts_int[i + 1]),
                     COL_GHOST, 2, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_stroke_coloured(canvas, pts, colors):
    """Draw pts[i]->pts[i+1] using colors[i].  len(colors) == len(pts)-1."""
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], colors[i], LINE_THICKNESS, cv2.LINE_AA)


def draw_arrow(img, origin, direction_unit, length=48,
               color=(255, 255, 255), thickness=2):
    if direction_unit is None:
        return
    end = (
        int(origin[0] + direction_unit[0] * length),
        int(origin[1] + direction_unit[1] * length),
    )
    cv2.arrowedLine(img, origin, end, color, thickness,
                    cv2.LINE_AA, tipLength=0.35)

# ─────────────────────────────────────────────────────────────────────────────
# Model download
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker model ...")
    try:
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
            MODEL_PATH,
        )
        print("Done.")
    except Exception as e:
        print("Download failed:", e)
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Load hanzi data
# ─────────────────────────────────────────────────────────────────────────────

try:
    char_data = load_hanzi_json(TARGET_CHAR, HANZI_DATA_DIR)
    medians   = [np.array(m, dtype=np.float32) for m in char_data["medians"]]
    print(f"Loaded '{TARGET_CHAR}'  ({len(medians)} strokes).")
except Exception as e:
    print("Failed to load hanzi data:", e)
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# State machine constants
# ─────────────────────────────────────────────────────────────────────────────

ST_CAL_FIRST  = "calibrate_first"
ST_CAL_SECOND = "calibrate_second"
ST_DRAW       = "draw"
ST_DONE       = "done"

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────────────────────────────────────────

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
mp_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO,
)

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Camera error.")
    sys.exit(1)

calib        = Calibration()
ref_strokes: list[RefStroke] = []

canvas       = None
state        = ST_CAL_FIRST

was_pinching  = False
pinch_debounce = 0

drawing      = False
prev_pt      = None
stroke_idx   = 0
user_pts:  list[tuple[int, int]] = []
seg_colors: list[tuple[int, int, int]] = []

live_ok    = None
live_angle = None
live_t     = None

frame_count = 0


def build_ref_strokes():
    global ref_strokes
    ref_strokes = [RefStroke(m, calib) for m in medians]
    print("Reference strokes projected into pixel space.")


def full_reset():
    global calib, ref_strokes, state
    global drawing, prev_pt, stroke_idx, user_pts, seg_colors
    global live_ok, live_angle, live_t, canvas
    calib       = Calibration()
    ref_strokes = []
    state       = ST_CAL_FIRST
    drawing     = False
    prev_pt     = None
    stroke_idx  = 0
    user_pts    = []
    seg_colors  = []
    live_ok     = None
    live_angle  = None
    live_t      = None
    if canvas is not None:
        canvas[:] = 0
    print("--- Full reset ---")


with vision.HandLandmarker.create_from_options(mp_options) as lmk:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        # ── MediaPipe inference ───────────────────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_count += 1
        result = lmk.detect_for_video(mp_img, frame_count * 33)

        is_pinching = False
        tip_xy      = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            idx  = hand[INDEX_TIP]
            mid  = hand[MIDDLE_TIP]
            tip_xy      = (int(idx.x * w), int(idx.y * h))
            dist        = math.hypot(idx.x - mid.x, idx.y - mid.y)
            is_pinching = dist < TWO_FINGER_THRESH_NORM

            cv2.circle(frame, (int(idx.x*w), int(idx.y*h)), 6, (0,255,255), -1)
            cv2.circle(frame, (int(mid.x*w), int(mid.y*h)), 6, (0,255,255), -1)
            cv2.line(frame, (int(idx.x*w), int(idx.y*h)),
                            (int(mid.x*w), int(mid.y*h)), (0,255,255), 2)

        pinch_just_down = is_pinching and not was_pinching
        pinch_just_up   = (not is_pinching) and was_pinching
        was_pinching    = is_pinching

        if pinch_debounce > 0:
            pinch_debounce  -= 1
            pinch_just_down  = False

        # ── Ghost layer (built each frame once calibrated) ────────────────
        ghost_layer = np.zeros_like(frame)
        if calib.ready and ref_strokes:
            for i, rs in enumerate(ref_strokes):
                if state == ST_DRAW:
                    if   i < stroke_idx: continue          # done, skip
                    elif i == stroke_idx:
                        rs.draw_ghost(ghost_layer)         # current – full brightness
                    else:
                        # future strokes: draw dimmer by scaling colour
                        tmp = np.zeros_like(frame)
                        rs.draw_ghost(tmp)
                        ghost_layer = cv2.addWeighted(ghost_layer, 1.0, tmp, 0.3, 0)
                        continue
                else:
                    rs.draw_ghost(ghost_layer)

        # ── CALIBRATION states ────────────────────────────────────────────
        if state in (ST_CAL_FIRST, ST_CAL_SECOND):
            n = 1 if state == ST_CAL_FIRST else 2
            cv2.putText(frame,
                f"CALIBRATE: pinch corner {n}/2 of your drawing area"
                if n == 1 else
                "CALIBRATE: pinch the OPPOSITE corner",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_BBOX, 2, cv2.LINE_AA)

            for c in calib.corners:
                cv2.drawMarker(frame, c, COL_BBOX, cv2.MARKER_CROSS, 20, 2)

            if pinch_just_down and tip_xy is not None:
                calib.add_corner(tip_xy)
                pinch_debounce = 18
                if calib.ready:
                    tl, br = calib.screen_bbox()
                    cv2.rectangle(frame, tl, br, COL_BBOX, 2)
                    build_ref_strokes()
                    state = ST_DRAW
                    print("Calibration done. Start drawing!")
                else:
                    state = ST_CAL_SECOND

        # ── DRAW state ────────────────────────────────────────────────────
        elif state == ST_DRAW:
            n_total = len(medians)

            if stroke_idx >= n_total:
                state = ST_DONE
            else:
                rs = ref_strokes[stroke_idx]

                cv2.putText(frame,
                    f"'{TARGET_CHAR}'  stroke {stroke_idx+1}/{n_total}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)

                # Pen-down: accumulate points + per-segment feedback
                if is_pinching and tip_xy is not None:
                    x, y = tip_xy

                    if not drawing:
                        drawing    = True
                        prev_pt    = (x, y)
                        user_pts   = [(x, y)]
                        seg_colors = []
                        live_ok    = None
                        live_angle = None
                        live_t     = None

                    else:
                        dist = math.hypot(x - prev_pt[0], y - prev_pt[1])
                        if dist > MOVE_THRESHOLD_PX:

                            # ── Core: nearest-point lookup ─────────────────
                            # Project fingertip onto the reference stroke
                            # (already in screen space) to get t in [0,1].
                            ref_idx, t = rs.nearest(x, y)
                            live_t = t

                            # Reference tangent at that position (unit vector,
                            # screen space – encodes the expected direction for
                            # THIS exact location on the curvy stroke).
                            ref_tan = rs.tangent_at(ref_idx)

                            # User's instantaneous direction: trailing window
                            # of drawn points, direction-only comparison.
                            drawn_len = polyline_length(
                                np.array(user_pts, dtype=np.float32))
                            u_dir = None
                            if drawn_len >= MIN_DRAWN_LEN_PX and len(user_pts) >= 2:
                                u_dir = dominant_direction(
                                    np.array(user_pts, dtype=np.float32),
                                    window=INSTANT_DIR_WINDOW)

                            if u_dir is not None:
                                ang        = angle_deg(u_dir, ref_tan)
                                ok         = ang <= MAX_DIR_ANGLE_DEG
                                live_ok    = ok
                                live_angle = ang
                                seg_colors.append(COL_OK if ok else COL_WRONG)
                            else:
                                seg_colors.append((140, 140, 140))

                            user_pts.append((x, y))
                            prev_pt = (x, y)

                # Pen-up: finalise stroke
                if pinch_just_up and drawing:
                    drawing = False

                    if len(seg_colors) > 0:
                        n_ok    = sum(1 for c in seg_colors if c == COL_OK)
                        pct     = 100.0 * n_ok / len(seg_colors)
                        verdict = "OK" if pct >= 60 else "WRONG DIRECTION"
                        print(f"Stroke {stroke_idx+1}/{n_total}  {verdict}"
                              f"  ({pct:.0f}% segments correct)")
                    else:
                        print(f"Stroke {stroke_idx+1}/{n_total}  – too short.")

                    # Bake onto permanent canvas
                    if len(user_pts) >= 2:
                        colors = seg_colors or [(140,140,140)]*(len(user_pts)-1)
                        draw_stroke_coloured(canvas, user_pts, colors)

                    stroke_idx += 1
                    user_pts    = []
                    seg_colors  = []
                    live_ok     = None
                    live_angle  = None
                    live_t      = None
                    prev_pt     = None

                # Live in-progress stroke drawn on top of frame each frame
                if drawing and len(user_pts) >= 2:
                    colors = seg_colors or [(140,140,140)]*(len(user_pts)-1)
                    draw_stroke_coloured(frame, user_pts, colors)

                    if live_ok is not None and tip_xy is not None:
                        ref_idx, _ = rs.nearest(*tip_xy)
                        ref_tan    = rs.tangent_at(ref_idx)
                        u_dir      = dominant_direction(
                            np.array(user_pts, dtype=np.float32),
                            window=INSTANT_DIR_WINDOW)

                        label  = ("correct" if live_ok else "wrong direction")
                        label  = f"direction: {label}  {live_angle:.1f}deg"
                        colour = COL_OK if live_ok else COL_WRONG
                        cv2.putText(frame, label, (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    colour, 2, cv2.LINE_AA)

                        # Yellow arrow = where reference wants you to go
                        draw_arrow(frame, tip_xy, ref_tan,
                                   length=50, color=COL_ARROW_E, thickness=3)
                        # White arrow = where you are actually going
                        draw_arrow(frame, tip_xy, u_dir,
                                   length=50, color=COL_ARROW_U, thickness=2)

                        if live_t is not None:
                            cv2.putText(frame,
                                f"ref progress: {live_t*100:.0f}%",
                                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (160, 160, 160), 1, cv2.LINE_AA)

        # ── DONE state ────────────────────────────────────────────────────
        elif state == ST_DONE:
            cv2.putText(frame,
                f"'{TARGET_CHAR}' complete! Press C to restart.",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                COL_OK, 2, cv2.LINE_AA)

        # ── Calibrated bbox outline ───────────────────────────────────────
        if calib.ready:
            tl, br = calib.screen_bbox()
            cv2.rectangle(frame, tl, br, COL_BBOX, 1)

        # ── Composite: frame + ghost + permanent canvas ───────────────────
        combined = cv2.addWeighted(frame,    1.0, ghost_layer, GHOST_ALPHA, 0)
        combined = cv2.addWeighted(combined, 1.0, canvas,      1.0,         0)

        cv2.imshow("Hanzi Stroke Trainer", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:           # ESC
            break
        elif key == ord("c"):
            full_reset()

cap.release()
cv2.destroyAllWindows()
