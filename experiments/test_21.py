#!/usr/bin/env python3
"""
Hanzi Stroke Order Trainer  –  v3  (curve-aware, calibrated)
═════════════════════════════════════════════════════════════

How curve feedback works
────────────────────────
Previous versions computed one "overall direction" for the whole stroke so
far, which averages out any curves and gives meaningless feedback mid-stroke.

This version works point-by-point:

  1. Each new fingertip position is projected from pixel space into hanzi
     space using the calibration inverse transform.

  2. The nearest point on the *reference* median is found in hanzi space
     (argmin distance). This gives t ∈ [0,1] — where on the reference
     stroke the user currently is.

  3. The pre-computed tangent at t gives the *expected* direction for that
     exact location on the curve.

  4. The user's *instantaneous* direction is computed from only the last
     INSTANT_WINDOW points so it reflects the current trajectory rather
     than the historical average.

  5. The angle between those two unit vectors is the live feedback signal.
     Each drawn segment is coloured green/red independently.

Calibration
───────────
Press 'k' at two opposite corners of your intended drawing area.
Press 'r' to reset calibration.
All direction math is done in hanzi space after the inverse transform.
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
from calibration import Calibration, HANZI_CANVAS

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_INDEX = 0
INDEX_TIP    = 8
MIDDLE_TIP   = 4

TWO_FINGER_THRESH_NORM = 0.06
MOVE_THRESHOLD_PX      = 5
LINE_THICKNESS         = 5

TARGET_CHAR    = "七"
HANZI_DATA_DIR = "../makemeahanzi"

# Angle tolerance in degrees — the only threshold, applies to every stroke.
MAX_DIR_ANGLE_DEG = 35

# How many trailing points to use for the user's instantaneous direction.
# Small = more responsive to curves; large = smoother but lags behind curves.
INSTANT_WINDOW = 6

# Don't give feedback until the user has drawn at least this far (hanzi units).
# Avoids jitter at stroke start. ~30 units is a short flick in 900-unit space.
MIN_DRAWN_HZ = 30.0

# Throttle feedback updates to every N new points (reduces flicker).
LIVE_CHECK_EVERY_N = 2

# Number of points to densely resample each reference median for nearest-point lookup.
REF_DENSE_N = 256

# Colours (BGR)
COL_OK       = (30,  210,  30)
COL_WRONG    = (30,   30, 220)
COL_NEUTRAL  = (140, 140, 140)
COL_GHOST    = (180, 180,  60)
COL_ARROW_EX = (255, 200,   0)   # expected direction
COL_ARROW_US = (255, 255, 255)   # user direction

GHOST_ALPHA = 0.35

# ─────────────────────────────────────────────────────────────────────────────
# MakeMeAHanzi loader
# ─────────────────────────────────────────────────────────────────────────────

def load_hanzi_json(char: str, data_dir: str) -> dict:
    path = os.path.join(data_dir, "graphics.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}\n"
            "Make sure graphics.txt is inside your makemeahanzi data folder."
        )
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("character") == char:
                return data
    raise ValueError(f"Character '{char}' not found in graphics.txt")

# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def resample_uniform(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to n uniformly-spaced points by arc-length."""
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
        a      = 0.0 if (t1 - t0) < 1e-6 else (ti - t0) / (t1 - t0)
        out[i] = (1.0 - a) * pts[j] + a * pts[j + 1]
    return out


def build_tangents(pts: np.ndarray) -> np.ndarray:
    """Central-difference unit tangents for each point in pts."""
    n   = len(pts)
    out = np.zeros_like(pts)
    for i in range(n):
        lo = max(i - 1, 0)
        hi = min(i + 1, n - 1)
        v  = pts[hi] - pts[lo]
        nv = np.linalg.norm(v)
        out[i] = v / nv if nv > 1e-8 else np.array([1.0, 0.0], dtype=np.float32)
    return out


def unit(v: np.ndarray):
    n = np.linalg.norm(v)
    return None if n < 1e-8 else v / n


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    un, vn = unit(u), unit(v)
    if un is None or vn is None:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(un, vn), -1.0, 1.0))))


def instantaneous_direction(pts_hz: np.ndarray, window: int):
    """
    Direction of the most recent `window` points in hanzi space.
    Uses the sum of segment vectors for stability within the window.
    """
    pts_hz = np.asarray(pts_hz, dtype=np.float32)
    recent = pts_hz[-window:] if len(pts_hz) >= window else pts_hz
    if len(recent) < 2:
        return None
    v = recent[1:] - recent[:-1]
    return unit(v.sum(axis=0))

# ─────────────────────────────────────────────────────────────────────────────
# Reference stroke  (hanzi space, dense + tangents precomputed)
# ─────────────────────────────────────────────────────────────────────────────

class RefStroke:
    """
    One reference median densely resampled and stored in hanzi space.

    Nearest-point lookup and tangent queries all happen in hanzi space,
    so user points must be inverse-transformed before querying.
    """

    def __init__(self, hanzi_pts: np.ndarray):
        self.dense    = resample_uniform(hanzi_pts, n=REF_DENSE_N)
        self.tangents = build_tangents(self.dense)

    def nearest(self, hx: float, hy: float) -> tuple[int, float]:
        """
        Return (index, t in [0,1]) of the closest dense point to (hx, hy).
        Arithmetic in hanzi space only — no pixel coordinate bias.
        """
        pt  = np.array([hx, hy], dtype=np.float32)
        d   = np.linalg.norm(self.dense - pt, axis=1)
        idx = int(np.argmin(d))
        t   = idx / max(REF_DENSE_N - 1, 1)
        return idx, t

    def tangent_at(self, idx: int) -> np.ndarray:
        return self.tangents[idx]

    def draw_ghost(self, frame: np.ndarray, calib: Calibration, alpha: float = 1.0):
        """Draw the reference stroke as a faint polyline in pixel space."""
        screen_pts = calib.hanzi_to_screen_array(self.dense).astype(np.int32)
        overlay = frame.copy()
        for i in range(len(screen_pts) - 1):
            cv2.line(overlay,
                     tuple(screen_pts[i]), tuple(screen_pts[i + 1]),
                     COL_GHOST, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

# ─────────────────────────────────────────────────────────────────────────────
# Per-point feedback
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_point(
    new_pt_hz:      np.ndarray,
    history_hz:     np.ndarray,
    ref:            RefStroke,
    total_drawn_hz: float,
) -> tuple:
    """
    Evaluate direction at the latest drawn point.

    The key idea: nearest-point lookup on the reference (in hanzi space)
    gives us the expected tangent for exactly where the user is right now
    on the curve — including any bends or hooks in the stroke.

    Returns (ok, angle, t, ref_tangent, user_dir)
      ok          : True/False/None  (None = not enough drawn yet)
      angle       : degrees between user and reference direction
      t           : position on reference [0, 1]
      ref_tangent : expected direction unit vector (hanzi space)
      user_dir    : actual direction unit vector (hanzi space) or None
    """
    idx, t  = ref.nearest(*new_pt_hz)
    ref_tan = ref.tangent_at(idx)

    if total_drawn_hz < MIN_DRAWN_HZ:
        return None, 0.0, t, ref_tan, None

    u_dir = instantaneous_direction(history_hz, window=INSTANT_WINDOW)
    if u_dir is None:
        return None, 0.0, t, ref_tan, None

    # Both vectors are in hanzi space.
    # user history_hz was built by screen_to_hanzi() which applies the Y flip,
    # so its Y convention matches the hanzi-space median tangents.
    ang = angle_deg(u_dir, ref_tan)
    ok  = ang <= MAX_DIR_ANGLE_DEG

    return ok, ang, t, ref_tan, u_dir

# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_coloured_stroke(canvas, pts_px, colors):
    for i in range(min(len(pts_px) - 1, len(colors))):
        cv2.line(canvas, pts_px[i], pts_px[i + 1], colors[i], LINE_THICKNESS, cv2.LINE_AA)


def draw_arrow_px(img, origin_px, dir_unit_hz, calib: Calibration,
                  length_hz=40.0, color=(255, 255, 255), thickness=2):
    """
    Draw an arrow at origin_px pointing in dir_unit_hz (hanzi space).

    Converts the hanzi-space unit vector to a pixel displacement:
      dx_px =  dir_hz[0] * scale       (x same direction)
      dy_px = -dir_hz[1] * scale       (Y flip: hanzi +y = screen -y)
    """
    if dir_unit_hz is None:
        return
    ox, oy = origin_px
    dx_px =  dir_unit_hz[0] * calib.hanzi_len_to_px(length_hz)
    dy_px = -dir_unit_hz[1] * calib.hanzi_len_to_px(length_hz)
    end = (int(ox + dx_px), int(oy + dy_px))
    cv2.arrowedLine(img, origin_px, end, color, thickness, cv2.LINE_AA, tipLength=0.35)


def draw_calibration_overlay(frame, calib: Calibration):
    if not calib.ready:
        msg = f"CALIBRATION: {len(calib.corners)}/2 corners. Pinch + press 'k' twice. ('r' reset)"
        cv2.putText(frame, msg, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    else:
        msg = f"CALIBRATED  scale={calib.scale:.3f} px/unit  ('r' reset)"
        cv2.putText(frame, msg, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(calib.corners):
        cv2.circle(frame, (int(x), int(y)), 8, (255, 0, 255), -1)
        cv2.putText(frame, f"{i+1}", (int(x)+10, int(y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    if calib.ready:
        tl, br = calib.screen_bbox()
        cv2.rectangle(frame,
                      (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])),
                      (255, 255, 0), 2)
        cx, cy = calib.hanzi_to_screen(HANZI_CANVAS / 2, HANZI_CANVAS / 2)
        cv2.drawMarker(frame, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
    return frame

# ─────────────────────────────────────────────────────────────────────────────
# Model download
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker model …")
    try:
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
            MODEL_PATH,
        )
        print("Model downloaded.")
    except Exception as e:
        print("Failed to download model:", e)
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Load hanzi data + build reference strokes (hanzi space, no calibration needed)
# ─────────────────────────────────────────────────────────────────────────────

try:
    char_data = load_hanzi_json(TARGET_CHAR, HANZI_DATA_DIR)
    medians   = char_data["medians"]
    print(f"Loaded '{TARGET_CHAR}' with {len(medians)} median strokes.")
except Exception as e:
    print("Failed to load hanzi data:", e)
    sys.exit(1)

ref_strokes: list[RefStroke] = [
    RefStroke(np.array(m, dtype=np.float32)) for m in medians
]

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────────────────────────────────────────

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO,
)

# ─────────────────────────────────────────────────────────────────────────────
# Application state
# ─────────────────────────────────────────────────────────────────────────────

calib = Calibration()

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Camera failed to initialize.")
    sys.exit(1)

canvas      = None
drawing     = False
prev_point  = None
frame_count = 0
stroke_idx  = 0

pts_px:      list[tuple[int, int]]     = []
pts_hz:      list[tuple[float, float]] = []
seg_colors:  list[tuple[int, int, int]]= []
drawn_len_hz = 0.0

live_ok      = None
live_angle   = None
live_t       = None
live_ref_tan = None
live_u_dir   = None
live_counter = 0

with vision.HandLandmarker.create_from_options(options) as hand_landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        # ── MediaPipe ─────────────────────────────────────────────────────
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_count += 1
        result   = hand_landmarker.detect_for_video(mp_image, frame_count * 33)

        finger_down = False
        tip_xy      = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            idx  = hand[INDEX_TIP]
            mid  = hand[MIDDLE_TIP]
            tip_xy      = (int(idx.x * w), int(idx.y * h))
            dist        = math.hypot(idx.x - mid.x, idx.y - mid.y)
            finger_down = dist < TWO_FINGER_THRESH_NORM

            cv2.circle(frame, (int(idx.x*w), int(idx.y*h)), 6, (0, 255, 255), -1)
            cv2.circle(frame, (int(mid.x*w), int(mid.y*h)), 6, (0, 255, 255), -1)
            cv2.line(frame,   (int(idx.x*w), int(idx.y*h)),
                               (int(mid.x*w), int(mid.y*h)), (0, 255, 255), 2)
            cv2.putText(frame, f"pinch={dist:.3f}", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # ── Ghost: current reference stroke ───────────────────────────────
        if calib.ready and stroke_idx < len(ref_strokes):
            ref_strokes[stroke_idx].draw_ghost(frame, calib, alpha=GHOST_ALPHA)

        # ── Stroke capture + per-point feedback ───────────────────────────
        if finger_down and tip_xy is not None and calib.ready:
            x, y = tip_xy

            if not drawing:
                drawing      = True
                prev_point   = (x, y)
                pts_px       = [(x, y)]
                pts_hz       = [calib.screen_to_hanzi(x, y)]
                seg_colors   = []
                drawn_len_hz = 0.0
                live_ok      = None
                live_angle   = None
                live_t       = None
                live_ref_tan = None
                live_u_dir   = None
                live_counter = 0

            else:
                move_px = math.hypot(x - prev_point[0], y - prev_point[1])
                if move_px > MOVE_THRESHOLD_PX:
                    hx, hy  = calib.screen_to_hanzi(x, y)
                    prev_hz = pts_hz[-1]

                    # Arc-length in hanzi space — scale-aware progress tracking
                    seg_hz       = math.hypot(hx - prev_hz[0], hy - prev_hz[1])
                    drawn_len_hz += seg_hz

                    pts_px.append((x, y))
                    pts_hz.append((hx, hy))
                    prev_point = (x, y)

                    # Throttled per-point direction check
                    live_counter += 1
                    if (live_counter % LIVE_CHECK_EVERY_N == 0
                            and stroke_idx < len(ref_strokes)):

                        ok, ang, t, ref_tan, u_dir = evaluate_point(
                            new_pt_hz      = np.array([hx, hy]),
                            history_hz     = np.array(pts_hz, dtype=np.float32),
                            ref            = ref_strokes[stroke_idx],
                            total_drawn_hz = drawn_len_hz,
                        )

                        seg_colors.append(
                            COL_NEUTRAL if ok is None else (COL_OK if ok else COL_WRONG)
                        )
                        if ok is not None:
                            live_ok    = ok
                            live_angle = ang
                        live_t       = t
                        live_ref_tan = ref_tan
                        live_u_dir   = u_dir

                    else:
                        seg_colors.append(seg_colors[-1] if seg_colors else COL_NEUTRAL)

        # ── Stroke end ─────────────────────────────────────────────────────
        if (not finger_down) and drawing:
            drawing    = False
            prev_point = None

            if stroke_idx < len(ref_strokes) and len(pts_px) >= 2:
                n_ok  = sum(1 for c in seg_colors if c == COL_OK)
                n_bad = sum(1 for c in seg_colors if c == COL_WRONG)
                n_ev  = n_ok + n_bad
                pct   = 100.0 * n_ok / n_ev if n_ev > 0 else 0.0
                verdict = "OK" if pct >= 60 else "WRONG DIRECTION"
                print(
                    f"Stroke {stroke_idx+1}/{len(ref_strokes)}  {verdict}"
                    f"  ({pct:.0f}% correct  |  {drawn_len_hz:.1f} hz units drawn)"
                )
                draw_coloured_stroke(canvas, pts_px, seg_colors)
            elif not calib.ready:
                print("Stroke ignored — calibrate first ('k' at two corners).")
            else:
                print(f"Stroke {stroke_idx+1}: too few points.")

            stroke_idx   += 1
            pts_px        = []
            pts_hz        = []
            seg_colors    = []
            drawn_len_hz  = 0.0
            live_ok       = None
            live_angle    = None
            live_t        = None
            live_ref_tan  = None
            live_u_dir    = None

        # ── In-progress stroke drawn live on frame ─────────────────────────
        if drawing and len(pts_px) >= 2:
            draw_coloured_stroke(frame, pts_px, seg_colors)

        # ── HUD ────────────────────────────────────────────────────────────
        total = len(ref_strokes)
        header = (f"'{TARGET_CHAR}'  stroke {stroke_idx+1}/{total}"
                  if stroke_idx < total
                  else f"'{TARGET_CHAR}'  done ({total}/{total})  —  press C to restart")
        cv2.putText(frame, header, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        if drawing and stroke_idx < total:
            if live_ok is None:
                label, col = "direction: draw more ...", (180, 180, 180)
            else:
                label = (f"direction: {'OK' if live_ok else 'WRONG'}"
                         f"  {live_angle:.1f}deg  (max {MAX_DIR_ANGLE_DEG}deg)")
                col   = COL_OK if live_ok else COL_WRONG
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)

            if live_t is not None:
                cv2.putText(frame, f"ref progress: {live_t*100:.0f}%", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

            # Arrows: yellow = expected, white = actual
            if tip_xy is not None and calib.ready:
                draw_arrow_px(frame, tip_xy, live_ref_tan, calib,
                              length_hz=40, color=COL_ARROW_EX, thickness=3)
                draw_arrow_px(frame, tip_xy, live_u_dir, calib,
                              length_hz=40, color=COL_ARROW_US, thickness=2)

        frame = draw_calibration_overlay(frame, calib)
        combined = cv2.addWeighted(frame, 0.7, canvas, 1.0, 0)
        cv2.imshow("Hanzi Stroke Trainer", combined)

        # ── Keys ───────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("c"):
            canvas       = np.zeros_like(frame)
            drawing      = False
            prev_point   = None
            stroke_idx   = 0
            pts_px       = []
            pts_hz       = []
            seg_colors   = []
            drawn_len_hz = 0.0
            live_ok = live_angle = live_t = live_ref_tan = live_u_dir = None
            live_counter = 0
            print("Restarted.")
        elif key == ord("k"):
            if tip_xy is not None:
                calib.add_corner(tip_xy)
                print(f"Corner {len(calib.corners)} at {tip_xy} -> {calib}")
        elif key == ord("r"):
            calib.reset()
            print("Calibration reset.")

cap.release()
cv2.destroyAllWindows()