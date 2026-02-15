#!/usr/bin/env python3
"""
Hanzi Stroke Order Trainer
──────────────────────────
Uses MediaPipe hand tracking to detect a two-finger pinch gesture as a
"pen down" signal. While drawing, the user's stroke is scaled into the
same coordinate system as the MakeMeAHanzi median data; we find where the
user is along the median by arc-length progress and compare local
tangents to see if they're following the curve (e.g. continuous curve OK).

Key design choices
──────────────────
• Scale user drawing to median: user points (pixels) → hanzi space (0–900)
  via calibration; medians are normalized from 1024 to 900 so lengths and
  positions share the same units. Progress = user_arc_len / ref_arc_len.

• Local tangent comparison: at the current progress fraction we take the
  median's unit tangent at that point and the user's local tangent at the
  stroke end. If the angle between them is within MAX_DIR_ANGLE_DEG, the
  stroke is "following the curve" (direction OK).

• Optional ghost: a yellow circle shows where on the median the user
  "should" be at current progress (SHOW_EXPECTED_POINT).
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

# MakeMeAHanzi graphics.txt uses a 1024-unit canvas (y down). We normalize to
# calibration space (900, y-up) so user and ref share the same coordinate system.
MEDIAN_CANVAS = 1024.0

# ─────────────────────────── Config ─────────────────────────────────────────

CAMERA_INDEX = 0

# MediaPipe Hand landmark indices
INDEX_TIP  = 8
MIDDLE_TIP = 4

# Two fingers held together → "pen down"
TWO_FINGER_THRESH_NORM = 0.06

# Drawing
MOVE_THRESHOLD_PX = 5
LINE_THICKNESS    = 5

# Hanzi
TARGET_CHAR      = "七"               # ← change me
HANZI_DATA_DIR   = "../makemeahanzi"  # folder containing graphics.txt

# Direction tolerance: angle (degrees) between user direction and reference.
# A single value is applied consistently to ALL strokes — no per-stroke magic.
MAX_DIR_ANGLE_DEG = 35

# Don't evaluate until the user has drawn at least this many pixels
MIN_LIVE_LEN_PX = 40

# Throttle live checks to every N new points (reduces flicker)
LIVE_CHECK_EVERY_N = 3

# Draw a yellow circle at "where the median says you should be" (curve-following)
SHOW_EXPECTED_POINT = True

calib = Calibration()

# ──────────────────────── MakeMeAHanzi loader ────────────────────────────────

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

# ──────────────────────── Geometry helpers ───────────────────────────────────

def polyline_length(pts: np.ndarray) -> float:
    """Total arc-length of a polyline."""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum())


def resample_polyline(pts: np.ndarray, n: int = 32) -> np.ndarray:
    """Resample polyline to *n* uniformly-spaced points by arc-length."""
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

def draw_calibration_overlay(frame, calib: Calibration):
    h, w = frame.shape[:2]

    # instructions
    if not calib.ready:
        msg = f"CALIBRATION: {len(calib.corners)}/2 corners. Pinch + press 'k' twice. ('r' reset)"
        cv2.putText(frame, msg, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 2, cv2.LINE_AA)
    else:
        msg = f"CALIBRATION READY  scale={calib.scale:.3f} px/unit  ('r' reset)"
        cv2.putText(frame, msg, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 2, cv2.LINE_AA)

    # draw corners already selected
    for i, (x, y) in enumerate(calib.corners):
        cv2.circle(frame, (int(x), int(y)), 8, (255, 0, 255), -1)
        cv2.putText(frame, f"{i+1}", (int(x)+10, int(y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2, cv2.LINE_AA)

    # if ready, draw the calibrated 900x900 canvas bbox
    if calib.ready:
        (tl, br) = calib.screen_bbox()   # tl=(x,y), br=(x,y)
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        cv2.rectangle(frame, tl, br, (255, 255, 0), 2)

        # draw center crosshair (hanzi 450,450)
        cx, cy = calib.hanzi_to_screen(HANZI_CANVAS/2, HANZI_CANVAS/2)
        cv2.drawMarker(frame, (cx, cy), (255, 255, 0), markerType=cv2.MARKER_CROSS,
                       markerSize=20, thickness=2)

    return frame

def dominant_direction(pts: np.ndarray) -> np.ndarray | None:
    """
    Return a unit vector representing the overall direction of the stroke.

    Uses the length-weighted average of tangent vectors along a resampled
    polyline.  This is coordinate-system and scale agnostic — it only
    cares about *direction*.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return None

    resampled = resample_polyline(pts, n=32)
    tangents  = resampled[1:] - resampled[:-1]          # (31, 2)
    lengths   = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
    unit_t    = tangents / lengths                       # unit tangents
    weights   = lengths[:, 0]                            # weight by segment length

    avg = (unit_t * weights[:, None]).sum(axis=0)
    mag = np.linalg.norm(avg)
    if mag < 1e-6:
        return None
    return avg / mag


def angle_between_dirs(u: np.ndarray, v: np.ndarray) -> float:
    """Angle in degrees between two direction vectors (need not be unit)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    un = u / (np.linalg.norm(u) + 1e-9)
    vn = v / (np.linalg.norm(v) + 1e-9)
    dot = float(np.clip(np.dot(un, vn), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def prefix_by_fraction(pts: np.ndarray, frac: float) -> np.ndarray:
    """
    Return the prefix of *pts* that covers *frac* of its total arc-length.
    frac = 0.0 → just the start point; frac = 1.0 → the whole polyline.
    """
    pts  = np.asarray(pts, dtype=np.float32)
    frac = float(np.clip(frac, 0.0, 1.0))
    if len(pts) < 2:
        return pts

    seg    = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s      = np.concatenate([[0.0], np.cumsum(seg)])
    total  = s[-1]
    if total < 1e-6:
        return pts[:1]

    target = frac * total
    idx    = int(np.clip(np.searchsorted(s, target, side="right") - 1, 0, len(pts) - 2))

    t0, t1 = s[idx], s[idx + 1]
    p0, p1 = pts[idx], pts[idx + 1]
    a      = 0.0 if (t1 - t0) < 1e-6 else (target - t0) / (t1 - t0)
    interp = (1.0 - a) * p0 + a * p1

    return np.vstack([pts[: idx + 1], interp])


def median_to_hanzi(pts: np.ndarray) -> np.ndarray:
    """
    Transform median points from MakeMeAHanzi space (0–1024, y down)
    to calibration hanzi space (0–900, y up) so lengths and positions
    match the user stroke after screen_to_hanzi.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 0:
        return pts
    scale = HANZI_CANVAS / MEDIAN_CANVAS
    out = pts.copy()
    out[:, 0] = pts[:, 0] * scale
    out[:, 1] = HANZI_CANVAS - pts[:, 1] * scale  # y flip
    return out


def point_and_tangent_at_fraction(pts: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Return (point, unit_tangent) at fraction frac (0–1) of arc-length along the polyline.
    Tangent is the direction of the segment containing that point (forward along the curve).
    """
    pts = np.asarray(pts, dtype=np.float32)
    frac = float(np.clip(frac, 0.0, 1.0))
    if len(pts) < 2:
        return None
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-6:
        return None
    target = frac * total
    idx = int(np.clip(np.searchsorted(s, target, side="right") - 1, 0, len(pts) - 2))
    t0, t1 = s[idx], s[idx + 1]
    p0, p1 = pts[idx], pts[idx + 1]
    a = 0.0 if (t1 - t0) < 1e-6 else (target - t0) / (t1 - t0)
    point = (1.0 - a) * p0 + a * p1
    tangent = p1 - p0
    mag = np.linalg.norm(tangent)
    if mag < 1e-8:
        return None
    unit_tangent = tangent / mag
    return point, unit_tangent


def local_tangent_at_end(pts: np.ndarray, num_segments: int = 3) -> np.ndarray | None:
    """
    Unit tangent at the end of the stroke (average over last num_segments segments).
    Returns None if not enough points.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return None
    tangents = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
    unit_t = tangents / lengths
    n = min(num_segments, len(unit_t))
    avg = unit_t[-n:].sum(axis=0)
    mag = np.linalg.norm(avg)
    if mag < 1e-8:
        return None
    return avg / mag


def user_pixels_to_hanzi(user_pts_px: np.ndarray, calib: Calibration) -> np.ndarray:
    """Convert list of (x, y) pixel points to hanzi-space (0–900, y-up) using calibration."""
    if not calib.ready or len(user_pts_px) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out = np.empty((len(user_pts_px), 2), dtype=np.float32)
    for i in range(len(user_pts_px)):
        hx, hy = calib.screen_to_hanzi(float(user_pts_px[i, 0]), float(user_pts_px[i, 1]))
        out[i, 0], out[i, 1] = hx, hy
    return out


def compare_stroke_direction(user_pts, ref_pts, calib) -> tuple[bool, float, float]:
    """
    Compare the current (partial) user stroke against the reference median.

    Strategy
    ────────
    1.  Compute frac = how far along the USER stroke we currently are,
        as a fraction of the user stroke's OWN arc-length so far (always 1.0
        while drawing — we use the full drawn prefix each time).
    2.  Compute the SAME fractional prefix of the REFERENCE stroke.
        This is progress-relative matching: "after drawing this much, are
        you going in the right direction?"
    3.  Compute dominant_direction() for each prefix independently.
        This is purely a direction comparison — scale, position, and
        coordinate system differences are all factored out.

    Returns
    ───────
    (ok, angle_deg, progress_frac)
        ok            – True if angle ≤ MAX_DIR_ANGLE_DEG
        angle_deg     – angle between user direction and reference direction
        progress_frac – how far along the reference stroke we've compared to
                        (0–1, based on point count of the user stroke)
    """
    user_pts = np.asarray(user_pts, dtype=np.float32)
    ref_pts  = np.asarray(ref_pts,  dtype=np.float32)

    if len(user_pts) < 2 or len(ref_pts) < 2:
        return False, 0.0, 0.0

    # ── Step 1: how far along has the user drawn (0–1 of *their own* stroke)?
    # Since we only receive the drawn prefix, this is always 1.0 of
    # what exists so far — but we estimate reference progress using point
    # count ratio, which is coordinate-system agnostic.
    user_len_px = polyline_length(user_pts)
    if user_len_px < MIN_LIVE_LEN_PX or not calib.ready:
        return False, 0.0, 0.0

    # Convert pixel length -> hanzi length
    user_len_hz = calib.px_len_to_hanzi(user_len_px)

    ref_len_hz = polyline_length(ref_pts)  # ref_pts already in hanzi units
    if ref_len_hz < 1e-6:
        return False, 0.0, 0.0

    frac = float(np.clip(user_len_hz / ref_len_hz, 0.05, 1.0))
    ref_prefix = prefix_by_fraction(ref_pts, frac)

    # ── Step 3: direction comparison — fully scale/position agnostic.
    u_dir = dominant_direction(user_pts)
    r_dir = dominant_direction(ref_prefix)

    if u_dir is None or r_dir is None:
        return False, 0.0, frac

    # NOTE: the reference median uses a coordinate system where Y increases
    # downward (SVG/screen convention), same as OpenCV — no flip needed.
    angle = angle_between_dirs(u_dir, r_dir)
    ok    = angle <= MAX_DIR_ANGLE_DEG

    return ok, angle, frac


def compare_stroke_to_median_tangent(
    user_pts_px: np.ndarray,
    ref_pts_hanzi: np.ndarray,
    calib: Calibration,
) -> tuple[bool, float, float, np.ndarray | None]:
    """
    Scale the user's drawing into the same coordinate system as the median,
    find where we are along the median by arc-length progress, and compare
    the user's local tangent to the median's tangent at that point.

    So while the person is drawing, we:
    1. Convert user stroke (pixels) → hanzi space (0–900).
    2. Compute progress frac = user_arc_len / ref_arc_len (same units).
    3. Get the median point and unit tangent at that frac (curve-following).
    4. Compare user's local tangent (at stroke end) to median tangent.

    Returns
    ───────
    (ok, angle_deg, progress_frac, expected_point_hanzi)
        expected_point_hanzi: point on the median at current progress (for ghost drawing).
    """
    user_pts_px = np.asarray(user_pts_px, dtype=np.float32)
    ref_pts_hanzi = np.asarray(ref_pts_hanzi, dtype=np.float32)
    if len(user_pts_px) < 2 or len(ref_pts_hanzi) < 2 or not calib.ready:
        return False, 0.0, 0.0, None

    user_hanzi = user_pixels_to_hanzi(user_pts_px, calib)
    user_len = polyline_length(user_hanzi)
    ref_len = polyline_length(ref_pts_hanzi)
    if ref_len < 1e-6 or user_len < MIN_LIVE_LEN_PX * calib.inv_scale:
        return False, 0.0, 0.0, None

    frac = float(np.clip(user_len / ref_len, 0.02, 1.0))
    result = point_and_tangent_at_fraction(ref_pts_hanzi, frac)
    if result is None:
        return False, 0.0, frac, None
    median_point, median_tangent = result

    user_tangent = local_tangent_at_end(user_hanzi)
    if user_tangent is None:
        return False, 0.0, frac, median_point

    angle = angle_between_dirs(user_tangent, median_tangent)
    ok = angle <= MAX_DIR_ANGLE_DEG
    return ok, angle, frac, median_point


# ──────────────────────── Model download ─────────────────────────────────────

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

# ──────────────────────── Load hanzi data ────────────────────────────────────

try:
    char_data = load_hanzi_json(TARGET_CHAR, HANZI_DATA_DIR)
    medians   = char_data["medians"]   # list[list[[x, y]]]
    print(f"Loaded '{TARGET_CHAR}' with {len(medians)} median strokes.")
except Exception as e:
    print("Failed to load hanzi data:", e)
    sys.exit(1)

# Pre-convert all medians to float32; also to hanzi (900) space for tangent/scale comparison
ref_medians: list[np.ndarray] = [
    np.array(m, dtype=np.float32) for m in medians
]
ref_medians_hanzi: list[np.ndarray] = [
    median_to_hanzi(m) for m in ref_medians
]

# ──────────────────────── MediaPipe setup ────────────────────────────────────

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO,
)

# ──────────────────────── Webcam + main loop ─────────────────────────────────

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Camera failed to initialize.")
    sys.exit(1)

canvas      = None
drawing     = False
prev_point  = None
strokes: list[list[tuple[int, int]]] = []
stroke_idx  = 0
frame_count = 0

# Live feedback state (reset on each new stroke)
live_ok    : bool | None  = None
live_angle : float | None = None
live_frac  : float | None = None
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

        # ── MediaPipe inference ──────────────────────────────────────────────
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_count += 1
        result = hand_landmarker.detect_for_video(mp_image, frame_count * 33)

        finger_down = False
        tip_xy      = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            idx  = hand[INDEX_TIP]
            mid  = hand[MIDDLE_TIP]

            tip_xy          = (int(idx.x * w), int(idx.y * h))
            two_finger_dist = math.hypot(idx.x - mid.x, idx.y - mid.y)
            finger_down     = two_finger_dist < TWO_FINGER_THRESH_NORM

            # Finger visualisation
            p_idx = (int(idx.x * w), int(idx.y * h))
            p_mid = (int(mid.x * w), int(mid.y * h))
            cv2.circle(frame, p_idx, 6, (0, 255, 255), -1)
            cv2.circle(frame, p_mid, 6, (0, 255, 255), -1)
            cv2.line(frame, p_idx, p_mid, (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"pinch={two_finger_dist:.3f}",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
            )

        # ── Stroke capture ───────────────────────────────────────────────────
        if finger_down and tip_xy is not None:
            x, y = tip_xy

            if not drawing:
                drawing      = True
                prev_point   = (x, y)
                live_ok      = None
                live_angle   = None
                live_frac    = None
                live_counter = 0
                strokes.append([(x, y)])

            else:
                dist = math.hypot(x - prev_point[0], y - prev_point[1])
                if dist > MOVE_THRESHOLD_PX:
                    cv2.line(canvas, prev_point, (x, y), (0, 255, 0), LINE_THICKNESS)
                    strokes[-1].append((x, y))
                    prev_point = (x, y)

                    # ── Live direction check: scale to median, compare local tangents ──
                    live_counter += 1
                    if (
                        live_counter % LIVE_CHECK_EVERY_N == 0
                        and stroke_idx < len(ref_medians_hanzi)
                    ):
                        user_pts = np.array(strokes[-1], dtype=np.float32)
                        ref_hanzi = ref_medians_hanzi[stroke_idx]
                        ok, angle, frac, expected_pt_hanzi = compare_stroke_to_median_tangent(
                            user_pts, ref_hanzi, calib
                        )
                        if (
                            SHOW_EXPECTED_POINT
                            and expected_pt_hanzi is not None
                            and calib.ready
                        ):
                            ex, ey = calib.hanzi_to_screen(
                                float(expected_pt_hanzi[0]), float(expected_pt_hanzi[1])
                            )
                            cv2.circle(frame, (ex, ey), 10, (255, 255, 0), 2)

                        # Only update display if we had enough data
                        if angle > 0.0 or frac > 0.0:
                            live_ok    = ok
                            live_angle = angle
                            live_frac  = frac

        # ── Stroke end ───────────────────────────────────────────────────────
        if (not finger_down) and drawing:
            drawing    = False
            prev_point = None

            if stroke_idx < len(ref_medians_hanzi) and len(strokes[-1]) >= 2:
                user_pts = np.array(strokes[-1], dtype=np.float32)
                ref_hanzi = ref_medians_hanzi[stroke_idx]
                ok, angle, frac, _ = compare_stroke_to_median_tangent(
                    user_pts, ref_hanzi, calib
                )
                status = "✓ OK" if ok else "✗ WRONG DIRECTION"
                print(
                    f"Stroke {stroke_idx + 1}/{len(ref_medians)}  {status}"
                    f"  angle={angle:.1f}°  progress={frac * 100:.0f}%"
                )
            elif stroke_idx >= len(ref_medians):
                print(f"Extra stroke (character only has {len(ref_medians)} strokes).")
            else:
                print(f"Stroke {stroke_idx + 1}: too few points to evaluate.")

            stroke_idx  += 1
            live_ok      = None
            live_angle   = None
            live_frac    = None

        # ── Overlays ─────────────────────────────────────────────────────────
        total = len(ref_medians)
        if stroke_idx < total:
            header = f"'{TARGET_CHAR}'  stroke {stroke_idx + 1}/{total}"
        else:
            header = f"'{TARGET_CHAR}'  done ({total}/{total})  —  press C to restart"

        cv2.putText(
            frame, header,
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Live direction feedback while pen is down
        if drawing and stroke_idx < total:
            if live_ok is None:
                label  = "direction: draw more …"
                colour = (180, 180, 180)
            else:
                label  = (
                    f"direction: {'✓ OK' if live_ok else '✗ WRONG'}"
                    f"  {live_angle:.1f}°  (max {MAX_DIR_ANGLE_DEG}°)"
                )
                colour = (0, 220, 0) if live_ok else (0, 60, 255)

            cv2.putText(
                frame, label,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA,
            )
        
        frame = draw_calibration_overlay(frame, calib)
        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        cv2.imshow("Hanzi Stroke Trainer", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC → quit
            break
        elif key == ord("c"):
            canvas      = np.zeros_like(frame)
            strokes     = []
            drawing     = False
            prev_point  = None
            stroke_idx  = 0
            live_ok     = None
            live_angle  = None
            live_frac   = None
            live_counter = 0
            print("─── Restarted ───")

        elif key == ord("k"):
            # capture a calibration corner at current fingertip position
            if tip_xy is not None:
                calib.add_corner(tip_xy)
                print("Added corner:", tip_xy, "->", calib)

        elif key == ord("r"):
            calib.reset()
            print("Calibration reset.")


cap.release()
cv2.destroyAllWindows()