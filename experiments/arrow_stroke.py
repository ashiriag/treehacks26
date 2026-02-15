"""
finger_stroke_tutor.py

Webcam + MediaPipe Hands (Tasks API) fingertip drawing, stroke segmentation,
and per-stroke feedback (order + direction + shape) against MakeMeAHanzi medians.
Uses same MakeMeAHanzi→display transform as draw_svg (scale(1,-1) translate(0,-900)).

Controls:
  - ESC : quit
  - c   : clear canvas + reset strokes
  - p   : toggle pinch-to-draw (recommended)
  - z   : toggle z-to-draw
  - g   : toggle showing ghost template stroke (current expected stroke)

Data:
  Download MakeMeAHanzi graphics JSONs and point DATA_DIR to the folder containing <char>.json
    https://github.com/skishore/makemeahanzi  (graphics/ data)
  Example:
    DATA_DIR = "./makemeahanzi_data/graphics"
    TARGET_CHAR = "永"

Notes:
  - Template medians are in a 1024x1024-ish coordinate system.
  - User strokes are normalized before comparison.
  - Thresholds are intentionally conservative; tune for your camera/setup.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import sys
import os
import json
import math
import time
import urllib.request

from draw_svg import makemeahanzi_to_box_px, draw_character

# -------------------------------
# Config
# -------------------------------
TARGET_CHAR = "十"
DATA_DIR = "./makemeahanzi"   # folder containing 永.json etc.

# Drawing / capture
INDEX_FINGER_TIP = 8
THUMB_TIP = 4

MOVE_THRESHOLD_PX = 5
LINE_THICKNESS = 5

# Pen-down signals
USE_PINCH = True           # toggle with 'p'
USE_Z = False              # toggle with 'z'
PINCH_THRESHOLD_NORM = 0.06  # distance in normalized coords (x,y) between thumb & index tips
Z_THRESHOLD = -0.05          # lower = closer to camera (tune); used if USE_Z True

# Feedback thresholds
DIRECTION_TOL_DEG = 35.0
DTW_THRESH = 35.0           # tune: lower = stricter, higher = more forgiving

# UI toggles
SHOW_GHOST = True           # toggle with 'g' – show/hide arrow guide on current stroke

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm
]

# -------------------------------
# Utilities: SVG path parsing (graphics.txt strokes) via svg.path
# -------------------------------
from svg.path import parse_path


def svg_path_to_polyline(path_d, n_samples=64):
    """
    Parse SVG path string and return list of (x,y) points for drawing.
    Uses svg.path parse_path; coordinates are in ~1024 space like medians.
    """
    path = parse_path(path_d)
    pts = []
    for i in range(n_samples + 1):
        t = i / n_samples
        p = path.point(t)
        pts.append((p.real, p.imag))
    return pts


def svg_strokes_to_polylines(hanzi_data):
    """
    Convert hanzi 'strokes' (SVG path strings) to list of polylines for display.
    Each polyline is list of (x,y) in MakeMeAHanzi space; use makemeahanzi_to_display
    when mapping to screen (as in draw_svg).
    """
    strokes = hanzi_data.get("strokes", [])
    return [svg_path_to_polyline(s) for s in strokes]


# -------------------------------
# Utilities: template loading
# -------------------------------
def load_hanzi_json(char, data_dir):
    path = os.path.join(data_dir, "graphics.txt")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}\n"
            f"Make sure graphics.txt is inside your makemeahanzi_data folder."
        )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("character") == char:
                return data

    raise ValueError(f"Character '{char}' not found in graphics.txt")


# -------------------------------
# Utilities: stroke preprocessing
# -------------------------------
def resample_polyline(pts, n=48):
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((n, 2), dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)

    deltas = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(deltas, axis=1)
    total = float(np.sum(seg_lens))
    if total < 1e-6:
        return np.repeat(pts[:1], n, axis=0)

    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    targets = np.linspace(0, total, n)

    out = []
    j = 0
    for t in targets:
        while j < len(cum) - 2 and cum[j + 1] < t:
            j += 1
        t0, t1 = cum[j], cum[j + 1]
        p0, p1 = pts[j], pts[j + 1]
        if t1 - t0 < 1e-6:
            out.append(p0)
        else:
            a = (t - t0) / (t1 - t0)
            out.append(p0 * (1 - a) + p1 * a)
    return np.vstack(out).astype(np.float32)

def normalize_to_1024(pts, pad=0.10):
    """
    Normalize a stroke to 1024x1024 using its own bbox (per-stroke).
    """
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 0:
        return pts

    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    size = mx - mn
    size[size < 1e-6] = 1.0

    mn = mn - pad * size
    mx = mx + pad * size
    size = mx - mn

    s = float(max(size[0], size[1]))
    pts01 = (pts - mn) / s
    pts1024 = pts01 * 1024.0
    return pts1024.astype(np.float32)

# -------------------------------
# Utilities: stroke comparison
# -------------------------------
def angle_deg(v):
    return math.degrees(math.atan2(float(v[1]), float(v[0])))

def angular_diff(a, b):
    d = (a - b + 180) % 360 - 180
    return abs(d)

def simple_dtw(A, B):
    """
    DTW distance between Nx2 arrays.
    Returns average per-step distance.
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    n, m = len(A), len(B)
    INF = 1e18
    dp = np.full((n + 1, m + 1), INF, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(A[i - 1] - B[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / (n + m))

def score_stroke(user_pts_px, template_median_1024,
                 direction_tol_deg=DIRECTION_TOL_DEG,
                 dtw_thresh=DTW_THRESH):
    # preprocess user
    u = resample_polyline(user_pts_px, n=48)
    u = normalize_to_1024(u)

    # preprocess template
    t = np.asarray(template_median_1024, dtype=np.float32)
    t = resample_polyline(t, n=48)

    # direction: start->end
    u_dir = angle_deg(u[-1] - u[0])
    t_dir = angle_deg(t[-1] - t[0])
    dir_err = angular_diff(u_dir, t_dir)
    direction_ok = dir_err <= direction_tol_deg

    # shape
    dtw = simple_dtw(u, t)
    shape_ok = dtw < dtw_thresh

    ok = direction_ok and shape_ok
    return ok, {
        "dir_err_deg": float(dir_err),
        "dtw": float(dtw),
        "direction_ok": bool(direction_ok),
        "shape_ok": bool(shape_ok),
    }

# -------------------------------
# UI drawing helpers
# -------------------------------
def draw_hand_landmarks(frame, hand_landmarks_list, w, h):
    for hand_landmarks in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for (x, y) in pts:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

def put_status(frame, lines, origin=(10, 30), line_h=26):
    x, y = origin
    for i, text in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(frame, text, (x, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def draw_ghost_stroke(frame, median_1024, bbox_px, color=(255, 255, 255)):
    """
    Draw template stroke (1024 coords) into a bbox on screen.
    bbox_px: (x0,y0,x1,y1) in image pixels where we render the ghost stroke.
    """
    x0, y0, x1, y1 = bbox_px
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)

    pts = np.asarray(median_1024, dtype=np.float32)
    if len(pts) < 2:
        return

    # median coords roughly in 0..1024, map into bbox
    # preserve aspect inside bbox
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    size = mx - mn
    size[size < 1e-6] = 1.0
    s = float(max(size[0], size[1]))
    pts01 = (pts - mn) / s

    # letterbox into bbox
    scale = min(w, h)
    ox = x0 + (w - scale) // 2
    oy = y0 + (h - scale) // 2

    pts_px = np.stack([
        ox + pts01[:, 0] * scale,
        oy + pts01[:, 1] * scale
    ], axis=1).astype(np.int32)

    for i in range(len(pts_px) - 1):
        cv2.line(frame, tuple(pts_px[i]), tuple(pts_px[i + 1]), color, 2)

    # start/end markers
    cv2.circle(frame, tuple(pts_px[0]), 5, (0, 255, 255), -1)   # start
    cv2.circle(frame, tuple(pts_px[-1]), 5, (0, 165, 255), -1)  # end


def _strokes_to_panel_px(strokes_points, bbox_px):
    """
    Convert stroke polylines (MakeMeAHanzi SVG coords) to pixel coords inside bbox.
    Uses same transform as draw_svg: scale(1,-1) translate(0,-900) -> (x, 900-y).
    Returns (all_pts_per_stroke, scale, offset) for consistent mapping.
    """
    x0, y0, x1, y1 = bbox_px
    pw = max(1, x1 - x0)
    ph = max(1, y1 - y0)

    all_pts = []
    for stroke in strokes_points:
        pts = np.asarray(stroke, dtype=np.float32)
        if len(pts) > 0:
            pts_display = makemeahanzi_to_box_px(pts, bbox_px)
            all_pts.extend(pts_display.tolist())

    if not all_pts:
        return [], 1.0, (x0, y0)

    all_arr = np.asarray(all_pts, dtype=np.float32)
    mn = all_arr.min(axis=0)
    mx = all_arr.max(axis=0)
    size = mx - mn
    size[size < 1e-6] = 1.0
    s = float(max(size[0], size[1]))
    scale = min(pw, ph) / s
    ox = x0 + (pw - min(pw, ph)) // 2
    oy = y0 + (ph - min(pw, ph)) // 2

    out = []
    for stroke in strokes_points:
        pts = np.asarray(stroke, dtype=np.float32)
        if len(pts) < 2:
            out.append([])
            continue
        pts = makemeahanzi_to_box_px(pts, bbox_px)
        pts01 = (pts - mn) / s
        pts_px = np.stack([
            ox + pts01[:, 0] * scale,
            oy + pts01[:, 1] * scale
        ], axis=1).astype(np.int32)
        out.append([tuple(p) for p in pts_px])
    return out, scale, (ox, oy)


def draw_full_character(frame, template_strokes, completed_strokes, current_stroke_idx, bbox_px):
    """
    Draw full character: all strokes visible (using SVG data).
    - Completed strokes: filled (thick green)
    - Incomplete strokes: hollow (thin outline)
    - current_stroke_idx: highlighted with hollow + arrow
    template_strokes: list of polylines (from SVG paths) for display
    """
    strokes_px = _strokes_to_panel_px(template_strokes, bbox_px)[0]
    if not strokes_px:
        return

    for i, pts in enumerate(strokes_px):
        if len(pts) < 2:
            continue
        if i in completed_strokes:
            # Filled: thick solid line
            for j in range(len(pts) - 1):
                cv2.line(frame, pts[j], pts[j + 1], (0, 200, 100), 5)
        else:
            # Hollow: thin outline
            color = (100, 255, 100) if i == current_stroke_idx else (180, 180, 180)
            for j in range(len(pts) - 1):
                cv2.line(frame, pts[j], pts[j + 1], color, 1)


def draw_animated_arrow_on_stroke(frame, stroke_points, bbox_px, progress, color=(255, 150, 0)):
    """
    Draw animated arrow along stroke showing direction.
    stroke_points: polyline (median or simplified) for arrow path.
    progress: 0..1, animates along stroke (use time for loop).
    """
    strokes_px = _strokes_to_panel_px([stroke_points], bbox_px)[0]
    if not strokes_px or len(strokes_px[0]) < 2:
        return

    pts = strokes_px[0]
    pts_arr = [np.array(p, dtype=np.float32) for p in pts]

    total_len = sum(np.linalg.norm(pts_arr[i + 1] - pts_arr[i]) for i in range(len(pts_arr) - 1))
    if total_len < 1e-6:
        return
    target_len = total_len * (progress % 1.0)

    cumulative = 0.0
    arrow_start = None
    arrow_end = None

    for i in range(len(pts_arr) - 1):
        seg_len = np.linalg.norm(pts_arr[i + 1] - pts_arr[i])
        if cumulative + seg_len >= target_len:
            alpha = (target_len - cumulative) / seg_len if seg_len > 0 else 0
            arrow_start = pts_arr[i] + alpha * (pts_arr[i + 1] - pts_arr[i])

            look_ahead = 25
            direction = (pts_arr[i + 1] - pts_arr[i]) / (seg_len + 1e-6)
            arrow_end = arrow_start + look_ahead * direction
            break
        cumulative += seg_len

    if arrow_start is not None and arrow_end is not None:
        start_pt = tuple(map(int, arrow_start))
        end_pt = tuple(map(int, arrow_end))
        cv2.line(frame, start_pt, end_pt, color, 3)

        # Arrowhead
        direction = arrow_end - arrow_start
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        angle = np.arctan2(direction[1], direction[0])
        arrow_size = 12
        pt1 = end_pt - arrow_size * np.array([np.cos(angle - np.pi / 6), np.sin(angle - np.pi / 6)])
        pt2 = end_pt - arrow_size * np.array([np.cos(angle + np.pi / 6), np.sin(angle + np.pi / 6)])
        cv2.line(frame, end_pt, tuple(map(int, pt1)), color, 3)
        cv2.line(frame, end_pt, tuple(map(int, pt2)), color, 3)


# ----------------------------
# MVP: semantic stroke grading
# ----------------------------
def stroke_features(points_px, n_resample=24):
    """
    Convert a raw stroke (list of (x,y) pixels) into robust features.
    Returns dict with:
      - start, end (2D)
      - dir_vec unit (2D)
      - angle_deg in [-180,180]
      - length
    """
    pts = np.asarray(points_px, dtype=np.float32)
    if len(pts) < 2:
        return None

    # light resample for stability
    pts = resample_polyline(pts, n=n_resample)

    start = pts[0]
    end = pts[-1]
    v = end - start
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return None
    v_unit = v / L
    ang = float(np.degrees(np.arctan2(v_unit[1], v_unit[0])))

    return {
        "start": start,
        "end": end,
        "v_unit": v_unit,
        "angle_deg": ang,
        "length": L,
        "pts": pts,
    }

def classify_angle(angle_deg):
    """
    Classify into 4 coarse stroke types by angle (in image coords: +x right, +y down).
    Returns one of: 'H', 'V', 'DL', 'DR'
      H  ~ horizontal
      V  ~ vertical
      DL ~ down-left (left-falling in writing, looks like \ in screen coords)
      DR ~ down-right (right-falling in writing, looks like / in screen coords)
    """
    a = (angle_deg + 180) % 360 - 180  # wrap
    aa = abs(a)

    # horizontal near 0 or 180
    if aa <= 25 or aa >= 155:
        return "H"
    # vertical near 90
    if 65 <= aa <= 115:
        return "V"
    # diagonals
    # angle around +45 => down-right, around +135 => down-left,
    # around -45 => up-right (usually wrong direction for most strokes), etc.
    if 0 < a < 90 or -180 < a < -90:
        return "DR"
    else:
        return "DL"

def angle_diff_deg(a, b):
    d = (a - b + 180) % 360 - 180
    return abs(d)

def normalize_points_to_char_space(all_points_px, pts_px):
    """
    Normalize points into [0,1024] using the bounding box of the whole character
    (all strokes drawn so far + current stroke).
    This makes region matching consistent across strokes.
    """
    allp = np.asarray(all_points_px, dtype=np.float32)
    if len(allp) < 2:
        # fallback: per-stroke normalization
        return normalize_to_1024(pts_px)

    mn = allp.min(axis=0)
    mx = allp.max(axis=0)
    size = mx - mn
    size[size < 1e-6] = 1.0
    s = float(max(size[0], size[1]))

    pts = np.asarray(pts_px, dtype=np.float32)
    pts01 = (pts - mn) / s
    return (pts01 * 1024.0).astype(np.float32)

def semantic_grade_stroke(user_stroke_px, template_median_1024, all_user_points_px,
                          dir_tol_deg=45.0,
                          start_dist_thresh=180.0):
    """
    Grade user stroke against template using:
      - type match
      - direction match (angle within tol)
      - start point near expected start (in normalized char space)

    Returns: ok, info dict
    """
    uf = stroke_features(user_stroke_px)
    if uf is None:
        return False, {"reason": "too_short"}

    # Template features (already in 1024 space)
    tf = stroke_features(template_median_1024)
    if tf is None:
        return False, {"reason": "template_bad"}

    # Compare stroke type
    u_type = classify_angle(uf["angle_deg"])
    t_type = classify_angle(tf["angle_deg"])
    type_ok = (u_type == t_type)

    # Compare direction (angle closeness)
    dir_err = angle_diff_deg(uf["angle_deg"], tf["angle_deg"])
    dir_ok = (dir_err <= dir_tol_deg)

    # Start region check:
    # map user's stroke start into character-normalized space based on ALL user points so far
    u_start_norm = normalize_points_to_char_space(all_user_points_px, np.array([uf["start"]]))[0]
    # template start already in 1024-ish space, but we should normalize template too for fairness
    # (template medians are generally 0..1024, so this is typically already OK)
    t_start = np.asarray(tf["start"], dtype=np.float32)

    start_dist = float(np.linalg.norm(u_start_norm - t_start))
    start_ok = (start_dist <= start_dist_thresh)

    ok = type_ok and dir_ok and start_ok

    info = {
        "u_type": u_type,
        "t_type": t_type,
        "type_ok": type_ok,
        "dir_err_deg": float(dir_err),
        "dir_ok": dir_ok,
        "start_dist": start_dist,
        "start_ok": start_ok,
    }

    # helpful feedback label
    if not type_ok:
        info["reason"] = "wrong_type"
    elif not dir_ok:
        info["reason"] = "wrong_direction"
    elif not start_ok:
        info["reason"] = "wrong_location"
    else:
        info["reason"] = "ok"

    return ok, info


# -------------------------------
# Download hand_landmarker model if needed
# -------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker model...")
    try:
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            MODEL_PATH
        )
        print("Model downloaded.")
    except Exception as e:
        print("Failed to download model:", e)
        print("Download manually from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        sys.exit(1)

# -------------------------------
# Load target character template
# -------------------------------
try:
    hanzi = load_hanzi_json(TARGET_CHAR, DATA_DIR)
    template_medians = hanzi.get("medians", [])
    if not template_medians:
        raise ValueError(f"{TARGET_CHAR}.json has no 'medians' field?")
except Exception as e:
    print("Template load error:", e)
    print("Set DATA_DIR to your MakeMeAHanzi graphics folder and ensure the JSON exists.")
    sys.exit(1)

template_svg_polylines = svg_strokes_to_polylines(hanzi)
print(f"Loaded template for '{TARGET_CHAR}' with {len(template_medians)} strokes (SVG for display).")

# -------------------------------
# MediaPipe Hand Setup (Tasks API)
# -------------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)

# -------------------------------
# Webcam Setup
# -------------------------------
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Camera failed to initialize.")
    print("Check System Settings → Privacy & Security → Camera")
    print("Make sure Terminal / IDE has camera access.")
    sys.exit(1)

# -------------------------------
# Drawing state
# -------------------------------
canvas = None
strokes = []
drawing = False
prev_point = None
completed_strokes = set()  # indices of correctly-drawn strokes (fill in character)

last_feedback = ""   # one-line feedback
last_info = None     # dict with dtw/dir errors
arrow_start_time = time.time()  # for arrow animation

# -------------------------------
# Main loop
# -------------------------------
frame_count = 0

with vision.HandLandmarker.create_from_options(options) as hand_landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from camera")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_count += 1
        timestamp_ms = frame_count * 33  # approx 30 fps

        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        finger_down = False
        tip_xy = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            tip = hand[INDEX_FINGER_TIP]
            thumb = hand[THUMB_TIP]

            tip_xy = (int(tip.x * w), int(tip.y * h))

            # pinch distance (normalized x/y)
            pinch_dist = math.hypot(tip.x - thumb.x, tip.y - thumb.y)

            if USE_PINCH and pinch_dist < PINCH_THRESHOLD_NORM:
                finger_down = True
            elif USE_Z and tip.z < Z_THRESHOLD:
                finger_down = True

            # draw landmarks for debugging
            draw_hand_landmarks(frame, result.hand_landmarks, w, h)

            # show pinch metric
            cv2.putText(frame, f"pinch={pinch_dist:.3f} z={tip.z:.3f}",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)

        # ---------------- stroke capture ----------------
        if finger_down and tip_xy is not None:
            x, y = tip_xy
            if not drawing:
                drawing = True
                prev_point = (x, y)
                strokes.append([(x, y)])
            else:
                dist = np.linalg.norm(np.array(prev_point) - np.array((x, y)))
                if dist > MOVE_THRESHOLD_PX:
                    cv2.line(canvas, prev_point, (x, y), (0, 255, 0), LINE_THICKNESS)
                    strokes[-1].append((x, y))
                    prev_point = (x, y)

        # ---------------- stroke end -> grade ----------------
        if (not finger_down) and drawing:
            drawing = False
            prev_point = None

            idx = len(strokes) - 1
            if idx < len(template_medians) and len(strokes[-1]) >= 4:
                # build "all points so far" for character-space normalization
                all_points = [p for stroke in strokes for p in stroke]

                ok, info = semantic_grade_stroke(
                    user_stroke_px=strokes[-1],
                    template_median_1024=template_medians[idx],
                    all_user_points_px=all_points,
                    dir_tol_deg=45.0,
                    start_dist_thresh=180.0
                )
                last_info = info
                if ok:
                    completed_strokes.add(idx)
                    last_feedback = f"Stroke {idx+1}: OK ✅"
                else:
                    reason = info.get("reason", "wrong")
                    if reason == "wrong_type":
                        last_feedback = f"Stroke {idx+1}: wrong stroke type ❌ (you={info['u_type']} expected={info['t_type']})"
                    elif reason == "wrong_direction":
                        last_feedback = f"Stroke {idx+1}: wrong direction ❌ (err={info['dir_err_deg']:.0f}°)"
                    elif reason == "wrong_location":
                        last_feedback = f"Stroke {idx+1}: wrong location ❌"
                    else:
                        last_feedback = f"Stroke {idx+1}: too short ❌"
                    print(last_feedback)
                    print(info)
            else:
                if idx >= len(template_medians):
                    last_feedback = f"Extra stroke {idx+1}: character has only {len(template_medians)} strokes"
                else:
                    last_feedback = f"Stroke {idx+1}: too short (try drawing longer)"
                last_info = None

        # ---------------- UI: full character with hollow/filled strokes + arrow ----------------
        current_stroke_idx = len(completed_strokes)
        panel = (w - 280, 10, w - 10, 280)
        cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (255, 255, 255), 2)
        cv2.putText(frame, f"'{TARGET_CHAR}'  Stroke {current_stroke_idx+1}/{len(template_medians)}",
                    (panel[0] + 6, panel[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        inner = (panel[0] + 10, panel[1] + 35, panel[2] - 10, panel[3] - 10)
        draw_character(frame, TARGET_CHAR, hanzi, inner)
        if SHOW_GHOST and current_stroke_idx < len(template_medians):
            arrow_progress = (time.time() - arrow_start_time) / 2.0  # 2 sec loop
            draw_animated_arrow_on_stroke(frame, template_svg_polylines[current_stroke_idx], inner, arrow_progress)

        # status text
        mode = []
        if USE_PINCH:
            mode.append("PINCH")
        if USE_Z:
            mode.append("Z")
        if not mode:
            mode.append("NONE")
        mode_str = "+".join(mode)

        lines = [
            f"Char: '{TARGET_CHAR}'  strokes: {len(strokes)}/{len(template_medians)}",
            f"Pen-down mode: {mode_str}   (p toggle pinch, z toggle z)",
            f"{last_feedback}" if last_feedback else "Draw the next stroke...",
        ]
        if last_info is not None:
            lines.append(
                f"type {last_info.get('u_type')}→{last_info.get('t_type')} | "
                f"dir_err={last_info.get('dir_err_deg', 0.0):.1f}° | "
                f"start_dist={last_info.get('start_dist', 0.0):.1f}"
            )
        lines.append("Keys: ESC quit | c clear | g ghost")

        put_status(frame, lines, origin=(10, 30), line_h=26)

        # composite
        combined = cv2.addWeighted(frame, 0.75, canvas, 0.25, 0)
        cv2.imshow("Chinese Stroke Tutor (MVP)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("c"):
            canvas = np.zeros_like(frame)
            strokes = []
            drawing = False
            prev_point = None
            completed_strokes = set()
            last_feedback = "Cleared."
            last_info = None
            print("Canvas cleared")
        elif key == ord("p"):
            USE_PINCH = not USE_PINCH
            last_feedback = f"Pinch-to-draw: {'ON' if USE_PINCH else 'OFF'}"
            print(last_feedback)
        elif key == ord("z"):
            USE_Z = not USE_Z
            last_feedback = f"Z-to-draw: {'ON' if USE_Z else 'OFF'}"
            print(last_feedback)
        elif key == ord("g"):
            SHOW_GHOST = not SHOW_GHOST
            last_feedback = f"Ghost: {'ON' if SHOW_GHOST else 'OFF'}"
            print(last_feedback)

cap.release()
cv2.destroyAllWindows()
