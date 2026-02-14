import cv2
import mediapipe as mp
import numpy as np
import sys
import time
import random
import json
import os

# ===============================
# Utils: geometry + matching
# ===============================

def dist(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))

def polyline_length(pts):
    if len(pts) < 2:
        return 0.0
    s = 0.0
    for i in range(1, len(pts)):
        s += dist(pts[i - 1], pts[i])
    return s

def resample_polyline(pts, n=64):
    """Resample polyline to exactly n points spaced by arc-length."""
    if len(pts) == 0:
        return [(0.0, 0.0)] * n
    if len(pts) == 1:
        return [pts[0]] * n

    pts = [(float(x), float(y)) for x, y in pts]
    L = polyline_length(pts)
    if L < 1e-6:
        return [pts[0]] * n

    # cumulative lengths
    dists = [0.0]
    for i in range(1, len(pts)):
        dists.append(dists[-1] + dist(pts[i - 1], pts[i]))

    # target distances
    targets = np.linspace(0.0, dists[-1], n)
    out = []
    j = 0
    for t in targets:
        while j < len(dists) - 2 and dists[j + 1] < t:
            j += 1
        d0, d1 = dists[j], dists[j + 1]
        p0, p1 = np.array(pts[j]), np.array(pts[j + 1])
        if abs(d1 - d0) < 1e-9:
            out.append(tuple(p0))
        else:
            alpha = (t - d0) / (d1 - d0)
            p = p0 + alpha * (p1 - p0)
            out.append((float(p[0]), float(p[1])))
    return out

def normalize_points(pts):
    """
    Normalize a stroke:
    - translate to centroid
    - scale to unit box (preserve aspect)
    """
    arr = np.array(pts, dtype=np.float32)
    if len(arr) == 0:
        return arr
    c = np.mean(arr, axis=0)
    arr = arr - c
    minxy = np.min(arr, axis=0)
    maxxy = np.max(arr, axis=0)
    size = np.max(maxxy - minxy)
    if size < 1e-6:
        size = 1.0
    arr = arr / size
    return arr

def dtw_distance(a, b):
    """DTW distance between two sequences of 2D points (numpy arrays shape [N,2])."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 1e9
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / (n + m))

def stroke_match_score(user_pts, tmpl_pts, resample_n=64):
    """
    Return a score where lower is better.
    We combine:
    - shape similarity (DTW on normalized strokes)
    - start/end direction consistency
    """
    u = resample_polyline(user_pts, resample_n)
    t = resample_polyline(tmpl_pts, resample_n)

    uN = normalize_points(u)
    tN = normalize_points(t)

    shape = dtw_distance(uN, tN)

    u0, u1 = np.array(u[0]), np.array(u[-1])
    t0, t1 = np.array(t[0]), np.array(t[-1])
    dir_u = u1 - u0
    dir_t = t1 - t0
    nu = np.linalg.norm(dir_u) + 1e-6
    nt = np.linalg.norm(dir_t) + 1e-6
    dir_u = dir_u / nu
    dir_t = dir_t / nt
    direction = float(1.0 - np.clip(np.dot(dir_u, dir_t), -1.0, 1.0))  # 0 is perfect

    return 0.75 * shape + 0.25 * direction

def bbox_of_strokes(strokes):
    pts = [p for s in strokes for p in s]
    if not pts:
        return None
    arr = np.array(pts, dtype=np.float32)
    x0, y0 = np.min(arr, axis=0)
    x1, y1 = np.max(arr, axis=0)
    return (float(x0), float(y0), float(x1), float(y1))

def draw_arrow_along_polyline(img, pts, t01, color=(0, 255, 255), thickness=3):
    """
    Draw a moving arrow tip along the stroke template.
    t01: 0..1 progression along polyline length.
    """
    if len(pts) < 2:
        return

    pts = [(int(x), int(y)) for x, y in pts]
    # compute cumulative lengths
    lens = [0.0]
    for i in range(1, len(pts)):
        lens.append(lens[-1] + dist(pts[i - 1], pts[i]))
    total = lens[-1]
    if total < 1e-6:
        return

    target = t01 * total
    j = 0
    while j < len(lens) - 2 and lens[j + 1] < target:
        j += 1

    # interpolate point on segment
    p0 = np.array(pts[j], dtype=np.float32)
    p1 = np.array(pts[j + 1], dtype=np.float32)
    d0, d1 = lens[j], lens[j + 1]
    if abs(d1 - d0) < 1e-6:
        p = p0
    else:
        a = (target - d0) / (d1 - d0)
        p = p0 + a * (p1 - p0)

    # arrow direction
    v = (p1 - p0)
    nv = np.linalg.norm(v) + 1e-6
    v = v / nv
    tip = (int(p[0]), int(p[1]))
    tail = (int(p[0] - v[0] * 35), int(p[1] - v[1] * 35))

    cv2.arrowedLine(img, tail, tip, color, thickness, tipLength=0.4)

# ===============================
# Character template library
# Coordinates are normalized in a 0..1 square
# Then mapped into a guide box on screen.
# ===============================

def load_character_db(path="characters.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "characters" not in data:
        raise ValueError("Invalid JSON format. Missing 'characters' key.")
    
    return data["characters"]

CHAR_DB = load_character_db()


def map_template_to_box(tmpl_strokes, box):
    """Map normalized strokes (0..1) into pixel box (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = box
    W = x1 - x0
    H = y1 - y0
    mapped = []
    for stroke in tmpl_strokes:
        pts = []
        for (u, v) in stroke:
            x = x0 + u * W
            y = y0 + v * H
            pts.append((float(x), float(y)))
        mapped.append(pts)
    return mapped

# ===============================
# Zoom compatibility later
# This function will return the final frame you want to send to:
# - a window (now)
# - a virtual cam (later)
# ===============================

class ZoomReadyRenderer:
    def __init__(self):
        self.last_frame = None

    def set_frame(self, frame_bgr):
        self.last_frame = frame_bgr

    def get_frame(self):
        return self.last_frame

    # Placeholder for later:
    # def stream_to_virtual_camera(self):
    #     Use pyvirtualcam to push self.last_frame into a virtual camera device.
    #     Zoom can then select that camera.

# ===============================
# MediaPipe Hand Setup
# ===============================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ===============================
# Webcam Setup (macOS built-in camera)
# ===============================

CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Camera failed to initialize.")
    print("Check System Settings → Privacy & Security → Camera")
    print("Make sure Terminal / IDE has camera access.")
    sys.exit(1)

# ===============================
# App configuration
# ===============================

MOVE_THRESHOLD = 5
Z_THRESHOLD = -0.05

MODE_LEARN = 1
MODE_RECOGNIZE = 2
MODE_TRANSLATE = 3

mode = MODE_LEARN

canvas = None
drawing = False
prev_point = None
current_stroke = []
user_strokes = []
stroke_finished_flash_t = 0.0

# scoring
score = 0
target_idx = 0
prompt_is_english = True  # used in translate mode
recognize_result = None
recognize_conf = 0.0

renderer = ZoomReadyRenderer()

# thresholds for matching
STROKE_PASS_THRESHOLD = 0.22   # lower is stricter
CHAR_PASS_THRESHOLD = 0.24

def reset_user_drawing():
    global drawing, prev_point, current_stroke, user_strokes, recognize_result, recognize_conf
    drawing = False
    prev_point = None
    current_stroke = []
    user_strokes = []
    recognize_result = None
    recognize_conf = 0.0

def pick_next_target():
    global target_idx, prompt_is_english
    target_idx = (target_idx + 1) % len(CHAR_DB)
    prompt_is_english = bool(random.getrandbits(1))

def compute_character_match(user_strokes_px, tmpl_strokes_px):
    """Match multi-stroke user drawing to template with stroke order enforced."""
    if len(user_strokes_px) != len(tmpl_strokes_px):
        return 1e9
    scores = []
    for us, ts in zip(user_strokes_px, tmpl_strokes_px):
        scores.append(stroke_match_score(us, ts))
    return float(np.mean(scores))

def best_character_recognition(user_strokes_px, box, top_k=3):
    """Return best char, confidence based on template matching."""
    if len(user_strokes_px) == 0:
        return None, 0.0, []

    candidates = []
    for entry in CHAR_DB:
        tmpl_px = map_template_to_box(entry["strokes"], box)
        if len(tmpl_px) != len(user_strokes_px):
            continue
        s = compute_character_match(user_strokes_px, tmpl_px)
        candidates.append((s, entry))

    if not candidates:
        return None, 0.0, []

    candidates.sort(key=lambda x: x[0])
    best_s, best_e = candidates[0]
    # convert score to confidence in 0..1
    conf = float(np.clip(1.0 - (best_s / 0.6), 0.0, 1.0))
    return best_e, conf, candidates[:top_k]

# ===============================
# Main loop
# ===============================

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to read frame from camera")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    # UI layout
    panel_w = int(0.32 * w)
    guide_margin = 20
    guide_size = int(min(h - 2 * guide_margin, w - panel_w - 2 * guide_margin))
    gx0 = panel_w + guide_margin
    gy0 = guide_margin
    gx1 = gx0 + guide_size
    gy1 = gy0 + guide_size
    guide_box = (gx0, gy0, gx1, gy1)

    # panel background
    cv2.rectangle(frame, (0, 0), (panel_w, h), (20, 20, 20), -1)

    # guide box
    cv2.rectangle(frame, (gx0, gy0), (gx1, gy1), (255, 255, 255), 2)

    # target
    target = CHAR_DB[target_idx]
    tmpl_px = map_template_to_box(target["strokes"], guide_box)

    # Animate current template stroke arrow
    now = time.time()
    anim_t = (now % 1.2) / 1.2
    if mode == MODE_LEARN:
        # show only current stroke to focus stroke order
        current_idx = min(len(user_strokes), len(tmpl_px) - 1)
        for i, stroke in enumerate(tmpl_px):
            # faint other strokes
            col = (80, 80, 80) if i != current_idx else (200, 200, 200)
            for j in range(1, len(stroke)):
                cv2.line(frame, (int(stroke[j - 1][0]), int(stroke[j - 1][1])),
                         (int(stroke[j][0]), int(stroke[j][1])), col, 3)
        if current_idx < len(tmpl_px):
            draw_arrow_along_polyline(frame, tmpl_px[current_idx], anim_t, color=(0, 255, 255), thickness=3)
    else:
        # show full faint character
        for stroke in tmpl_px:
            for j in range(1, len(stroke)):
                cv2.line(frame, (int(stroke[j - 1][0]), int(stroke[j - 1][1])),
                         (int(stroke[j][0]), int(stroke[j][1])), (80, 80, 80), 2)

    # Hand tracking
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    finger_detected = False
    drawing_allowed = False
    fingertip_xy = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y, z = int(tip.x * w), int(tip.y * h), tip.z
        fingertip_xy = (x, y)

        # Allow drawing only inside guide box for tutor behavior
        inside_guide = (gx0 <= x <= gx1 and gy0 <= y <= gy1)

        if z < Z_THRESHOLD and inside_guide:
            drawing_allowed = True
            finger_detected = True
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            if not drawing:
                drawing = True
                prev_point = (x, y)
                current_stroke = [(x, y)]
            else:
                d = dist(prev_point, (x, y)) if prev_point is not None else 999.0
                if d > MOVE_THRESHOLD:
                    cv2.line(canvas, prev_point, (x, y), (0, 255, 0), 5)
                    current_stroke.append((x, y))
                    prev_point = (x, y)
        else:
            drawing_allowed = False
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Stroke end (pen lift)
    if drawing and (not finger_detected):
        drawing = False
        prev_point = None

        # save stroke if meaningful
        if len(current_stroke) >= 6 and polyline_length(current_stroke) > 20:
            user_strokes.append(current_stroke)
            stroke_finished_flash_t = time.time()
        current_stroke = []

    # Mode specific logic
    status_lines = []
    status_color = (220, 220, 220)

    if mode == MODE_LEARN:
        status_lines.append("MODE: Learn (Trace strokes in order)")
        status_lines.append(f"Target: {target['char']}   Pinyin: {target['pinyin']}")

        # Check per-stroke correctness as they complete them
        if len(user_strokes) > 0:
            idx = len(user_strokes) - 1
            if idx < len(tmpl_px):
                s = stroke_match_score(user_strokes[idx], tmpl_px[idx])
                if s < STROKE_PASS_THRESHOLD:
                    status_lines.append(f"Stroke {idx+1}/{len(tmpl_px)}: OK  score={s:.3f}")
                    status_color = (0, 255, 120)
                else:
                    status_lines.append(f"Stroke {idx+1}/{len(tmpl_px)}: Try again  score={s:.3f}")
                    status_color = (0, 140, 255)

        # When full stroke count reached, evaluate full character
        if len(user_strokes) == len(tmpl_px):
            char_score = compute_character_match(user_strokes, tmpl_px)
            if char_score < CHAR_PASS_THRESHOLD:
                status_lines.append(f"Character correct!  score={char_score:.3f}")
                status_color = (0, 255, 120)
            else:
                status_lines.append(f"Character not quite.  score={char_score:.3f}")
                status_color = (0, 140, 255)

    elif mode == MODE_RECOGNIZE:
        status_lines.append("MODE: Recognize (Write a character)")
        status_lines.append("Press ENTER to recognize, or R to reset drawing")

    elif mode == MODE_TRANSLATE:
        status_lines.append("MODE: Translate (Prompt -> Write the character)")
        if prompt_is_english:
            status_lines.append(f"Prompt: {target['english']}  (write the character)")
        else:
            status_lines.append(f"Prompt: {target['pinyin']}  (write the character)")
        status_lines.append(f"Score: {score}")

    # Flash stroke finished
    if time.time() - stroke_finished_flash_t < 0.35:
        cv2.putText(frame, "STROKE SAVED", (gx0, gy1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Draw current stroke in progress on top of frame for clarity
    if len(current_stroke) >= 2:
        for i in range(1, len(current_stroke)):
            cv2.line(frame,
                     (int(current_stroke[i - 1][0]), int(current_stroke[i - 1][1])),
                     (int(current_stroke[i][0]), int(current_stroke[i][1])),
                     (0, 255, 0), 3)

    # Panel text
    y_text = 40
    cv2.putText(frame, "Chinese Air Tutor (MVP)", (18, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_text += 35

    # instructions
    inst = [
        "Keys:",
        "1 Learn / Trace",
        "2 Recognize",
        "3 Translate",
        "N Next target",
        "R Reset drawing",
        "C Clear canvas",
        "ENTER Run recognize",
        "ESC Quit",
    ]
    for s in inst:
        cv2.putText(frame, s, (18, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        y_text += 22

    y_text += 10
    for s in status_lines:
        cv2.putText(frame, s, (18, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y_text += 26

    # Drawing allowed indicator
    if drawing_allowed:
        cv2.putText(frame, "DRAWING ENABLED", (gx0, gy0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NOT DRAWING", (gx0, gy0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Overlay canvas (for persistent ink)
    combined = cv2.addWeighted(frame, 0.8, canvas, 0.2, 0)

    # Store final frame for future Zoom virtual cam
    renderer.set_frame(combined)

    cv2.imshow("Chinese Air Tutor MVP", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord("c"):
        canvas = np.zeros_like(frame)
        reset_user_drawing()
        score = 0
    elif key == ord("r"):
        reset_user_drawing()
    elif key == ord("n"):
        pick_next_target()
        reset_user_drawing()
        canvas = np.zeros_like(frame)
    elif key == ord("1"):
        mode = MODE_LEARN
        reset_user_drawing()
        canvas = np.zeros_like(frame)
    elif key == ord("2"):
        mode = MODE_RECOGNIZE
        reset_user_drawing()
        canvas = np.zeros_like(frame)
    elif key == ord("3"):
        mode = MODE_TRANSLATE
        reset_user_drawing()
        canvas = np.zeros_like(frame)
    elif key == 13:  # ENTER
        if mode in [MODE_RECOGNIZE, MODE_TRANSLATE]:
            # run recognition
            best, conf, top = best_character_recognition(user_strokes, guide_box)
            if best is None:
                recognize_result = None
                recognize_conf = 0.0
            else:
                recognize_result = best
                recognize_conf = conf

            # display result
            if recognize_result is not None:
                msg = f"Recognized: {recognize_result['char']}  ({recognize_result['pinyin']})  conf={recognize_conf:.2f}"
                print(msg)

                # translate mode scoring
                if mode == MODE_TRANSLATE:
                    if recognize_result["char"] == target["char"]:
                        score += 1
                        pick_next_target()
                    else:
                        score = max(0, score - 1)

            # reset ink but keep score
            canvas = np.zeros_like(frame)
            reset_user_drawing()

cap.release()
cv2.destroyAllWindows()
