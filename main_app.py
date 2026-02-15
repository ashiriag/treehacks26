"""
Chinese Character Tutor - Main Application

Three Modes:
1. Teaching Mode: Learn to write (shows stroke guide with median animations)
2. Pinyin Recognition: See pinyin, recall character
3. English Translation: See English, recall character + gamification

Uses MakeMeAHanzi stroke data and median lines for stroke matching.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import sys
import time
import os
from typing import List, Tuple

from characters import (
    get_character,
    CharacterData,
    CHARACTER_LIST,
    get_random_character_info,
)

# ===============================
# Hand Detection Setup (Tasks API)
# ===============================

INDEX_FINGER_TIP = 8
THUMB_TIP = 4
PINCH_THRESHOLD_NORM = 0.06

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm
]

# Download hand_landmarker model if needed
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker model...")
    try:
        import urllib.request
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            MODEL_PATH
        )
        print("Model downloaded.")
    except Exception as e:
        print(f"Failed to download model: {e}")
        sys.exit(1)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# ===============================
# Constants
# ===============================

MOVE_THRESHOLD = 5
CAMERA_INDEX = 0
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
AUTO_ADVANCE_DELAY = 2.5

# Colors (BGR)
COLOR_GUIDE_OUTLINE = (120, 120, 120)    # Gray for character outline
COLOR_COMPLETED = (0, 180, 0)            # Green for completed strokes
COLOR_CURRENT_MEDIAN = (0, 200, 255)     # Yellow for current stroke guide
COLOR_ARROW = (255, 100, 0)              # Orange for animated arrow
COLOR_USER_STROKE = (100, 200, 100)      # Light green for user drawing
COLOR_USER_ACTIVE = (100, 255, 100)      # Bright green for active stroke
COLOR_TEXT = (100, 255, 100)             # Green text
COLOR_TEXT_DIM = (150, 150, 150)         # Gray text
COLOR_TEXT_TITLE = (150, 200, 255)       # Light blue for titles

STROKE_THICKNESS = 3


# ===============================
# Stroke Matching (Median-based)
# ===============================

def resample_stroke(pts, n=20):
    """Resample a polyline to n evenly-spaced points along its arc length."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        pt = pts[0] if len(pts) > 0 else np.array([0.0, 0.0])
        return np.tile(pt, (n, 1))

    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    if total_len < 1e-6:
        return np.tile(pts[0], (n, 1))

    targets = np.linspace(0, total_len, n)
    resampled = np.zeros((n, 2))
    for i, t in enumerate(targets):
        idx = np.searchsorted(cum_len, t, side='right') - 1
        idx = max(0, min(idx, len(pts) - 2))
        seg_start = cum_len[idx]
        seg_end = cum_len[idx + 1]
        alpha = (t - seg_start) / (seg_end - seg_start) if (seg_end - seg_start) > 0 else 0
        resampled[i] = pts[idx] + alpha * (pts[idx + 1] - pts[idx])

    return resampled


def match_stroke_to_median(user_pts, median_pts, char_size, threshold=0.20):
    """
    Compare user-drawn stroke to expected median, both in pixel coordinates.

    Algorithm:
    1. Resample both curves to N equidistant points
    2. Compute average point-to-point distance (forward and reversed)
    3. Normalize by char_size (character display dimension)
    4. Match if normalized distance < threshold AND direction is correct

    Returns dict: matched (bool), distance (float), direction_ok (bool).
    """
    if len(user_pts) < 2 or len(median_pts) < 2:
        return {'matched': False, 'distance': 1.0, 'direction_ok': True}

    n = 20
    user_rs = resample_stroke(np.array(user_pts, dtype=np.float64), n)
    median_rs = resample_stroke(np.array(median_pts, dtype=np.float64), n)

    # Forward distance (same direction)
    fwd_dists = np.sqrt(np.sum((user_rs - median_rs) ** 2, axis=1))
    fwd_avg = np.mean(fwd_dists)

    # Reverse distance (opposite direction)
    rev_dists = np.sqrt(np.sum((user_rs - median_rs[::-1]) ** 2, axis=1))
    rev_avg = np.mean(rev_dists)

    direction_ok = fwd_avg <= rev_avg
    best_dist = min(fwd_avg, rev_avg)
    norm_dist = best_dist / char_size if char_size > 0 else 1.0

    return {
        'matched': norm_dist < threshold and direction_ok,
        'distance': norm_dist,
        'direction_ok': direction_ok,
    }


# ===============================
# Application
# ===============================

class TutorApp:
    def __init__(self):
        self.mode = "mode_select"  # mode_select, teaching, pinyin, english

        # Character state
        self.char_info = None    # dict with char, pinyin, english
        self.char_data = None    # CharacterData instance
        self.current_stroke_idx = 0

        # Drawing state
        self.user_strokes = []           # Completed strokes (lists of pixel coords)
        self.current_user_stroke = []    # In-progress stroke points
        self.drawing = False
        self.prev_point = None

        # Display
        self.drawing_bbox = (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.char_size = min(WINDOW_WIDTH, WINDOW_HEIGHT)

        # Feedback & scoring
        self.feedback = []
        self.score = 0
        self.completed_characters = 0
        self.character_complete = False
        self.complete_time = None

        # Animation
        self.anim_start = time.time()

        # Camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            print("Error: Camera failed to initialize.")
            print("Check camera permissions in system settings.")
            sys.exit(1)

        self.frame_count = 0

    # ----------------------------
    # Character Management
    # ----------------------------

    def _compute_drawing_bbox(self, w, h):
        """Compute a centered square bbox for character display (with margin)."""
        size = int(min(w, h) * 0.85)
        x0 = (w - size) // 2
        y0 = (h - size) // 2
        self.drawing_bbox = (x0, y0, x0 + size, y0 + size)
        self.char_size = size

    def select_new_character(self):
        """Select a new random character for the current session."""
        self.char_info = get_random_character_info()
        self.char_data = get_character(self.char_info["char"])
        self.current_stroke_idx = 0
        self.user_strokes = []
        self.current_user_stroke = []
        self.character_complete = False
        self.complete_time = None
        self.feedback = []
        self.anim_start = time.time()

    # ----------------------------
    # Stroke Drawing & Matching
    # ----------------------------

    def finish_stroke(self):
        """Complete the current stroke and validate against the expected median."""
        if len(self.current_user_stroke) < 3:
            self.current_user_stroke = []
            return

        stroke = self.current_user_stroke[:]
        self.current_user_stroke = []

        if self.character_complete or not self.char_data:
            return

        if self.current_stroke_idx >= self.char_data.num_strokes:
            return

        # Get expected median in pixel coords
        median_px = self.char_data.get_stroke_midline(
            self.current_stroke_idx, self.drawing_bbox
        )
        if len(median_px) == 0:
            return

        result = match_stroke_to_median(stroke, median_px, self.char_size)

        if result['matched']:
            if self.mode == "teaching":
                self.user_strokes = [stroke]
            else:
                self.user_strokes.append(stroke)
            # self.user_strokes.append(stroke)
            self.current_stroke_idx += 1
            self.anim_start = time.time()

            if self.current_stroke_idx >= self.char_data.num_strokes:
                # Character complete!
                self.character_complete = True
                self.complete_time = time.time()
                self.completed_characters += 1

                if self.mode == "pinyin":
                    self.score += 150
                elif self.mode == "english":
                    self.score += 200
                else:
                    self.score += 100

                self.feedback.append(
                    f"Character {self.char_info['char']} complete! +points"
                )
            else:
                self.feedback.append(
                    f"Stroke {self.current_stroke_idx} correct!"
                )
        else:
            # Stroke didn't match — give feedback
            if not result['direction_ok']:
                self.feedback.append(
                    f"Stroke {self.current_stroke_idx + 1}: wrong direction! Try again."
                )
            else:
                self.feedback.append(
                    f"Stroke {self.current_stroke_idx + 1}: not quite right. Try again."
                )

    # ----------------------------
    # Hand Detection & Drawing Input
    # ----------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process camera frame: detect hand, track drawing input."""
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        self._compute_drawing_bbox(w, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.frame_count += 1
        timestamp_ms = self.frame_count * 33

        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        finger_detected = False

        if result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]
            tip = hand_lms[INDEX_FINGER_TIP]
            thumb = hand_lms[THUMB_TIP]
            x, y = int(tip.x * w), int(tip.y * h)

            # Pinch detection
            pinch_dist = np.hypot(tip.x - thumb.x, tip.y - thumb.y)

            if pinch_dist < PINCH_THRESHOLD_NORM:
                finger_detected = True
                if not self.drawing:
                    self.drawing = True
                    self.prev_point = (x, y)
                    self.current_user_stroke = [(x, y)]
                else:
                    if self.prev_point:
                        dist = np.hypot(
                            x - self.prev_point[0], y - self.prev_point[1]
                        )
                        if dist > MOVE_THRESHOLD:
                            self.current_user_stroke.append((x, y))
                            self.prev_point = (x, y)

            # Draw hand landmarks
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for px, py in pts:
                cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

        if not finger_detected and self.drawing:
            self.drawing = False
            self.prev_point = None
            self.finish_stroke()

        return frame

    # ----------------------------
    # Rendering Helpers
    # ----------------------------

    def _draw_animated_arrow(self, frame, median_pts, progress):
        """Draw an animated arrow traveling along the median polyline."""
        if len(median_pts) < 2:
            return

        pts = [(int(p[0]), int(p[1])) for p in median_pts]

        # Compute total length
        total_len = sum(
            np.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1])
            for i in range(len(pts) - 1)
        )
        target_len = total_len * progress

        cumulative = 0.0
        for i in range(len(pts) - 1):
            seg_len = np.hypot(
                pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1]
            )
            if cumulative + seg_len >= target_len:
                alpha = ((target_len - cumulative) / seg_len
                         if seg_len > 0 else 0)
                p = (
                    int(pts[i][0] + alpha * (pts[i+1][0] - pts[i][0])),
                    int(pts[i][1] + alpha * (pts[i+1][1] - pts[i][1]))
                )

                # Direction vector
                dx = pts[i+1][0] - pts[i][0]
                dy = pts[i+1][1] - pts[i][1]
                mag = np.hypot(dx, dy)
                if mag > 0:
                    dx, dy = dx / mag, dy / mag
                else:
                    break

                # Arrow line
                arrow_len = 25
                end = (int(p[0] + arrow_len * dx), int(p[1] + arrow_len * dy))
                cv2.line(frame, p, end, COLOR_ARROW, 3)

                # Arrowhead
                angle = np.arctan2(dy, dx)
                for da in [-np.pi / 6, np.pi / 6]:
                    hx = int(end[0] - 12 * np.cos(angle + da))
                    hy = int(end[1] - 12 * np.sin(angle + da))
                    cv2.line(frame, end, (hx, hy), COLOR_ARROW, 3)

                # Dot at current position
                cv2.circle(frame, p, 8, COLOR_ARROW, -1)
                break
            cumulative += seg_len

    def _draw_user_strokes(self, frame):
        """Draw user's completed and in-progress strokes."""
        for stroke in self.user_strokes:
            pts = np.array(stroke, dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(frame, [pts], False, COLOR_USER_STROKE, STROKE_THICKNESS)

        if len(self.current_user_stroke) >= 2:
            pts = np.array(self.current_user_stroke, dtype=np.int32)
            cv2.polylines(frame, [pts], False, COLOR_USER_ACTIVE, STROKE_THICKNESS)

    def _draw_drawing_box(self, frame):
        """Draw a faint reference box and crosshairs for the drawing area."""
        bbox = self.drawing_bbox
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        cv2.line(frame, (cx, bbox[1]), (cx, bbox[3]), (50, 50, 50), 1)
        cv2.line(frame, (bbox[0], cy), (bbox[2], cy), (50, 50, 50), 1)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (50, 50, 50), 1)

    def _draw_status_bar(self, frame, status_text):
        """Draw a semi-transparent status bar at the bottom of the frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Status line
        cv2.putText(frame, status_text, (20, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)

        # Latest feedback
        if self.feedback:
            cv2.putText(frame, self.feedback[-1], (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_DIM, 1)

    def _draw_shortcuts(self, frame):
        """Draw keyboard shortcut hints."""
        h, w = frame.shape[:2]
        shortcuts = "C:Clear  N:Next  M:Menu  Q:Quit"
        cv2.putText(frame, shortcuts, (w - 450, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_DIM, 1)

    # ----------------------------
    # Mode Renderers
    # ----------------------------

    def render_teaching_mode(self, frame: np.ndarray) -> np.ndarray:
        """Render teaching mode: character guide + user drawing."""
        display = frame.copy()
        h, w = display.shape[:2]

        if not self.char_data:
            self.select_new_character()

        bbox = self.drawing_bbox
        cd = self.char_data

        # 1. Draw faint reference box
        self._draw_drawing_box(display)

        # 2. Draw character union outline (gray guide)
        cd.draw_union(display, bbox, color=COLOR_GUIDE_OUTLINE, thickness=2)

        # 3. Fill completed strokes (green)
        for i in range(self.current_stroke_idx):
            cd.draw_stroke(display, i, bbox, filled=True, color=COLOR_COMPLETED)

        # 4. Draw current stroke median (yellow guide) with animated arrow
        if self.current_stroke_idx < cd.num_strokes:
            cd.draw_stroke_midline(
                display, self.current_stroke_idx, bbox,
                color=COLOR_CURRENT_MEDIAN, thickness=3
            )
            median_px = cd.get_stroke_midline(self.current_stroke_idx, bbox)
            if len(median_px) > 0:
                progress = ((time.time() - self.anim_start) % 2.0) / 2.0
                self._draw_animated_arrow(display, median_px, progress)

        # 5. Draw user strokes
        self._draw_user_strokes(display)

        # 6. UI text
        title = (
            f"Teaching: {self.char_info['char']}  "
            f"({self.char_info['pinyin']}) - {self.char_info['english']}"
        )
        cv2.putText(display, title, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT, 2)

        if self.character_complete:
            status = "Character complete! (auto-advancing...)"
        else:
            status = f"Stroke {self.current_stroke_idx + 1} / {cd.num_strokes}"

        self._draw_status_bar(display, status)
        self._draw_shortcuts(display)

        return display

    def render_recall_mode(self, frame: np.ndarray) -> np.ndarray:
        """Render recall mode (pinyin or english): user draws from memory."""
        display = frame.copy()
        h, w = display.shape[:2]

        if not self.char_data:
            self.select_new_character()

        # Mode-specific prompt
        if self.mode == "pinyin":
            mode_title = "Pinyin Mode"
            prompt = f"Write: {self.char_info['pinyin']}"
        else:
            mode_title = "Translation Mode"
            prompt = f"Write: {self.char_info['english'].upper()}"

        cv2.putText(display, mode_title, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT_TITLE, 2)
        cv2.putText(display, prompt, (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT_TITLE, 2)

        # Score
        cv2.putText(display, f"Score: {self.score:05d}", (w - 250, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT, 2)

        bbox = self.drawing_bbox

        # Faint reference box (guides position)
        self._draw_drawing_box(display)

        # Draw user strokes
        self._draw_user_strokes(display)

        # On completion, reveal the character
        if self.character_complete:
            cd = self.char_data
            cd.draw_union(display, bbox, color=COLOR_COMPLETED, thickness=2)
            for i in range(cd.num_strokes):
                cd.draw_stroke(display, i, bbox, filled=True, color=COLOR_COMPLETED)

        # Status
        if self.character_complete:
            status = (
                f"Correct! {self.char_info['char']} "
                f"({self.char_info['pinyin']})"
            )
        else:
            status = (
                f"Stroke {self.current_stroke_idx + 1} / "
                f"{self.char_data.num_strokes}"
            )

        self._draw_status_bar(display, status)
        self._draw_shortcuts(display)

        return display

    def render_mode_select(self, frame: np.ndarray) -> np.ndarray:
        """Render mode selection screen."""
        display = frame.copy()
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

        h, w = display.shape[:2]

        # Title
        cv2.putText(display, "Chinese Character Tutor",
                    (w // 2 - 300, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 200, 100), 3)

        # Mode options
        modes = [
            ("1", "Teaching Mode", "Learn stroke-by-stroke with guides"),
            ("2", "Pinyin Recognition", "See pinyin, recall character"),
            ("3", "English Translation", "See English, recall character"),
            ("Q", "Quit", "Exit application"),
        ]

        y = 200
        for key, title, desc in modes:
            cv2.putText(display, f"[{key}] {title}", (60, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT_TITLE, 2)
            cv2.putText(display, f"    {desc}", (100, y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT_DIM, 1)
            y += 120

        cv2.putText(display, "Press a key to select...",
                    (w // 2 - 200, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)

        return display

    # ----------------------------
    # Input Handling
    # ----------------------------

    def handle_key(self, key: int) -> bool:
        """Handle keyboard input. Returns False to quit."""
        if self.mode == "mode_select":
            if key == ord('1'):
                self.mode = "teaching"
                self.select_new_character()
            elif key == ord('2'):
                self.mode = "pinyin"
                self.select_new_character()
            elif key == ord('3'):
                self.mode = "english"
                self.select_new_character()
            elif key == ord('q'):
                return False
        else:
            if key == ord(' '):
                # Space: advance if character is complete
                if self.character_complete:
                    self.select_new_character()
            elif key == ord('c'):
                # Clear: restart current character
                self.current_stroke_idx = 0
                self.user_strokes = []
                self.current_user_stroke = []
                self.character_complete = False
                self.complete_time = None
                self.feedback = []
                self.anim_start = time.time()
            elif key == ord('n'):
                # Next: skip to a new character
                self.select_new_character()
            elif key == ord('m'):
                # Menu: return to mode selection
                self.mode = "mode_select"
            elif key == ord('q'):
                return False

        return True

    # ----------------------------
    # Main Loop
    # ----------------------------

    def run(self):
        """Main application loop."""
        cv2.namedWindow("Chinese Character Tutor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Chinese Character Tutor", WINDOW_WIDTH, WINDOW_HEIGHT)

        running = True
        while running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break

            frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

            # Hand detection + drawing input
            frame = self.process_frame(frame)

            # Render current mode
            if self.mode == "mode_select":
                display = self.render_mode_select(frame)
            elif self.mode == "teaching":
                display = self.render_teaching_mode(frame)
            elif self.mode in ("pinyin", "english"):
                display = self.render_recall_mode(frame)
            else:
                display = frame.copy()

            # Auto-advance after character completion
            if self.complete_time is not None:
                elapsed = time.time() - self.complete_time
                if elapsed >= AUTO_ADVANCE_DELAY:
                    self.select_new_character()

            cv2.imshow("Chinese Character Tutor", display)

            key = cv2.waitKey(30) & 0xFF
            if key != 255:
                running = self.handle_key(key)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = TutorApp()
    app.run()
