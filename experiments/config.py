"""
Configuration file for Chinese Character Tutor
Easily customize behavior, difficulty, and appearance
"""

# ===============================
# WINDOW & DISPLAY
# ===============================

# Window dimensions (Zoom-optimized)
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Show camera feed overlay in corner (disable for clean Zoom share)
SHOW_CAMERA_OVERLAY = False

# UI Color scheme (B, G, R in OpenCV)
UI_COLORS = {
    "background": (20, 20, 20),
    "title_bg": (50, 50, 50),
    "text_white": (255, 255, 255),
    "text_gray": (200, 200, 200),
    "text_bright": (100, 255, 100),
    "correct_green": (0, 255, 0),
    "error_red": (0, 0, 255),
    "template_gray": (200, 200, 200),
    "guide_green": (150, 255, 150),
    "arrow_orange": (255, 100, 0),
    "user_stroke": (0, 255, 0),
    "completed_stroke": (0, 200, 100),
}

# ===============================
# HAND DETECTION
# ===============================

# Camera settings
CAMERA_INDEX = 0  # 0 = default camera, adjust if needed
CAMERA_FPS = 30

# Hand detection sensitivity (detection confidence threshold)
HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.7

# Finger position threshold for "drawing" (negative z means close to camera)
# Lower = closer to camera required
Z_THRESHOLD = -0.05

# Minimum pixel movement to register as drawing
MOVE_THRESHOLD = 5

# ===============================
# STROKE RECOGNITION
# ===============================

# DTW Resampling resolution (number of points per stroke)
RESAMPLE_POINTS = 64

# Stroke matching thresholds (lower = stricter)
# Score = DTW_Distance + (Angle_Penalty * 0.3)
MATCH_THRESHOLD_TEACHING = 0.25   # Stricter for learning
MATCH_THRESHOLD_PRACTICE = 0.30   # More forgiving for practice

# Character recognition threshold (minimum accuracy to show as possible)
RECOGNITION_THRESHOLD = 0.5

# Correct character threshold (automatic validation)
CORRECT_CHARACTER_THRESHOLD = 0.8  # 80% of strokes must match

# ===============================
# ANIMATION
# ===============================

# Arrow animation loop duration (seconds)
ARROW_ANIMATION_DURATION = 2.0

# Frame rate for smooth animation
TARGET_FPS = 30

# ===============================
# SCORING (Gamification)
# ===============================

# Points per mode
PINYIN_MODE_POINTS = 150
ENGLISH_MODE_POINTS = 200

# Bonus multipliers
COMBO_MULTIPLIER = 1.5  # 3+ correct in a row
SPEED_BONUS_THRESHOLD = 5.0  # seconds (faster than this gets bonus)

# ===============================
# DIFFICULTY LEVELS
# ===============================

DIFFICULTY_LEVELS = {
    "easy": {
        "threshold": 0.35,
        "max_tries": 5,
        "hint_frequency": "after_2_fails",
    },
    "normal": {
        "threshold": 0.25,
        "max_tries": 3,
        "hint_frequency": "after_3_fails",
    },
    "hard": {
        "threshold": 0.15,
        "max_tries": 2,
        "hint_frequency": "never",
    },
}

# ===============================
# CHARACTER DATABASE
# ===============================

CHARACTER_DB_PATH = "characters.json"

# Characters to include in random selection
# Leave empty to use all characters
# Example: ["一", "二", "十"] - only these three
ENABLED_CHARACTERS = []  # Empty = all

# ===============================
# UI TEXT & LABELS
# ===============================

MODE_NAMES = {
    "teaching": "🎓 Teaching Mode",
    "pinyin": "📝 Pinyin Recognition",
    "english": "📖 English Translation",
    "freestyle": "✏️ Freestyle",
}

STROKE_DIRECTIONS = {
    "left_to_right": "→ Left to Right",
    "top_to_bottom": "↓ Top to Bottom",
    "tilted_down_left": "↘ Tilted Down-Left",
    "tilted_down_right": "↙ Tilted Down-Right",
}

# ===============================
# FEEDBACK MESSAGES
# ===============================

FEEDBACK_MESSAGES = {
    "correct_stroke": "✓ Stroke {stroke_num} correct!",
    "correct_character": "🎉 Character {char} completed!",
    "incorrect_stroke": "✗ Stroke {stroke_num} incorrect. Try again!",
    "extra_stroke": "Too many strokes! Expected {expected}",
    "missing_stroke": "Missing stroke {stroke_num}",
    "wrong_direction": "Try drawing this stroke in the correct direction",
    "great_accuracy": "Great accuracy on that character!",
    "keep_practicing": "Keep practicing - you're getting better!",
}

# ===============================
# ZOOM INTEGRATION
# ===============================

# Use Zoom-optimized rendering (no camera overlay, high contrast)
ZOOM_MODE = False

# Optimal resolution for Zoom screen sharing
ZOOM_WIDTH = 1280
ZOOM_HEIGHT = 720

# ===============================
# ACCESSIBILITY
# ===============================

# Font sizes
FONT_SIZE_TITLE = 2
FONT_SIZE_SUBTITLE = 1.2
FONT_SIZE_NORMAL = 0.8
FONT_SIZE_SMALL = 0.6

# Text thickness
FONT_THICKNESS_NORMAL = 2
FONT_THICKNESS_BOLD = 3
FONT_THICKNESS_LIGHT = 1

# Stroke thickness
STROKE_THICKNESS_TEMPLATE = 3
STROKE_THICKNESS_USER = 5
STROKE_THICKNESS_GUIDE = 4
STROKE_THICKNESS_COMPLETED = 2

# ===============================
# KEYBOARD LAYOUT
# ===============================

KEYBOARD_LAYOUT = {
    "mode_teaching": "1",
    "mode_pinyin": "2",
    "mode_english": "3",
    "submit": " ",  # SPACE
    "clear": "c",
    "menu": "m",
    "quit": "q",
}

# ===============================
# DEBUG & DEVELOPMENT
# ===============================

# Enable debug output
DEBUG_MODE = False

# Show FPS counter
SHOW_FPS = True

# Show stroke recognition scores
SHOW_SCORES = False

# Draw bounding boxes and debug info
SHOW_DEBUG_OVERLAY = False

# ===============================
# ADVANCED: DTW Settings
# ===============================

# Dynamic Time Warping parameters
DTW_LOCAL_CONSTRAINT = "sakoe_chiba"
DTW_WINDOW_SIZE = 64

# Angle threshold for direction validation (degrees)
MAX_ANGLE_DIFF = 45
ANGLE_PENALTY_WEIGHT = 0.3

# ===============================
# SYSTEM SETTINGS
# ===============================

# Application name (for display)
APP_NAME = "Chinese Character Tutor"
APP_VERSION = "1.0.0"

# Frame capture timeout (seconds)
FRAME_TIMEOUT = 5.0

# Maximum hand travel distance per frame (for filtering noise)
MAX_HAND_SPEED = 200  # pixels per frame

# Character loading animation speed
CHARACTER_REVEAL_SPEED = 0.1  # seconds per character

def get_config(key: str, default=None):
    """Get configuration value by key."""
    parts = key.split(".")
    obj = globals()
    
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part, default)
        else:
            return default
    
    return obj if obj is not None else default


if __name__ == "__main__":
    # Print all configuration
    print("Chinese Character Tutor Configuration")
    print("=" * 50)
    print(f"Window: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    print(f"Camera: Index {CAMERA_INDEX}, {CAMERA_FPS} FPS")
    print(f"Character DB: {CHARACTER_DB_PATH}")
    print(f"Teaching Threshold: {MATCH_THRESHOLD_TEACHING}")
    print(f"Practice Threshold: {MATCH_THRESHOLD_PRACTICE}")
    print(f"Debug Mode: {DEBUG_MODE}")
    print(f"Zoom Mode: {ZOOM_MODE}")
