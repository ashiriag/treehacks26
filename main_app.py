"""
Chinese Character Tutor - Main Application
Three Modes:
1. Teaching Mode: Learn to write (shows stroke guide with animations)
2. Pinyin Recognition: See pinyin, recall character
3. English Translation: See English, recall character + gamification
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
import time
import random
from typing import List, Tuple, Optional, Dict

from stroke_engine import CharacterDatabase, match_stroke_to_template, validate_stroke_order
from ui_renderer import UIRenderer, AnimationManager

# ===============================
# Hand Detection Setup
# ===============================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ===============================
# Constants
# ===============================

MOVE_THRESHOLD = 5
Z_THRESHOLD = -0.05
CAMERA_INDEX = 0
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# ===============================
# Application State
# ===============================

class TutorApp:
    def __init__(self):
        self.mode = "mode_select"  # mode_select, teaching, pinyin, english
        self.character_db = CharacterDatabase("characters.json")
        self.renderer = UIRenderer(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.animator = AnimationManager()
        
        # Game state
        self.current_character = None
        self.current_stroke_idx = 0
        self.user_strokes = []
        self.current_user_stroke = []
        self.feedback = []
        self.score = 0
        self.character_history = []
        self.completed_characters = 0
        self.incorrect_strokes = []
        
        # Canvas
        self.canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        self.canvas.fill(20)
        
        # Camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            print("Error: Camera failed to initialize.")
            sys.exit(1)
        
        self.drawing = False
        self.prev_point = None
        
        # UI state
        self.mode_selected = False
        self.character_submitted = False
        self.character_correct = False
    
    def select_new_character(self):
        """Select a new character for the current mode."""
        self.current_character = self.character_db.get_random_character()
        self.user_strokes = []
        self.current_stroke_idx = 0
        self.character_submitted = False
        self.character_correct = False
        self.incorrect_strokes = []
        self.animator.reset()
        self.feedback = []
    
    def reset_for_next_character(self):
        """Reset state for next character."""
        self.select_new_character()
        if self.character_correct:
            self.completed_characters += 1
            if self.mode == "english" or self.mode == "pinyin":
                self.score += 100  # Base points
    
    def add_point_to_current_stroke(self, x: int, y: int):
        """Add a point to the current stroke being drawn."""
        self.current_user_stroke.append((x, y))
    
    def finish_stroke(self):
        """Finish the current stroke."""
        if len(self.current_user_stroke) >= 3:
            # Normalize stroke from pixel to 0-1
            normalized_stroke = self._normalize_to_template_space(self.current_user_stroke)
            self.user_strokes.append(normalized_stroke)
            
            # Validate this stroke
            if self.mode == "teaching":
                self._validate_teaching_stroke()
            
            self.current_user_stroke = []
    
    def _normalize_to_template_space(self, pixel_stroke: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Convert pixel coordinates to normalized 0-1 space (template space)."""
        if not pixel_stroke:
            return []
        
        # Drawing area
        padding = 50
        drawing_width = WINDOW_WIDTH - 2 * padding
        drawing_height = WINDOW_HEIGHT - 2 * padding
        x_offset = padding
        y_offset = padding
        
        normalized = []
        for px, py in pixel_stroke:
            norm_x = (px - x_offset) / drawing_width if drawing_width > 0 else 0
            norm_y = (py - y_offset) / drawing_height if drawing_height > 0 else 0
            # Clamp to 0-1
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            normalized.append((norm_x, norm_y))
        
        return normalized
    
    def _validate_teaching_stroke(self):
        """Validate stroke in teaching mode."""
        if not self.current_character:
            return
        
        template_strokes = self.current_character['strokes']
        
        if self.current_stroke_idx < len(template_strokes):
            template = template_strokes[self.current_stroke_idx]
            last_user_stroke = self.user_strokes[-1]
            
            result = match_stroke_to_template([last_user_stroke], [template], threshold=0.3)
            
            if result['accuracy'] >= 0.7:
                self.feedback.append(f"✓ Stroke {self.current_stroke_idx + 1} correct!")
                self.current_stroke_idx += 1
                
                # Check if character is complete
                if self.current_stroke_idx >= len(template_strokes):
                    self.character_correct = True
                    self.character_submitted = True
                    self.feedback.append(f"🎉 Character {self.current_character['char']} completed!")
            else:
                self.feedback.append(f"✗ Stroke {self.current_stroke_idx + 1} incorrect. Try again!")
    
    def submit_character(self):
        """Submit character drawing for evaluation."""
        if not self.current_character or self.character_submitted:
            return
        
        template_strokes = self.current_character['strokes']
        result = match_stroke_to_template(self.user_strokes, template_strokes, threshold=0.25)
        
        if result['matched']:
            self.character_correct = True
            self.feedback.append(f"✓ Character correct! {self.current_character['char']}")
            
            # Scoring
            if self.mode == "pinyin":
                self.score += 150
            elif self.mode == "english":
                self.score += 200
            
            self.completed_characters += 1
        else:
            self.character_correct = False
            accuracy_pct = int(result['accuracy'] * 100)
            self.feedback.append(f"✗ Only {accuracy_pct}% correct. Try again!")
            
            # Show which strokes were wrong
            for wrong in result['wrong_strokes']:
                idx = wrong['stroke_idx']
                self.feedback.append(f"  Stroke {idx + 1}: {wrong['reason']}")
                self.incorrect_strokes.append(idx)
        
        self.character_submitted = True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a camera frame and detect hand."""
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        finger_detected = False
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y, z = int(tip.x * w), int(tip.y * h), tip.z
            
            if z < Z_THRESHOLD:
                finger_detected = True
                cv2.circle(frame, (x, y), 12, (0, 255, 0), -1)
                
                if not self.drawing:
                    self.drawing = True
                    self.prev_point = (x, y)
                else:
                    if self.prev_point:
                        dist = np.linalg.norm(np.array(self.prev_point) - np.array((x, y)))
                        if dist > MOVE_THRESHOLD:
                            cv2.line(frame, self.prev_point, (x, y), (0, 255, 0), 5)
                            self.add_point_to_current_stroke(x, y)
                            self.prev_point = (x, y)
            else:
                cv2.circle(frame, (x, y), 12, (0, 0, 255), -1)
        
        if not finger_detected and self.drawing:
            self.drawing = False
            self.prev_point = None
            self.finish_stroke()
        
        return frame
    
    def render_teaching_mode(self) -> np.ndarray:
        """Render teaching mode interface."""
        canvas = self.canvas.copy()
        
        if not self.current_character:
            self.select_new_character()
        
        char = self.current_character
        template_strokes = char['strokes']
        
        # Title
        self.renderer.draw_ui_panel(
            canvas,
            title=f"Teaching Mode - {char['char']} ({char['english']})",
            subtitle=f"Pinyin: {char['pinyin']}"
        )
        
        # Draw border
        self.renderer.draw_drawing_area_border(canvas)
        
        # Draw completed strokes (in gray)
        if self.current_stroke_idx > 0:
            completed_colors = [(100, 100, 200)] * self.current_stroke_idx
            self.renderer.draw_strokes(canvas, template_strokes[:self.current_stroke_idx], completed_colors, thickness=2)
        
        # Draw current stroke guide with animation
        if self.current_stroke_idx < len(template_strokes):
            current_template = template_strokes[self.current_stroke_idx]
            
            # Animated arrow
            progress = self.animator.get_progress(2.0)  # 2 second animation loop
            self.renderer.draw_animated_arrow(canvas, current_template, progress)
            
            # Semi-transparent guide
            self.renderer.draw_template_stroke(canvas, current_template, color=(150, 255, 150), thickness=4, alpha=0.3)
        
        # Draw user's current stroke
        if self.current_user_stroke:
            self.renderer.draw_user_stroke(canvas, self.current_user_stroke, color=(0, 255, 0), thickness=5)
        
        # Draw already-drawn user strokes
        self.renderer.draw_strokes(canvas, self.user_strokes, [(0, 200, 100)] * len(self.user_strokes))
        
        # Status
        status = f"Stroke {self.current_stroke_idx + 1} / {len(template_strokes)}"
        if self.character_submitted:
            if self.character_correct:
                status = "✓ Character Complete!"
            else:
                status = "Try again or press SPACE for next"
        
        # Feedback
        self.renderer.draw_ui_panel(
            canvas,
            title=f"Teaching Mode - {char['char']}",
            status=status,
            feedback=self.feedback[-3:]
        )
        
        return canvas
    
    def render_pinyin_mode(self) -> np.ndarray:
        """Render Pinyin recognition mode."""
        canvas = self.canvas.copy()
        
        if not self.current_character:
            self.select_new_character()
        
        char = self.current_character
        
        # Title with pinyin hint
        self.renderer.draw_ui_panel(
            canvas,
            title=f"Pinyin Mode",
            subtitle=f"Write the character that sounds like: {char['pinyin']}"
        )
        
        # Draw border
        self.renderer.draw_drawing_area_border(canvas)
        
        # Draw user strokes
        self.renderer.draw_strokes(canvas, self.user_strokes, [(0, 200, 100)] * len(self.user_strokes))
        if self.current_user_stroke:
            self.renderer.draw_user_stroke(canvas, self.current_user_stroke, color=(0, 255, 0), thickness=5)
        
        # Status
        status = ""
        if self.character_submitted:
            if self.character_correct:
                status = f"✓ Correct! {char['char']}"
            else:
                status = "✗ Incorrect. Try again!"
        
        # Score
        self.renderer.draw_score_display(canvas, self.score)
        self.renderer.draw_mode_indicator(canvas, "pinyin")
        
        self.renderer.draw_ui_panel(
            canvas,
            title="Pinyin Mode",
            status=status,
            feedback=self.feedback[-3:]
        )
        
        return canvas
    
    def render_english_mode(self) -> np.ndarray:
        """Render English translation mode (gamified)."""
        canvas = self.canvas.copy()
        
        if not self.current_character:
            self.select_new_character()
        
        char = self.current_character
        
        # Title with English hint
        self.renderer.draw_ui_panel(
            canvas,
            title=f"Translation Mode",
            subtitle=f"Write the character for: {char['english'].upper()}"
        )
        
        # Draw border
        self.renderer.draw_drawing_area_border(canvas)
        
        # Draw user strokes
        self.renderer.draw_strokes(canvas, self.user_strokes, [(0, 200, 100)] * len(self.user_strokes))
        if self.current_user_stroke:
            self.renderer.draw_user_stroke(canvas, self.current_user_stroke, color=(0, 255, 0), thickness=5)
        
        # Status
        status = ""
        bonus_text = ""
        if self.character_submitted:
            if self.character_correct:
                status = f"✓ Correct! {char['char']}"
                bonus_text = "+200 Points!"
            else:
                status = "✗ Incorrect. Try again!"
        
        # Score display
        self.renderer.draw_score_display(canvas, self.score)
        self.renderer.draw_mode_indicator(canvas, "english")
        
        # Draw stats
        cv2.putText(canvas, f"Completed: {self.completed_characters}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 200, 255), 2)
        
        self.renderer.draw_ui_panel(
            canvas,
            title="Translation Mode",
            status=status,
            feedback=self.feedback[-3:]
        )
        
        return canvas
    
    def render_mode_select(self) -> np.ndarray:
        """Render mode selection screen."""
        canvas = self.canvas.copy()
        canvas.fill(20)
        
        h, w = canvas.shape[:2]
        
        # Title
        cv2.putText(canvas, "Chinese Character Tutor", (w//2 - 300, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 200, 100), 3)
        
        # Mode options
        modes = [
            ("1", "Teaching Mode", "Learn stroke-by-stroke with guides"),
            ("2", "Pinyin Recognition", "See pinyin, recall character"),
            ("3", "English Translation", "See English, recall character (Gamified)"),
            ("Q", "Quit", "Exit application")
        ]
        
        y = 200
        for key, title, desc in modes:
            cv2.putText(canvas, f"[{key}] {title}", (60, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 200, 255), 2)
            cv2.putText(canvas, f"    {desc}", (100, y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)
            y += 120
        
        cv2.putText(canvas, "Press a key to select...", (w//2 - 200, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
        
        return canvas
    
    def handle_key(self, key: int):
        """Handle keyboard input."""
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
            if key == ord(' '):  # Space to submit
                if not self.character_submitted:
                    self.submit_character()
                else:
                    self.reset_for_next_character()
            elif key == ord('c'):  # C to clear current drawing
                self.current_user_stroke = []
                self.user_strokes = []
                self.incorrect_strokes = []
                self.character_submitted = False
                self.character_correct = False
                self.feedback = []
            elif key == ord('m'):  # M to return to mode select
                self.mode = "mode_select"
            elif key == ord('q'):
                return False
        
        return True
    
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
            
            # Process hand detection
            frame = self.process_frame(frame)
            
            # Render current mode
            if self.mode == "mode_select":
                display_canvas = self.render_mode_select()
            elif self.mode == "teaching":
                display_canvas = self.render_teaching_mode()
            elif self.mode == "pinyin":
                display_canvas = self.render_pinyin_mode()
            elif self.mode == "english":
                display_canvas = self.render_english_mode()
            else:
                display_canvas = self.canvas.copy()
            
            # Overlay camera feed in corner (optional - set to False for Zoom compatibility)
            SHOW_CAMERA = False
            if SHOW_CAMERA:
                frame_resized = cv2.resize(frame, (320, 240))
                display_canvas[WINDOW_HEIGHT-250:WINDOW_HEIGHT-10, 10:330] = frame_resized
            
            # Display
            cv2.imshow("Chinese Character Tutor", display_canvas)
            
            # Handle input
            key = cv2.waitKey(30) & 0xFF
            if key != 255:
                running = self.handle_key(key)
            
            # Update animation
            self.animator.update()
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = TutorApp()
    app.run()
