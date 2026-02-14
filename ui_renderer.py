"""
UI and Rendering Layer
- Handles all visual display
- Zoom-compatible rendering
- Animations and feedback
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import math
import time

class UIRenderer:
    """Main UI rendering class."""
    
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.frame_time = time.time()
        self.animation_frame = 0
    
    def draw_template_stroke(
        self,
        canvas: np.ndarray,
        stroke: List[Tuple[float, float]],
        color: Tuple[int, int, int] = (200, 200, 200),
        thickness: int = 3,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Draw a template stroke (normalized 0-1 coordinates)."""
        h, w = canvas.shape[:2]
        
        # Draw area: center with padding
        padding = 50
        drawing_width = w - 2 * padding
        drawing_height = h - 2 * padding
        x_offset = padding
        y_offset = padding
        
        # Convert normalized coords to pixel coords
        points = []
        for norm_x, norm_y in stroke:
            px = x_offset + norm_x * drawing_width
            py = y_offset + norm_y * drawing_height
            points.append((int(px), int(py)))
        
        if len(points) < 2:
            return canvas
        
        # Draw with transparency
        overlay = canvas.copy()
        for i in range(len(points) - 1):
            cv2.line(overlay, points[i], points[i+1], color, thickness)
        
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        return canvas
    
    def draw_strokes(
        self,
        canvas: np.ndarray,
        strokes: List[List[Tuple[float, float]]],
        colors: Optional[List[Tuple[int, int, int]]] = None,
        thickness: int = 4
    ) -> np.ndarray:
        """Draw multiple strokes on canvas."""
        h, w = canvas.shape[:2]
        padding = 50
        drawing_width = w - 2 * padding
        drawing_height = h - 2 * padding
        x_offset = padding
        y_offset = padding
        
        if colors is None:
            colors = [(0, 255, 0)] * len(strokes)
        
        for stroke, color in zip(strokes, colors):
            points = []
            for norm_x, norm_y in stroke:
                px = x_offset + norm_x * drawing_width
                py = y_offset + norm_y * drawing_height
                points.append((int(px), int(py)))
            
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i+1], color, thickness)
        
        return canvas
    
    def draw_animated_arrow(
        self,
        canvas: np.ndarray,
        stroke: List[Tuple[float, float]],
        animation_progress: float = 0.5,
        color: Tuple[int, int, int] = (255, 100, 0)
    ) -> np.ndarray:
        """
        Draw animated arrow showing stroke direction.
        animation_progress: 0 = start, 1 = end
        """
        h, w = canvas.shape[:2]
        padding = 50
        drawing_width = w - 2 * padding
        drawing_height = h - 2 * padding
        x_offset = padding
        y_offset = padding
        
        if len(stroke) < 2:
            return canvas
        
        # Convert normalized coords
        points = []
        for norm_x, norm_y in stroke:
            px = x_offset + norm_x * drawing_width
            py = y_offset + norm_y * drawing_height
            points.append(np.array([px, py], dtype=np.float32))
        
        # Get segment based on animation progress
        total_len = sum(np.linalg.norm(points[i+1] - points[i]) for i in range(len(points)-1))
        target_len = total_len * animation_progress
        
        cumulative = 0.0
        arrow_start = None
        arrow_end = None
        
        for i in range(len(points) - 1):
            seg_len = np.linalg.norm(points[i+1] - points[i])
            if cumulative + seg_len >= target_len:
                # The arrow is in this segment
                alpha = (target_len - cumulative) / seg_len
                arrow_start = points[i] + alpha * (points[i+1] - points[i])
                
                # Arrow end (look ahead)
                look_ahead_len = 30
                if i + 1 < len(points) - 1:
                    next_seg_len = np.linalg.norm(points[i+2] - points[i+1])
                    if seg_len - alpha * seg_len + next_seg_len >= look_ahead_len:
                        alpha2 = look_ahead_len / (seg_len - alpha * seg_len)
                        arrow_end = points[i+1] + alpha2 * (points[i+2] - points[i+1])
                    else:
                        arrow_end = arrow_start + (points[i+1] - points[i]) / np.linalg.norm(points[i+1] - points[i]) * look_ahead_len
                else:
                    arrow_end = arrow_start + (points[i+1] - points[i]) / np.linalg.norm(points[i+1] - points[i]) * look_ahead_len
                break
            cumulative += seg_len
        
        if arrow_start is not None and arrow_end is not None:
            arrow_start = tuple(map(int, arrow_start))
            arrow_end = tuple(map(int, arrow_end))
            
            # Draw arrow line
            cv2.line(canvas, arrow_start, arrow_end, color, 3)
            
            # Draw arrowhead
            direction = np.array(arrow_end) - np.array(arrow_start)
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            
            # Arrowhead size
            arrow_size = 15
            angle = np.arctan2(direction[1], direction[0])
            
            pt1 = arrow_end - arrow_size * np.array([np.cos(angle - np.pi/6), np.sin(angle - np.pi/6)])
            pt2 = arrow_end - arrow_size * np.array([np.cos(angle + np.pi/6), np.sin(angle + np.pi/6)])
            
            cv2.line(canvas, arrow_end, tuple(map(int, pt1)), color, 3)
            cv2.line(canvas, arrow_end, tuple(map(int, pt2)), color, 3)
        
        return canvas
    
    def draw_user_stroke(
        self,
        canvas: np.ndarray,
        stroke: List[Tuple[float, float]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 5
    ) -> np.ndarray:
        """Draw user's current stroke in pixel coordinates (not normalized)."""
        if len(stroke) < 2:
            return canvas
        
        points = [tuple(map(int, pt)) for pt in stroke]
        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i+1], color, thickness)
        
        return canvas
    
    def draw_ui_panel(
        self,
        canvas: np.ndarray,
        title: str = "",
        subtitle: str = "",
        status: str = "",
        feedback: List[str] = None
    ) -> np.ndarray:
        """Draw UI panel with text information."""
        h, w = canvas.shape[:2]
        
        # Title bar
        cv2.rectangle(canvas, (0, 0), (w, 60), (50, 50, 50), -1)
        cv2.putText(canvas, title, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Subtitle
        if subtitle:
            cv2.putText(canvas, subtitle, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Status
        if status:
            status_color = (0, 255, 0) if "correct" in status.lower() else (0, 0, 255)
            cv2.putText(canvas, status, (20, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
        # Feedback
        if feedback:
            y_pos = h - 200
            for line in feedback[-5:]:  # Last 5 lines
                cv2.putText(canvas, line, (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y_pos += 25
        
        return canvas
    
    def draw_drawing_area_border(
        self,
        canvas: np.ndarray,
        color: Tuple[int, int, int] = (100, 100, 100),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw border around drawing area."""
        h, w = canvas.shape[:2]
        padding = 50
        
        cv2.rectangle(
            canvas,
            (padding, padding),
            (w - padding, h - padding),
            color,
            thickness
        )
        
        return canvas
    
    def draw_progress_bar(
        self,
        canvas: np.ndarray,
        progress: float,
        x: int,
        y: int,
        width: int = 200,
        height: int = 20
    ) -> np.ndarray:
        """Draw progress bar (0.0 to 1.0)."""
        progress = max(0.0, min(1.0, progress))
        
        # Background
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (50, 50, 50), -1)
        
        # Progress
        filled_width = int(width * progress)
        cv2.rectangle(canvas, (x, y), (x + filled_width, y + height), (0, 255, 0), -1)
        
        # Border
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (100, 100, 100), 1)
        
        # Text
        text = f"{int(progress * 100)}%"
        cv2.putText(canvas, text, (x + width // 2 - 15, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return canvas
    
    def draw_score_display(
        self,
        canvas: np.ndarray,
        score: int,
        x: int = None,
        y: int = None
    ) -> np.ndarray:
        """Draw score display (for gamified modes)."""
        h, w = canvas.shape[:2]
        if x is None:
            x = w - 120
        if y is None:
            y = 80
        
        # Background circle
        cv2.circle(canvas, (x, y), 50, (30, 30, 30), -1)
        cv2.circle(canvas, (x, y), 50, (100, 200, 100), 2)
        
        # Score text
        score_text = f"{score:04d}"
        cv2.putText(canvas, score_text, (x - 40, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
        
        return canvas
    
    def draw_mode_indicator(
        self,
        canvas: np.ndarray,
        mode: str,
        x: int = 20,
        y: int = 100
    ) -> np.ndarray:
        """Draw current mode indicator."""
        modes = {
            "teaching": "🎓 Teaching Mode",
            "pinyin": "📝 Pinyin Recognition",
            "english": "📖 English Translation",
            "freestyle": "✏️ Freestyle"
        }
        
        text = modes.get(mode, mode)
        cv2.putText(canvas, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 255), 2)
        
        return canvas


class AnimationManager:
    """Manage animations and timing."""
    
    def __init__(self):
        self.start_time = time.time()
        self.current_frame = 0
        self.fps = 30
    
    def get_progress(self, duration: float) -> float:
        """Get animation progress (0 to 1) for a given duration."""
        elapsed = time.time() - self.start_time
        progress = (elapsed % duration) / duration
        return progress
    
    def reset(self):
        """Reset animation."""
        self.start_time = time.time()
    
    def update(self):
        """Update frame."""
        self.current_frame += 1


if __name__ == "__main__":
    # Test rendering
    renderer = UIRenderer(1280, 720)
    
    # Create test canvas
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas.fill(20)
    
    # Test template stroke
    test_stroke = [[0.2, 0.5], [0.8, 0.5]]
    canvas = renderer.draw_template_stroke(canvas, test_stroke)
    
    # Test UI
    canvas = renderer.draw_ui_panel(canvas, "Test", "Subtitle", "Drawing...")
    canvas = renderer.draw_drawing_area_border(canvas)
    
    cv2.imshow("Test", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
