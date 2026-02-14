"""
Zoom Integration Layer
Makes the tutor compatible with Zoom screen sharing.
The application windows are designed to be shareable as-is.
"""

import cv2
import numpy as np
from typing import Optional
import threading
import queue

class ZoomCompatibleRenderer:
    """
    Renderer optimized for Zoom screen sharing.
    - High contrast graphics
    - Large text for visibility
    - Designed for 16:9 aspect ratio (1280x720 or 1920x1080)
    """
    
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.frame_queue = queue.Queue(maxsize=2)
    
    def ensure_zoom_compatible(self, canvas: np.ndarray) -> np.ndarray:
        """
        Ensure canvas is optimized for Zoom.
        - Sufficient contrast
        - No camera feed overlay (it competes for attention)
        - Large readable fonts
        """
        # This is already handled in the main app.
        # To use in Zoom:
        # 1. Launch the tutor app
        # 2. In Zoom, click "Share Screen"
        # 3. Select the "Chinese Character Tutor" window
        # 4. The app will appear at full resolution for all participants
        
        return canvas
    
    @staticmethod
    def setup_for_zoom():
        """Print setup instructions for Zoom integration."""
        instructions = """
═══════════════════════════════════════════════════════════════
  ZOOM INTEGRATION SETUP
═══════════════════════════════════════════════════════════════

The Chinese Character Tutor is Zoom-compatible by default!

HOW TO USE WITH ZOOM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Start a Zoom meeting
2. Launch the tutor application (python main_app.py)
3. In Zoom meeting controls, click "Share Screen"
4. Select the "Chinese Character Tutor" window
5. Everyone in the meeting will see your tutor session!

DESIGN FEATURES FOR ZOOM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ High contrast UI (dark background, bright text)
✓ Large fonts for readability on screen share
✓ Clean layout without clutter
✓ 1280x720 resolution (optimal for Zoom)
✓ No camera overlay (clean instructor view)
✓ Real-time feedback visible to all participants
✓ Animations and visual indicator adjustments

TEACHING WITH ZOOM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEACHING MODE:
- Students see animated strokes showing proper technique
- Can pause and ask questions (press 'P' to pause)
- Instant visual feedback helps students learn

GAMIFICATION MODES (Pinyin & English):
- Make it interactive: ask students to write in their cameras
- You see their attempts on tutor, they see same feedback
- Great for virtual classroom engagement

INSTRUCTOR TIPS:
1. Use "Teaching Mode" to demonstrate character writing
2. Switch to "Pinyin/English" modes for student practice
3. Keep the window at native resolution (don't zoom browser)
4. Use "Presenter View" so only you see the next character
5. For hybrid: project the screen on smart board + Zoom share

KEYBOARD SHORTCUTS (press these during share):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPACE   - Submit/Next character
C       - Clear current drawing
M       - Return to mode select
Q       - Quit application

═══════════════════════════════════════════════════════════════
        """
        print(instructions)


# ===============================
# Virtual Tutor (Headless mode)
# ===============================

class HeadlessTutorServer:
    """
    Run tutor as a backend service for Zoom apps/bots.
    (Future enhancement for direct Zoom app integration)
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        self.host = host
        self.port = port
        # Would integrate with Flask/FastAPI for API endpoints
        # Allows Zoom bots to request character data and scoring
    
    def start(self):
        """Start server (placeholder for future implementation)."""
        pass


if __name__ == "__main__":
    ZoomCompatibleRenderer.setup_for_zoom()
