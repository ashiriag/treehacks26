#!/usr/bin/env python3
"""
SETUP GUIDE - Chinese Character Tutor

This script provides interactive setup assistance
"""

import os
import sys
import subprocess

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def main():
    print_header("Chinese Character Tutor - Setup Guide")
    
    guide = """
STEP 1: VERIFY VIRTUAL ENVIRONMENT
──────────────────────────────────────────────────────────

Your venv exists at: .venv/

To activate it:
    source .venv/bin/activate         (macOS/Linux)
    .venv\\Scripts\\activate          (Windows)

To verify it's working:
    which python                      (should show .venv path)
    python --version                 (should show 3.8+)


STEP 2: INSTALL DEPENDENCIES
──────────────────────────────────────────────────────────

Run this command (inside venv):
    pip install -r requirements.txt

This installs:
    ✓ OpenCV (cv2)         - real-time image processing
    ✓ MediaPipe            - hand detection
    ✓ NumPy                - numerical computing
    ✓ Plus 20+ other dependencies

Expected time: 3-5 minutes


STEP 3: VERIFY CAMERA ACCESS
──────────────────────────────────────────────────────────

macOS:
    1. Go to System Settings
    2. Privacy & Security → Camera
    3. Grant access to Terminal / Your IDE
    
Windows:
    1. Check Device Manager → Cameras
    2. Ensure camera is enabled
    3. Check if any privacy settings block it
    
Linux:
    1. ls /dev/video*
    2. Should show /dev/video0 or similar


STEP 4: TEST THE INSTALLATION
──────────────────────────────────────────────────────────

Run the quick test:
    python -m py_compile stroke_engine.py ui_renderer.py main_app.py

Expected output: (no errors)

Or test components individually in Python:

    python -c "import cv2; print('✓ OpenCV works')"
    python -c "import mediapipe; print('✓ MediaPipe works')"
    python -c "import numpy; print('✓ NumPy works')"


STEP 5: LAUNCH THE APPLICATION
──────────────────────────────────────────────────────────

Option A - Use the launcher (recommended):
    python launcher.py
    
    This will:
    ✓ Check all dependencies
    ✓ Verify camera access
    ✓ Show keyboard shortcuts
    ✓ Launch the application

Option B - Direct launch:
    python main_app.py

Expected: Mode selection screen appears


STEP 6: FIRST-TIME USAGE
──────────────────────────────────────────────────────────

1. SELECT A MODE:
   Press 1 - Teaching Mode (learn technique)
   Press 2 - Pinyin Mode (audio recall)
   Press 3 - English Mode (word recall + gamification)

2. POSITION YOUR HAND:
   - Sit 12-18 inches from camera
   - Ensure good lighting
   - Keep hand fully visible

3. DRAW YOUR FIRST CHARACTER:
   - Lift index finger in front of camera
   - Follow the animated guide
   - Draw the stroke in the correct direction
   - Lift finger to finish the stroke

4. GET FEEDBACK:
   - Green = correct stroke
   - Red feedback = needs adjustment
   - Press SPACE to submit or continue


STEP 7: ZOOM INTEGRATION
──────────────────────────────────────────────────────────

To use in Zoom:
    1. Start Zoom meeting
    2. Launch: python launcher.py
    3. In Zoom: Click "Share Screen"
    4. Select "Chinese Character Tutor" window
    5. ALL participants see live teaching!

For detailed instructions:
    python zoom_integration.py


STEP 8: CUSTOMIZATION
──────────────────────────────────────────────────────────

Edit config.py to customize:
    - Window size and colors
    - Camera sensitivity
    - Difficulty levels
    - Scoring system
    - Keyboard shortcuts

Add new characters to characters.json:
    {
        "char": "新",
        "pinyin": "xin1",
        "english": "new",
        "strokes": [[...], [...]],
        "stroke_directions": [...]
    }


TROUBLESHOOTING
──────────────────────────────────────────────────────────

Problem: "ModuleNotFoundError: No module named 'cv2'"
Solution: 
    - Ensure venv is activated
    - Run: pip install -r requirements.txt
    - Try: pip install --upgrade opencv-python

Problem: "Camera failed to initialize"
Solution:
    - macOS: Grant camera access to Terminal
    - Check System Settings → Privacy & Security
    - Try restarting the app
    - Try a different CAMERA_INDEX (2, 3, etc)

Problem: "Characters.json not found"
Solution:
    - Ensure you're in the project directory
    - Check file exists: ls characters.json
    - Copy if missing: cp template.json characters.json

Problem: Low frame rate / lag
Solution:
    - Close other applications
    - Update camera drivers
    - Reduce WINDOW_WIDTH/HEIGHT in config.py
    - Lower RESAMPLE_POINTS value

Problem: App crashes on startup
Solution:
    - Check console for error message
    - Verify Python version: python --version
    - Try reinstalling: pip install --force-reinstall -r requirements.txt
    - Check file permissions: chmod 755 *.py


PROJECT STRUCTURE
──────────────────────────────────────────────────────────

treehacks_2026/
├── main_app.py              ← Run this (or launcher.py)
├── launcher.py              ← Recommended entry point
├── stroke_engine.py         ← Stroke recognition engine
├── ui_renderer.py           ← Visual rendering system
├── config.py                ← Configuration file
├── zoom_integration.py      ← Zoom setup helper
├── characters.json          ← Character database
├── DEMO_GUIDE.py           ← Demo walkthrough
├── SETUP_GUIDE.py          ← This file
├── README.md               ← Comprehensive documentation
└── requirements.txt         ← Python dependencies


KEY FILES EXPLAINED
──────────────────────────────────────────────────────────

main_app.py (437 lines)
    - TutorApp class: main application controller
    - Mode management: teaching, pinyin, english
    - State management: current character, score, strokes
    - Hand detection: mediapipe integration
    - UI rendering: calling the renderer
    - Keyboard handling: mode selection, submit, etc

stroke_engine.py (250+ lines)
    - CharacterDatabase: loads/manages characters
    - DTW matching: matches user strokes to templates
    - Normalization: converts strokes to normalized space
    - match_stroke_to_template(): validates individual strokes
    - recognize_character(): identifies which character was drawn
    - validate_stroke_order(): checks stroke sequence

ui_renderer.py (200+ lines)
    - UIRenderer: main rendering class
    - draw_template_stroke(): shows character guides
    - draw_animated_arrow(): animated direction indicators
    - draw_user_stroke(): renders drawing in progress
    - draw_ui_panel(): status and feedback display
    - AnimationManager: timing and animation state

config.py (150+)
    - Window and display settings
    - Hand detection parameters
    - Stroke recognition thresholds
    - Scoring system values
    - UI colors and fonts
    - Keyboard layout

characters.json
    - 10 fundamental Chinese characters
    - Each with stroke data
    - Pinyin and English translations
    - Direction guidance for each stroke

launcher.py
    - Pre-launch checks (dependencies, camera)
    - User-friendly startup
    - Keyboard shortcut reference


PERFORMANCE BENCHMARKS
──────────────────────────────────────────────────────────

Expected Performance (on modern machine):
    - Frame capture: 30 FPS
    - Hand detection: 25-30 FPS
    - Stroke recognition: < 100ms per stroke
    - Rendering: 60 FPS (smooth)
    - Total latency: 100-150ms

On slower machines:
    - Reduce WINDOW_HEIGHT/WIDTH
    - Lower camera FPS
    - Reduce RESAMPLE_POINTS
    - Close other applications


API REFERENCE (for developers)
──────────────────────────────────────────────────────────

stroke_engine.py:
    CharacterDatabase(json_path)
        .load_characters()
        .get_character_by_index(idx)
        .get_random_character()
        .recognize_character(user_strokes, threshold)
    
    match_stroke_to_template(user, template, threshold)
        Returns: {matched, correct_strokes, accuracy, ...}
    
    validate_stroke_order(user, template, threshold)
        Returns: {correct_order, feedback, accuracy}

ui_renderer.py:
    UIRenderer(width, height)
        .draw_template_stroke(canvas, stroke, ...)
        .draw_user_stroke(canvas, stroke, ...)
        .draw_animated_arrow(canvas, stroke, progress)
        .draw_ui_panel(canvas, title, status, feedback)
    
    AnimationManager()
        .get_progress(duration)
        .reset()

main_app.py:
    TutorApp()
        .select_new_character()
        .submit_character()
        .process_frame(frame)
        .render_teaching_mode()
        .run()


NEXT STEPS
──────────────────────────────────────────────────────────

1. Complete basic setup (activate venv, install deps)
2. Try launcher.py to verify everything works
3. Read DEMO_GUIDE.py for learning path
4. Experiment with all three modes
5. Add your own custom characters
6. Try with Zoom for virtual teaching
7. Integrate into your Chinese learning workflow!


SUPPORT & TROUBLESHOOTING
──────────────────────────────────────────────────────────

For detailed documentation:
    - See README.md (comprehensive guide)
    - Run DEMO_GUIDE.py (interactive walkthrough)
    - Check config.py (all customization options)
    - Review source code comments (well documented)

Common issues already addressed in README.md:
    - Camera initialization
    - Permission issues
    - Stroke recognition accuracy
    - Zoom screen sharing
    - Character database management


YOU'RE ALL SET! 🚀

Now run:
    source .venv/bin/activate  (if not already activated)
    python launcher.py         (launch the app!)

Enjoy learning Chinese! 加油! 💚
"""
    
    print(guide)
    
    # Offer to run launcher
    print("\n" + "="*60)
    response = input("Ready to start? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\nActivating venv and launching...")
        try:
            subprocess.run(["./.venv/bin/python", "launcher.py"])
        except KeyboardInterrupt:
            print("\n\nLauncher closed.")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("\nSetup complete! Run 'python launcher.py' when ready.")


if __name__ == "__main__":
    main()
