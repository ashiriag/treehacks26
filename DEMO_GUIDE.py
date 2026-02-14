"""
Demo & Test Guide for Chinese Character Tutor

This guide helps you test all three modes and understand how the system works.
"""

DEMO_GUIDE = """
╔════════════════════════════════════════════════════════════════════════════╗
║                     CHINESE CHARACTER TUTOR - DEMO GUIDE                   ║
╚════════════════════════════════════════════════════════════════════════════╝

🎯 OBJECTIVE:
Learn how to write Chinese characters with real-time AI feedback and stroke
detection. Three modes for different learning styles.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 BEFORE YOU START:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ✓ Check camera is working
2. ✓ Ensure good lighting
3. ✓ Give Terminal camera permissions (macOS)
4. ✓ Clear table space for hand movements

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 QUICK TEST (5 minutes):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

$ python launcher.py

Expected: Mode selection screen
┌─────────────────────────────────────────┐
│ Chinese Character Tutor                 │
│                                         │
│ [1] Teaching Mode                       │
│ [2] Pinyin Recognition                  │
│ [3] English Translation                 │
│ [Q] Quit                                │
└─────────────────────────────────────────┘

ACTION: Press "1" to start Teaching Mode

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ TEST 1: TEACHING MODE (5 mins)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOAL: Learn to write character "一" (yi1 - one)

WHAT YOU'LL SEE:
┌─────────────────────────────────────────┐
│ Teaching Mode - 一 (one)                │
│ Pinyin: yi1                             │
├─────────────────────────────────────────┤
│                                         │
│         ➜ ➜ ➜ ➜  (ANIMATED ARROW)     │
│         ───────────────────   (GUIDE)  │
│                                         │
│ Stroke 1 / 1                           │
└─────────────────────────────────────────┘

WHAT TO DO:
1. Position hand in front of camera
2. Hold up your index finger
3. Follow the animated arrow direction
4. Draw the stroke horizontally (left to right)
5. When finished, lift your finger to end the stroke

EXPECTED RESULT:
- ✓ Your stroke appears in green
- ✓ "Stroke 1 correct!" feedback
- ✓ Character complete message

TIPS FOR SUCCESS:
✓ Move finger slowly and deliberately
✓ Keep finger visible to camera
✓ The guide line shows the exact path
✓ You don't need to be perfectly on the path

Next: Press SPACE to go to next character

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ TEST 2: PINYIN RECOGNITION (5 mins)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOAL: See pinyin, recall the character

WHAT YOU'LL SEE:
┌─────────────────────────────────────────┐
│ Pinyin Mode                             │
│ Write the character that sounds like:   │
│         shui3  (WATER)                  │
├─────────────────────────────────────────┤
│                                         │
│      [Your drawing appears here]        │
│                                         │
│                          Score: 0150   │
└─────────────────────────────────────────┘

WHAT TO DO:
1. Remember: "shui3" is the word for WATER
2. Draw the character from memory (it's 水)
3. For 水, you should draw:
   - Vertical line (top to bottom)
   - Horizontal line (left to right)
   - Horizontal line (left to right)
4. Press SPACE to submit

EXPECTED RESULT:
- ✓ If correct: "Correct! 水" → +150 points!
- ✗ If wrong: "Incorrect. Try again!"

DIFFICULTY:
- Medium difficulty (you have the audio hint)
- Good for building vocabulary
- Build muscle memory for characters

Try again or press SPACE for next character

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ TEST 3: ENGLISH TRANSLATION (Challenge!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOAL: See English, recall the character (hardest mode!)

WHAT YOU'LL SEE:
┌─────────────────────────────────────────┐
│ Translation Mode                        │
│ Write the character for:                │
│         FIRE                            │
├─────────────────────────────────────────┤
│                                         │
│      [Your drawing appears here]        │
│                                         │
│ Completed: 1          Score: 00200     │
└─────────────────────────────────────────┘

WHAT TO DO:
1. You see English: "FIRE"
2. Recall the Chinese character from memory (火 - huo3)
3. Draw it in the box:
   - Vertical line
   - Horizontal line
   - Down-left diagonal
   - Down-right diagonal
4. Press SPACE to submit

EXPECTED RESULT:
- ✓ Correct: "Correct! 火" → +200 points! (highest reward)
- ✗ Wrong: "Incorrect. Try again!"

DIFFICULTY:
- HARDEST mode (no hints)
- Full memory recall challenge
- Highest reward: 200 points vs 150 in Pinyin mode

GAMIFICATION:
- Score accumulates
- Completion counter increases
- Track your progress!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔑 KEYBOARD SHORTCUTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

│ KEY   │ ACTION                              │
├───────┼─────────────────────────────────────┤
│ 1     │ Launch Teaching Mode                │
│ 2     │ Launch Pinyin Recognition Mode      │
│ 3     │ Launch English Translation Mode     │
│ M     │ Return to Mode Selection Screen     │
│ SPACE │ Submit character / Go to next       │
│ C     │ Clear current drawing               │
│ Q     │ Quit Application                    │

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 UNDERSTANDING STROKE MATCHING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW IT WORKS:
1. You draw a stroke with your finger
2. System captures your finger movement
3. Stroke is normalized (position, scale)
4. Compared to template stroke using DTW algorithm
5. Angle is checked (reward correct directions)
6. Accuracy score calculated
7. Instant feedback given

WHAT COUNTS AS CORRECT:
✓ Stroke drawn in correct direction
✓ Roughly correct shape
✓ Approximately correct position
✓ Different speeds are OK
✓ Individual handwriting styles accepted

WHAT COUNTS AS WRONG:
✗ Stroke drawn backwards/reversed
✗ Completely different shape
✗ Wrong position on canvas
✗ Stray marks outside character area

TIPS FOR BETTER ACCURACY:
1. Draw deliberately (not too fast, not too slow)
2. Keep entire hand visible to camera
3. Ensure good lighting
4. Position yourself 12-18 inches from camera
5. Avoid shadows on your hand

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📱 USING WITH ZOOM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOR VIRTUAL CHINESE CLASSES:

1. Start your Zoom meeting (invite students)
2. Open Terminal: python launcher.py
3. Application window appears
4. In Zoom: Click "Share Screen"
5. Select "Chinese Character Tutor" window
6. All students see your teaching!

WHAT STUDENTS SEE:
- Animated stroke demonstrations
- Your real-time feedback
- Character validation
- Scoring system
- Learning progression

FOR INTERACTIVE LESSONS:
- Ask students to draw along in their own rooms
- They send screenshots in chat
- You provide feedback using the tutor
- Great for group accountability!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 EXAMPLE 30-MINUTE LESSON:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MINUTE  ACTIVITY                          MODE
─────   ─────────────────────────────────────────────
0-5     Introduction, camera setup         -
5-15    Learn: 一 二 十 (numbers)          Teaching
15-25   Recognize pinyin: shui3, huo3, etc Pinyin
25-35   Challenge: write from memory       English
35-40   Review & feedback                  Teaching
40+     Q&A, tips, next lesson prep        -

EXPECTED OUTCOMES:
✓ Students learn proper stroke order
✓ Build character recognition
✓ Practice writing technique
✓ Have fun with gamification!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎓 LEARNING PROGRESSION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEGINNER (Week 1):
- Master basic strokes: 一 二 十
- Learn stroke direction concepts
- Complete Teaching Mode for 5 characters
- Score target: 500+ points in Pinyin mode

INTERMEDIATE (Week 2-3):
- Learn complex characters: 人 口 水
- Practice in both Pinyin and English modes
- Improve speed and accuracy
- Score target: 1000+ points

ADVANCED (Week 4+):
- Challenge mode (English only)
- New character sets
- Aim for 100% accuracy
- Help teach friends!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ TROUBLESHOOTING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBLEM: Camera not working
SOLUTION: macOS → Settings → Privacy & Security → Camera → Enable Terminal

PROBLEM: Strokes not detecting
SOLUTION: 
- Improve lighting
- Move closer to camera (12 inches)
- Move slower
- Ensure full hand is visible

PROBLEM: "Incorrect" when I'm sure I drew it right
SOLUTION:
- Try drawing in the correct direction
- Make sure strokes are in correct order
- Practice a couple times (learns your style)
- Check template guide carefully

PROBLEM: Low FPS / Choppy performance
SOLUTION:
- Close other applications
- Reduce window size
- Update camera drivers
- Check CPU usage

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 GETTING STARTED CHECKLIST:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

□ Clone repository
□ Create virtual environment
□ Install dependencies (pip install -r requirements.txt)
□ Grant camera permissions to Terminal
□ Run launcher.py
□ Try Teaching Mode with one character
□ Try Pinyin Recognition Mode
□ Try English Translation Mode
□ Add personal characters to characters.json
□ Set up Zoom and share screen
□ Practice with friends!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎉 YOU'RE READY TO START!

Commands to remember:
$ source .venv/bin/activate     # Activate virtual environment
$ python launcher.py             # Start the tutor
$ python zoom_integration.py     # Show Zoom setup instructions

For detailed documentation: See README.md

Happy learning! 加油! 🚀
"""

if __name__ == "__main__":
    print(DEMO_GUIDE)
