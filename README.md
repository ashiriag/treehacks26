# 中文 Chinese Character Tutor 🎨

**Learn to write Chinese characters with real-time camera-based stroke detection and AI feedback!**

An interactive application that teaches Chinese character writing through three distinct modes:
1. **Teaching Mode** - Guided learning with animated stroke directions
2. **Pinyin Recognition** - Recall-based practice (see pinyin, write character)
3. **English Translation** - Gamified learning (see English, write character)

Perfect for virtual classrooms via Zoom, in-person tutoring, or self-study.

---

## ✨ Features

### 🎓 Teaching Mode
- **Animated Stroke Guides**: Real-time arrows showing stroke direction
- **Instant Validation**: Know immediately if you drew the stroke correctly
- **Step-by-Step Learning**: Complete one stroke at a time with visual feedback
- **Direction Indicators**: See which way to draw (left→right, top→bottom, diagonal)

### 📝 Pinyin Recognition Mode
- **Audio Recall**: See the pinyin and recall the character
- **Scoreboard**: Earn 150 points per correct character
- **Accuracy Tracking**: Real-time feedback on stroke accuracy

### 📖 English Translation Mode (Gamified)
- **Translation Practice**: See English word, write the corresponding character
- **Gamification**: 200 points per character, completion counter
- **Progressive Difficulty**: Build character vocabulary systematically
- **Performance Metrics**: Track your progress

### 🎥 Camera Integration
- **Hand Detection**: Real-time finger tracking via MediaPipe
- **Stroke Capture**: Converts finger movements into digital strokes
- **Automatic Validation**: Proper character formation detection

### 🎨 Zoom-Ready UI
- **Screen Share Compatible**: Optimized 1280x720 resolution
- **High Contrast Design**: Always readable in presentations
- **Clean Layout**: Perfect for virtual classrooms

---

## 🚀 Quick Start

### Installation

```bash
cd /Users/lukeqiao/Documents/Projects/treehacks_2026
source .venv/bin/activate
python main_app.py
```

### First Run
1. Press **1**, **2**, or **3** to choose a learning mode
2. Start writing characters with your finger in front of the camera
3. Press **SPACE** to submit your work or move to the next character

---

## 📚 Learning Modes

### 🎓 Teaching Mode (Press 1)
Learn proper stroke technique with guided instructions.

```
Teaching Mode - 一 (one)
Pinyin: yi1

    ➜ ➜ ➜ ➜  (animated arrow)
    ─────────────────  (stroke 1)

Stroke 1 / 1
```

**How it works:**
- Animated arrow shows stroke direction
- Semi-transparent guide shows exact path
- Real-time validation after each stroke
- Move to next stroke automatically on success

### 📝 Pinyin Recognition Mode (Press 2)
Practice recalling characters from their sound.

```
Pinyin Mode
Write the character that sounds like: shui3 (water)

    [Your drawing here]

Score: 0150
```

**Scoring:**
- Correct character: +150 points
- Incorrect: 0 points (try again)
- Build your vocabulary systematically

### 📖 English Translation Mode (Press 3)
The ultimate memory challenge with gamification.

```
Translation Mode
Write the character for: WATER

    [Your drawing here]

Completed: 5        Score: 00850
✓ Correct! 水
```

**Gamification Features:**
- Higher points (200 vs 150)
- No hints - full recall challenge
- Completion counter
- Progress visible to everyone in Zoom

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| **1** | Teaching Mode |
| **2** | Pinyin Recognition Mode |
| **3** | English Translation Mode |
| **SPACE** | Submit drawing / Next character |
| **C** | Clear current drawing |
| **M** | Return to mode selection |
| **Q** | Quit application |

---

## 🎥 Zoom Integration

### Setup for Virtual Teaching

1. **Start Zoom meeting**
2. **Launch tutor**: `python main_app.py`
3. **Share screen**: Click "Share Screen" → Select tutor window
4. **Everyone sees**: Live character learning with feedback!

### Perfect For:
- Virtual Chinese classes
- Group tutoring sessions
- Hybrid learning (in-person + Zoom)
- Student demonstrations
- Interactive practice sessions

### Best Practices:
- Use **Teaching Mode** for demonstrations
- Use **Pinyin/English Modes** for interactive practice
- Ask students to draw in their own cameras while you evaluate
- Keep window at native resolution for clarity

---

## 📊 Built-in Characters

The system includes 10 fundamental characters to get started:

| Character | Pinyin | English | Strokes |
|-----------|--------|---------|---------|
| 一 | yi1 | one | 1 |
| 二 | er4 | two | 2 |
| 十 | shi2 | ten | 2 |
| 人 | ren2 | person | 2 |
| 口 | kou3 | mouth | 4 |
| 水 | shui3 | water | 3 |
| 火 | huo3 | fire | 4 |
| 木 | mu4 | tree | 4 |
| 金 | jin1 | gold | 5 |
| 土 | tu3 | earth | 4 |

**Want to add more?** Edit `characters.json` with new characters and stroke data.

---

## 🔧 Technical Architecture

### Core Components

**`main_app.py`** - Main application
- Mode management, state, scoring, event handling

**`stroke_engine.py`** - Recognition engine
- DTW stroke matching algorithm
- Character recognition & validation
- Stroke order verification

**`ui_renderer.py`** - Visual rendering
- Template stroke drawing
- Animated arrow generation
- UI panels & feedback

**`zoom_integration.py`** - Zoom optimization
- Screen share compatibility
- Setup instructions

### Stroke Matching Algorithm

We use **Dynamic Time Warping (DTW)** with angle scoring:

```
Score = DTW_Distance + (Angle_Penalty × 0.3)
Match = Score ≤ Threshold
```

**Why DTW?**
- ✓ Handles speed variations (fast/slow writing)
- ✓ Accounts for individual handwriting styles
- ✓ Sensitive to stroke direction mistakes
- ✓ Forgiving for minor deviations

---

## 🛠️ Installation & Troubleshooting

### Prerequisites
- Python 3.8+
- Webcam with 30+ FPS
- Camera permissions enabled

### Troubleshooting

**"Camera failed to initialize"**
- macOS: Settings → Privacy & Security → Camera → Grant access to Terminal/IDE
- Windows: Check camera in Device Manager
- Restart the application

**"Strokes not detecting"**
- Improve lighting in your room
- Move finger closer to camera (but keep full hand visible)
- Ensure minimum finger movement (MOVE_THRESHOLD = 5 pixels)

**"Strokes not matching"**
- Stroke must be drawn in the correct direction
- Try different handwriting styles - it learns from you
- Practice a few times - accuracy improves with familiarity

**"Low FPS / Lag"**
- Close unnecessary applications
- Update camera driver
- Lower resolution if needed (edit WINDOW_WIDTH/HEIGHT in main_app.py)

---

## 🎯 Game Scoring System

### Teaching Mode
- Goal: Complete all strokes correctly
- Instant stroke-by-stroke feedback
- Binary: Correct or Try Again
- Time-unlimited, accuracy-focused

### Pinyin Recognition Mode
- 150 points per correct character
- Encourages accuracy through scoring
- Mid-difficulty recall
- Great for vocabulary building

### English Translation Mode
- 200 points per correct character (hardest level)
- No hints - full memory challenge
- Completion counter motivates progress
- Best for advanced learners

### Future: Multipliers & Achievements
- Combo bonus: 3+ correct in a row
- Daily streak tracking
- Achievement badges
- Level progression

---

## 🚀 Roadmap & Future Features

### Next Phase:
- [ ] HSK level categories (1-6)
- [ ] Pronunciation audio for pinyin
- [ ] Radical-based learning system
- [ ] Multi-character words
- [ ] Stroke error analysis & weak point detection
- [ ] Leaderboards & multiplayer

### Advanced:
- [ ] Mobile app (iOS/Android)
- [ ] Native Zoom app (Zoom Marketplace)
- [ ] Browser-based version
- [ ] Offline mode
- [ ] Custom character sets

---

## 📜 File Structure

```
treehacks_2026/
├── main_app.py              # Main application (run this!)
├── stroke_engine.py         # Recognition & validation
├── ui_renderer.py           # Visual rendering
├── zoom_integration.py      # Zoom setup & helpers
├── characters.json          # Character database
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 💡 Learning Tips

### For Best Results:
1. **Start with Teaching Mode** - Learn proper stroke order
2. **Practice consistently** - Handwriting style matters
3. **Vary your speed** - Both fast and slow writing is acceptable
4. **Master basics first** - Begin with simple characters (一 二 十)
5. **Add new characters weekly** - Progressive learning
6. **Share in Zoom** - Group accountability helps!

### Cultural Context:
- Stroke order is fundamental to Chinese writing education
- Proper order often reflects the logic of the character
- Simplified characters (taught here) vs Traditional variants
- Pinyin makes pronunciation accessible to learners

---

## 🙏 Credits

- **MediaPipe**: Real-time hand detection
- **OpenCV**: Image processing & rendering
- **DTW Algorithm**: Dynamic Time Warping for stroke matching
- **HSK Standard**: Chinese learning framework

---

## 📞 Support

**Issues?** Common problems and solutions are in the Troubleshooting section above.

**Want to contribute?** Ideas for new features or characters are welcome!

---

**Made with 💚 for language learners everywhere**

*TreeHacks 2026 - Learn Chinese, Learn Fast! 🚀*
