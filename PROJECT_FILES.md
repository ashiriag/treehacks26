# Project Files Overview

## Core Application Files

### main_app.py (437 lines)
The main application controller that orchestrates everything.
- `TutorApp` class: Main state machine
- Three modes: teaching, pinyin, english
- Hand detection integration via MediaPipe
- Game logic: scoring, character selection, stroke validation
- Rendering pipeline calling UIRenderer
- Keyboard event handling

**Run with:** `python main_app.py`

### stroke_engine.py (280 lines)
Stroke recognition and validation engine.
- `CharacterDatabase`: Loads and manages character templates
- `normalize_points()`: Convert strokes to unit space
- `dtw_distance()`: Dynamic Time Warping algorithm
- `stroke_match_score()`: Rate how well user stroke matches template
- `match_stroke_to_template()`: Full stroke validation
- `validate_stroke_order()`: Check stroke sequence validity
- `recognize_character()`: Identify which character was drawn

**Core Algorithm# Project Files Overview

## Core Application Files

### main_app.py (437 lines)
The ma r
## Core Application Fi
- 
### main_app.py (437 li reThe main application contrplate_stroke()`: Display character guides
- `draw_animated_arrow- Three modes: teaching, pinyin, engl- - Hand detection integration via MediaPwi- Game logic: scoring, character selectios,- Rendering pipeline calling UIRrokes()`: Multiple stroke rend- Keyboard event handling

**Run with: a
**Run with:** `python mtio
### stroke_engine.py (280 lines)## Stroke recognition and validatico- `CharacterDatabase`: Loads and managesan- `normalize_points()`: Convert strokes to unit space
- `dtognition thresholds
- Animation settings
- Scoring value- `stroke_match_score()`: Rate how well user stro o- `match_stroke_to_template()`: Full stroke validation
- `validate_ I- `validate_stroke_order()`: Check stroke sequence vaUs- `recognize_character()`: Identify which character was drs 
**Core Algorithm# Project Files Overview

## Core Applicatis c
## Core Application Files

### main_aptcu
### main_app.py (437 li HaThe ma r
## Core Applicati*R## Cith:*- 
### main_app.py (4` #re- `draw_animated_arrow- Three modes: teaching, pinyin, engl- - Hand detection integration viui
**Run with: a
**Run with:** `python mtio
### stroke_engine.py (280 lines)## Stroke recognition and validatico- `CharacterDatabase`: Loads and managesan- `normalize_points()`: Convert strokes to unit space
- `dtognition thresholds
ide**Run with:*E.### stroke_engine.py (280si- `dtognition thresholds
- Animation settings
- Scoring value- `stroke_match_score()`: Rate how well user stro o- `match_stroke_to_template()`: Full stroke valida- - Animation settings
-ons- Scoring value- `s**- `validate_ I- `validate_stroke_order()`: Check stroke sequencenes)
Interactive step-by-step demo walkthrough.
- Mode se**Core Algorithm# Project Files Overview

## Core Applicatis c
## Core Application Files

### main_aptcu
### main_app.py (437 lill
## Core Applicatis c
## Core Applicati St## Core Applicationit
### main_aptcu
### mainteg### main_app.al## Core Applicati*R## Cith:*- 
##- ### main_app.py (4` #re- `draTr**Run with: a
**Run with:** `python mtio
### stroke_engine.py (280 lines)## Stroke recognition and validatico- `Chargu**Run with:*al### stroke_engine.py (280 D- `dtognition thresholds
ide**Run with:*E.### stroke_engine.py (280si- `dtognition thresholds
- Animation settings
- Scoring value- `stroke_match_score()`: Rate hblide**Run with:*E.### st
-- Animation ucture overview
- API reference for developers

**Run wit- Scoring value- `sGU-ons- Scoring value- `s**- `validate_ I- `validate_stroke_order()`: Check stroke sequencenes)
Interactive step-by-step demo walkthrough.
erInteractive step-by-step demo walkthrough.
- Mode se**Core Algorithm# Project Files Overviewma- Mode se**Core Algorithm# Project Files wi
## Core Applicatis c
## Core Application Files
# D## Core Applicationac
### main_aptcu
### mainara### main_app. d## Core Applicatis c
## ta## Core Applicati S c### main_aptcu
### mainteg### main_app.a?## mainteg## /##- ### main_app.py (4` #re- `draTr**Run with: a
** with **Run with:** `python mtio
### stroke_engine.pyla### stroke_engine.py (280  ide**Run with:*E.### stroke_engine.py (280si- `dtognition thresholds
- Animation settings
- Scorctions": [  // Direction metadata
    {"direction": "left_to_right", "angle": 0}
  ]
}
```

**Contains:**
- 一 (yi1- Scoring value- `s --- Animation ucture overview
- API reference for developers

**Run wit-? - API reference for develop) 
**Run wit- Scoring value- `s?? Interactive step-by-step demo walkthrough.
erInteractive step-by-step demo walkthrough.
- Mode se**Core Algorithm# Project perInteractive step-by-step demo walkthrouen- Mode se**Core Algorithm# Project Files Ovon## Core Applicatis c
## Core Application Files
# D## Core Applicationac
### main_aptcu
### mare## Core Application-
# D## Core Applicationac| ### mai| Value |
|--------|-------|
| To## ta## Core Applicati S c### main_aptcu
###  1### mainteg### main_app.a?## mainteg#s ** with **Run with:** `python mtio
### stroke_engine.pyla### stroke_engine.py (280  ide**ri### stroke_engine.pyla### stroke_le- Animation settings
- Scorctions": [  // Direction metadata
    {"direction": "left_to_right", "angle": 0}
  ]
}
``00- Scorctions": [  /le    {"direction": "left_to_right", "anau  ]
}
```

**Contains:**
- 一 (yi1- Scoring 
 }
 ├
*?? 一 (yi1- in- API reference for developers

**Run wit-? - API referencnd
**Run wit-? - API reference nMa**Run wit- Scoring value- `s?? InteractiaterInteractive step-by-step demo walkthrough.
- Mode se**Core Algorithm# ??- Mode se**Core Algorithm# Project perInteren## Core Application Files
# D## Core Applicationac
### main_aptcu
### mare## Core Application-
# D## Core Applicationac| ### mai| Value |
|----io# D## Core ApplicationacEN### main_aptcu
### mare o### mar)

Confi# D## Core Applicationac| #fi|--------|-------|
| To## ta## Core Appli? ??→ characters.json (Character definitions)

Dependencies:
 ### stroke_engine.pyla### stroke_engine.py (280  ide**ri### stroke_engine.pyla##*W- Scorctions": [  // Direction metadata
    {"direction": "left_to_right", "angle": 0}
  ]
}
``00- Scorctions"-     {"direction": "left_to_right", "anUI  ]
}
``00- Scorctions": [  /le    {"directiom_}
tegr}
```

**Contains:**
- 一 (yi1- Scoring 
 }
 ├
*?? 一 (yi1- iarac
*rs?- 一 (yi1- `c }
 ├
*?? 一 (*U de*??n
**Run wit-? - API referencnd
**Run engine.py`
- ***Run witolors/fonts?** → E- Mode se**Core Algorithm# ??- Mode se**Core Algorithm# Project perInteren## Core Application Files
# D## Core AppliPy# D## Core Applicationac
### main_aptcu
### mare## Core Application-
# D## Core Applicationac| ###mu### main_aptcu
### mare--### mare## Cono# D## Core Applicationac| #al|----io# D## Core ApplicationacEN### mainca### mare o### mar)

Confi# D## Core Applicatiopu
Confi# D## Core : D| To## ta## Core Appli? ??→ characters.json (Ch d
Dependencies:
 ### stroke_engine.pyla### stroke_engine.py (280  ide**ent ### stroke_ c    {"direction": "left_to_right", "angle": 0}
  ]
}
``00- Scorctions"-     {"direction": "left_to_right", "anUI  ]
}
``0if  ]
}
``00- Scorctions"-     {"direction": "lme}
atio}
``00- Scorctions": [  /le    {"directiom_}
tegr}
```

**Con-proof architecture

Ready to use! 🚀
