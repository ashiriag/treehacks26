"""
Stroke Recognition & Validation Engine
- Matches user-drawn strokes to template strokes
- Validates stroke order and correctness
- Detects character recognition
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional

# ===============================
# Geometry & Normalization Utils
# ===============================

def dist(a, b):
    """Euclidean distance between two points."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def polyline_length(pts):
    """Total length of a polyline."""
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

    dists = [0.0]
    for i in range(1, len(pts)):
        dists.append(dists[-1] + dist(pts[i - 1], pts[i]))

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
    """Normalize stroke: translate to centroid, scale to unit box."""
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
    """DTW distance between two sequences of 2D points."""
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


# ===============================
# Stroke Matching
# ===============================

def calculate_stroke_angle(pts):
    """Calculate primary direction angle of a stroke (0-360 degrees)."""
    if len(pts) < 2:
        return 0.0
    
    pts_arr = np.array(pts, dtype=np.float32)
    start = pts_arr[0]
    end = pts_arr[-1]
    
    delta = end - start
    angle = np.arctan2(delta[1], delta[0]) * 180.0 / np.pi
    # Normalize to 0-360
    if angle < 0:
        angle += 360
    return angle


def stroke_match_score(user_pts, template_pts, resample_n=64):
    """
    Score how well user stroke matches template.
    Returns score (0 = perfect, higher = worse).
    """
    if len(user_pts) < 3:
        return 1e9
    
    # Resample both strokes
    user_resampled = resample_polyline(user_pts, resample_n)
    template_resampled = resample_polyline(template_pts, resample_n)
    
    # Normalize to unit space
    user_norm = normalize_points(user_resampled)
    template_norm = normalize_points(template_resampled)
    
    # DTW distance
    dtw = dtw_distance(user_norm, template_norm)
    
    # Angle consistency (prefer same direction as template)
    user_angle = calculate_stroke_angle(user_pts)
    template_angle = calculate_stroke_angle(template_pts)
    angle_diff = abs(user_angle - template_angle)
    # Normalize angle difference to 0-180
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    angle_penalty = angle_diff / 180.0 * 0.3  # 30% weight
    
    final_score = dtw + angle_penalty
    return final_score


def match_stroke_to_template(
    user_strokes: List[List[Tuple[float, float]]],
    template_strokes: List[List[Tuple[float, float]]],
    threshold: float = 0.25
) -> Dict:
    """
    Match user-drawn strokes to template strokes.
    Returns: {
        'matched': bool,
        'correct_strokes': int,
        'wrong_strokes': [],
        'missing_strokes': [],
        'accuracy': float (0-1)
    }
    """
    n_template = len(template_strokes)
    matched_count = 0
    wrong_strokes = []
    
    for i, user_stroke in enumerate(user_strokes):
        if i >= n_template:
            # Extra strokes
            wrong_strokes.append({"stroke_idx": i, "reason": "extra_stroke"})
            continue
        
        template_stroke = template_strokes[i]
        score = stroke_match_score(user_stroke, template_stroke)
        
        if score <= threshold:
            matched_count += 1
        else:
            wrong_strokes.append({
                "stroke_idx": i,
                "reason": "incorrect_shape",
                "score": score
            })
    
    # Missing strokes
    missing_count = max(0, n_template - len(user_strokes))
    missing_strokes = list(range(len(user_strokes), n_template))
    
    accuracy = matched_count / max(1, n_template)
    matched = (accuracy >= 0.8)  # 80% threshold for "correct"
    
    return {
        "matched": matched,
        "correct_strokes": matched_count,
        "total_strokes": n_template,
        "wrong_strokes": wrong_strokes,
        "missing_strokes": missing_strokes,
        "accuracy": accuracy
    }


# ===============================
# Character Recognition
# ===============================

class CharacterDatabase:
    """Load and manage character templates."""
    
    def __init__(self, json_path: str = "characters.json"):
        self.json_path = json_path
        self.characters = []
        self.load_characters()
    
    def load_characters(self):
        """Load characters from JSON."""
        if not os.path.exists(self.json_path):
            print(f"Warning: {self.json_path} not found")
            return
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.characters = data.get('characters', [])
    
    def get_character_by_index(self, idx: int) -> Optional[Dict]:
        """Get character by index."""
        if 0 <= idx < len(self.characters):
            return self.characters[idx]
        return None
    
    def get_random_character(self) -> Dict:
        """Get random character."""
        import random
        return random.choice(self.characters)
    
    def recognize_character(
        self,
        user_strokes: List[List[Tuple[float, float]]],
        threshold: float = 0.25
    ) -> List[Dict]:
        """
        Recognize which character user drew.
        Returns list of possible matches sorted by confidence.
        """
        results = []
        
        for char_data in self.characters:
            template_strokes = char_data['strokes']
            match_result = match_stroke_to_template(
                user_strokes,
                template_strokes,
                threshold
            )
            
            if match_result['accuracy'] > 0.5:  # Only return decent matches
                results.append({
                    "character": char_data['char'],
                    "pinyin": char_data['pinyin'],
                    "english": char_data['english'],
                    "accuracy": match_result['accuracy'],
                    "details": match_result
                })
        
        # Sort by accuracy (descending)
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        return results


# ===============================
# Stroke Order Validation
# ===============================

def validate_stroke_order(
    user_strokes: List[List[Tuple[float, float]]],
    template_strokes: List[List[Tuple[float, float]]],
    threshold: float = 0.25
) -> Dict:
    """
    Validate that strokes were drawn in correct order.
    Returns detailed feedback.
    """
    correct_order = True
    feedback = []
    
    for i, user_stroke in enumerate(user_strokes):
        if i >= len(template_strokes):
            feedback.append({
                "stroke": i,
                "status": "extra",
                "message": f"You drew too many strokes! (expected {len(template_strokes)})"
            })
            correct_order = False
            break
        
        template_stroke = template_strokes[i]
        score = stroke_match_score(user_stroke, template_stroke)
        
        if score <= threshold:
            feedback.append({
                "stroke": i,
                "status": "correct",
                "message": f"✓ Stroke {i+1} correct!"
            })
        else:
            feedback.append({
                "stroke": i,
                "status": "incorrect",
                "message": f"✗ Stroke {i+1} incorrect. Shape doesn't match.",
                "score": score
            })
            correct_order = False
    
    if len(user_strokes) < len(template_strokes):
        for i in range(len(user_strokes), len(template_strokes)):
            feedback.append({
                "stroke": i,
                "status": "missing",
                "message": f"Missing stroke {i+1}"
            })
        correct_order = False
    
    return {
        "correct_order": correct_order,
        "feedback": feedback,
        "accuracy": sum(1 for f in feedback if f['status'] == 'correct') / len(template_strokes)
    }


if __name__ == "__main__":
    # Test
    db = CharacterDatabase()
    print(f"Loaded {len(db.characters)} characters")
    
    # Test first character
    if db.characters:
        char = db.characters[0]
        print(f"\nCharacter: {char['char']} ({char['pinyin']}) - {char['english']}")
        print(f"Strokes: {len(char['strokes'])}")
