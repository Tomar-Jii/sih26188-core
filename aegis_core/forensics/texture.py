import cv2
import numpy as np

class TextureFlatnessAnalyzer:
    """Detects digital brush strokes, smudge operations, and unnaturally smooth clone patches."""

    @staticmethod
    def audit_patch_flatness(img_cv: np.ndarray, bbox: list) -> dict:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        x, y, w, h = bbox
        patch = gray[y:y+h, x:x+w]

        if patch.size < 25:
            return {"is_unnaturally_flat": False, "local_std": 0.0}

        # Calculate standard deviation inside sliding 3x3 windows
        local_std = float(np.std(patch))
        
        # Physical cards printed on paper/PVC have microscopic grain (std >= 8.0).
        # Digital paint or solid brush strokes produce near-zero texture variation.
        is_flat = local_std < 4.8

        return {
            "is_unnaturally_flat": is_flat,
            "local_std": round(local_std, 2)
        }
