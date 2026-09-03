import cv2
import numpy as np
from aegis_core.config import CONFIG

class DocumentQualityGate:
    @staticmethod
    def audit(img_cv: np.ndarray) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"passed": False, "reason": "Zero-dimension image buffer", "metrics": {}}

        h, w = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        pure_white_ratio = float(np.sum(hist[250:]) / max(h * w, 1))

        passed = bool(lap_var >= CONFIG.MIN_SHARPNESS_LAPLACIAN and 
                      pure_white_ratio <= CONFIG.MAX_SPECULAR_GLARE_RATIO and 
                      w >= CONFIG.MIN_DOCUMENT_WIDTH and 
                      h >= CONFIG.MIN_DOCUMENT_HEIGHT)

        reason = None
        if not passed:
            reasons = []
            if lap_var < CONFIG.MIN_SHARPNESS_LAPLACIAN: reasons.append("Insufficient Edge Sharpness / Motion Blur")
            if pure_white_ratio > CONFIG.MAX_SPECULAR_GLARE_RATIO: reasons.append("Specular Surface Glare")
            reason = "; ".join(reasons)

        return {
            "passed": passed,
            "abstain_reason": reason,
            "metrics": {
                "sharpness_laplacian": round(lap_var, 1),
                "blur_status": "Good" if lap_var > 45 else ("Acceptable" if lap_var >= 14 else "Blurry"),
                "glare_status": "High" if pure_white_ratio > 0.15 else "Normal",
                "resolution": f"{w}x{h}"
            }
        }
