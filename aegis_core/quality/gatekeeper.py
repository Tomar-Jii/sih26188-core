import cv2
import numpy as np

class DocumentQualityGate:
    """Evaluates sharpness, blur, and lighting conditions with adaptive document substrate awareness."""

    @staticmethod
    def audit(img_cv: np.ndarray) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {
                "passed": False,
                "abstain_reason": "Zero-dimension image buffer",
                "metrics": {"sharpness_laplacian": 0.0, "blur_status": "Unusable", "glare_status": "Unknown", "resolution": "0x0"}
            }

        h, w = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_status = "Good" if lap_var > 45.0 else ("Acceptable" if lap_var > 12.0 else "Poor (Blurry)")

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_pixels = max(h * w, 1)
        pure_white_ratio = float(np.sum(hist[250:]) / total_pixels)

        # Distinguish white paper documents from localized camera flash glare
        # Authentic white paper has high sharpness (lap_var > 80) and uniform luminance
        is_white_paper_doc = (pure_white_ratio > 0.25) and (lap_var > 60.0)

        if is_white_paper_doc:
            glare_status = "Natural Paper White"
        else:
            glare_status = "High (Glare Detected)" if pure_white_ratio > 0.35 else "Normal"

        passed = bool(lap_var > 12.0 and (glare_status != "High (Glare Detected)") and w >= 200 and h >= 200)
        abstain_reason = None
        if not passed:
            reasons = []
            if lap_var <= 12.0: reasons.append("Motion Blur or Out-of-Focus")
            if glare_status == "High (Glare Detected)": reasons.append("Severe Camera Flash Specular Glare")
            abstain_reason = "; ".join(reasons)

        return {
            "passed": passed,
            "abstain_reason": abstain_reason,
            "metrics": {
                "sharpness_laplacian": round(lap_var, 1),
                "blur_status": blur_status,
                "glare_status": glare_status,
                "resolution": f"{w}x{h}"
            }
        }
