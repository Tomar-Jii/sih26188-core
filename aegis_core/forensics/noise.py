import cv2
import numpy as np

class LocalNoiseAnalyzer:
    """High-frequency sensor noise consistency evaluator."""

    @staticmethod
    def extract_residual_map(img_cv: np.ndarray) -> tuple[np.ndarray, float]:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        denoised = cv2.medianBlur(gray, 3)
        residual = cv2.absdiff(gray, denoised)
        global_variance = float(np.var(residual))
        norm_map = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
        return norm_map, round(global_variance, 2)

    @staticmethod
    def evaluate_region_noise(img_cv: np.ndarray, bbox: list) -> float:
        """Returns relative noise ratio between ROI and outer document background."""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        x, y, w, h = bbox
        roi = gray[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 1.0

        denoised_roi = cv2.medianBlur(roi, 3)
        roi_noise = float(np.var(cv2.absdiff(roi, denoised_roi)))

        denoised_global = cv2.medianBlur(gray, 3)
        global_noise = float(np.var(cv2.absdiff(gray, denoised_global))) + 1e-5

        return round(roi_noise / global_noise, 3)
