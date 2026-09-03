import cv2
import numpy as np

class EdgeDiscontinuityAnalyzer:
    """Computes boundary gradient jumps to isolate cut-and-paste splicing edges."""

    @staticmethod
    def evaluate_boundary_gradient(img_cv: np.ndarray, bbox: list) -> float:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        x, y, w, h = bbox
        
        # Extract edge perimeter band
        pad = 2
        y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
        band = gray[y1:y2, x1:x2]

        if band.size == 0:
            return 0.0

        sobel_x = cv2.Sobel(band, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(band, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        mean_jump = float(np.mean(magnitude))
        return round(min(mean_jump / 128.0, 1.0), 3)
