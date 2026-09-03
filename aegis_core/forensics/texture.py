import cv2
import numpy as np

class TextureFlatnessAnalyzer:
    """Detects digital pen strokes, markup scribbles, and clone patches via micro-texture variance."""

    @classmethod
    def detect_digital_strokes(cls, img_cv: np.ndarray, qr_bbox: list = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "mean_variance": 0.0}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        h_img, w_img = gray.shape[:2]
        total_area = h_img * w_img

        # 1. Compute Local Sliding Standard Deviation (5x5 Window)
        gray_f = gray.astype(np.float32)
        local_mean = cv2.blur(gray_f, (5, 5))
        local_sq = cv2.blur(gray_f ** 2, (5, 5))
        local_std = np.sqrt(np.maximum(local_sq - (local_mean ** 2), 0.0))

        # 2. Digital Stroke Mask
        # Physical print ink has sensor grain (std > 5.5); digital brushes have std < 3.6
        is_dark_stroke = gray < 65
        is_zero_grain = local_std < 3.6
        raw_stroke_mask = np.logical_and(is_dark_stroke, is_zero_grain).astype(np.uint8) * 255

        # 3. Mask out QR Code matrix if detected
        if qr_bbox:
            qx, qy, qw, qh = qr_bbox
            pad = 6
            cv2.rectangle(
                raw_stroke_mask,
                (max(0, qx - pad), max(0, qy - pad)),
                (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)),
                0, -1
            )

        # 4. Filter micro-specks
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(raw_stroke_mask, cv2.MORPH_OPEN, kernel_clean)
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        connected_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_connect)

        # 5. Extract Contours
        contours, _ = cv2.findContours(connected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tamper_zones = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Area range: captures slashes, dots (>14px) and large strokes
            if 14 < area < (total_area * 0.15):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / max(h, 1)

                # Skip genuine thin dividing lines spanning card width
                if aspect > 6.0 and h < 7:
                    continue

                tamper_zones.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "score": 0.92,
                    "signal": "Digital Ink / Flat Texture Signature"
                })

        return {
            "tamper_zones": tamper_zones,
            "count": len(tamper_zones),
            "mean_variance": round(float(np.mean(local_std)), 2)
        }
