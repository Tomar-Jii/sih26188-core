import cv2
import numpy as np

class TextureFlatnessAnalyzer:
    """Isolates hand-drawn digital brush strokes, scribbles, and blackouts using Euclidean Distance Transform."""

    @classmethod
    def detect_digital_strokes(cls, img_cv: np.ndarray, qr_bbox: list = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "mean_variance": 0.0}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        h_img, w_img = gray.shape[:2]
        total_area = h_img * w_img

        # 1. Isolate all dark marks (printed text + digital markup)
        # Using Otsu thresholding bounded to dark ink
        _, raw_dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dark_mask = np.logical_and(raw_dark > 0, gray < 95).astype(np.uint8) * 255

        # Mask out QR Code if present
        if qr_bbox:
            qx, qy, qw, qh = qr_bbox
            pad = 8
            cv2.rectangle(dark_mask, (max(0, qx - pad), max(0, qy - pad)),
                          (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)), 0, -1)

        # 2. Euclidean Distance Transform: Measures radius/thickness from edge to stroke core
        dist = cv2.distanceTransform(dark_mask, cv2.DIST_L2, 5)

        # Normal printed font stroke thickness is 1.5 - 2.2px (dist peak < 2.0).
        # Mobile gallery markup / digital brush thickness is >= 6px (dist peak >= 2.7).
        thick_stroke_seeds = (dist >= 2.7).astype(np.uint8) * 255

        # 3. Suppress Standard Government Logos (Ashoka emblem & UIDAI fingerprint)
        # Ashoka Pillar: top-left corner (x < 18%, y < 25%)
        cv2.rectangle(thick_stroke_seeds, (0, 0), (int(w_img * 0.18), int(h_img * 0.25)), 0, -1)
        # Aadhaar Fingerprint Logo: top-right corner (x > 78%, y < 25%)
        cv2.rectangle(thick_stroke_seeds, (int(w_img * 0.78), 0), (w_img, int(h_img * 0.25)), 0, -1)

        # 4. Reconstruct full stroke geometry from thick seeds
        kernel_grow = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        reconstructed = cv2.dilate(thick_stroke_seeds, kernel_grow, iterations=2)
        reconstructed = cv2.bitwise_and(reconstructed, dark_mask)

        # Bridge broken scribble segments
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed_strokes = cv2.morphologyEx(reconstructed, cv2.MORPH_CLOSE, kernel_bridge)

        contours, _ = cv2.findContours(closed_strokes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tamper_zones = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Area filter: captures real scribbles, strike-through lines, and hair doodles (> 50px)
            if 50 < area < (total_area * 0.15):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / max(h, 1)

                # Ignore thin dividing lines spanning card width
                if aspect > 7.0 and h < 7:
                    continue

                tamper_zones.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "score": 0.94,
                    "signal": "Digital Ink / Markup Stroke Signature"
                })

        return {
            "tamper_zones": tamper_zones,
            "count": len(tamper_zones),
            "mean_variance": round(float(np.mean(dist)), 2)
        }
