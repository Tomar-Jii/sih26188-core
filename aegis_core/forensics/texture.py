import cv2
import numpy as np

class TextureFlatnessAnalyzer:
    """Isolates hand-drawn digital scribbles and strike-through strokes while exempting machine typography."""

    @classmethod
    def detect_digital_strokes(cls, img_cv: np.ndarray, qr_bbox: list = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "mean_variance": 0.0}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        h_img, w_img = gray.shape[:2]
        total_area = h_img * w_img

        # Isolate dark marks
        _, raw_dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dark_mask = np.logical_and(raw_dark > 0, gray < 90).astype(np.uint8) * 255

        if qr_bbox:
            qx, qy, qw, qh = qr_bbox
            pad = 8
            cv2.rectangle(dark_mask, (max(0, qx - pad), max(0, qy - pad)),
                          (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)), 0, -1)

        # Distance Transform for stroke width
        dist = cv2.distanceTransform(dark_mask, cv2.DIST_L2, 5)

        # Suppress standard government emblems (Ashoka pillar top-left & UIDAI logo top-right)
        dist[0:int(h_img * 0.15), 0:int(w_img * 0.20)] = 0.0
        dist[0:int(h_img * 0.15), int(w_img * 0.75):w_img] = 0.0

        # Thick stroke seeds (hand scribbles / thick markers)
        thick_stroke_seeds = (dist >= 3.2).astype(np.uint8) * 255

        # Reconstruct continuous strokes
        kernel_grow = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        reconstructed = cv2.dilate(thick_stroke_seeds, kernel_grow, iterations=1)
        reconstructed = cv2.bitwise_and(reconstructed, dark_mask)

        contours, _ = cv2.findContours(reconstructed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tamper_zones = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / max(h, 1)

            # DISCRETE TYPOGRAPHY FILTER:
            # Individual numbers (0-9) and characters have bounded dimensions:
            # Width < 45px, Height < 55px, and aspect ratio between 0.25 and 1.3
            is_discrete_char = (w <= 45 and h <= 55 and 0.20 <= aspect <= 1.4)
            if is_discrete_char and area < 650:
                continue

            # Skip genuine horizontal dividing lines spanning the document
            if aspect > 6.5 and h < 8:
                continue

            # Real manual scribbles / strike-throughs cover substantial continuous area (> 120px)
            if 120 < area < (total_area * 0.15):
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
