import cv2
import numpy as np

class TextureFlatnessAnalyzer:
    """Isolates real manual defacements while preserving legitimate document components."""

    @classmethod
    def detect_digital_strokes(cls, img_cv: np.ndarray, qr_bbox: list = None, **kwargs) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "mean_variance": 0.0}

        h_img, w_img = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        tamper_zones = []

        all_qr_boxes = kwargs.get("qr_bboxes", [])
        if qr_bbox and qr_bbox not in all_qr_boxes:
            all_qr_boxes.append(qr_bbox)

        # ---------------------------------------------------------------------
        # 1. MANUAL DIGITAL BRUSH DOODLES (Zero Entropy Synthetic Ink)
        # ---------------------------------------------------------------------
        b, g, r = cv2.split(img_cv)
        chroma_diff = np.abs(r.astype(np.int16) - g.astype(np.int16)) + \
                      np.abs(g.astype(np.int16) - b.astype(np.int16)) + \
                      np.abs(b.astype(np.int16) - r.astype(np.int16))

        gray_f = gray.astype(np.float32)
        local_mean = cv2.blur(gray_f, (3, 3))
        local_sq = cv2.blur(gray_f ** 2, (3, 3))
        local_std = np.sqrt(np.maximum(local_sq - (local_mean ** 2), 0.0))

        # Real print ink has diffuse camera grain; synthetic digital pen is flat (std < 2.0)
        is_synthetic_markup = (gray < 40) & (chroma_diff <= 3) & (local_std < 2.0)
        markup_mask = is_synthetic_markup.astype(np.uint8) * 255

        # Mask ALL QR matrices
        for qx, qy, qw, qh in all_qr_boxes:
            pad = 12
            cv2.rectangle(markup_mask, (max(0, qx - pad), max(0, qy - pad)),
                          (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)), 0, -1)

        # Mask Government Emblems
        cv2.rectangle(markup_mask, (0, 0), (int(w_img * 0.20), int(h_img * 0.20)), 0, -1)
        cv2.rectangle(markup_mask, (int(w_img * 0.75), 0), (w_img, int(h_img * 0.20)), 0, -1)

        k_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_markup = cv2.morphologyEx(markup_mask, cv2.MORPH_OPEN, k_clean)
        k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        connected_markup = cv2.morphologyEx(cleaned_markup, cv2.MORPH_CLOSE, k_bridge)

        cnts_markup, _ = cv2.findContours(connected_markup, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_markup:
            area = cv2.contourArea(cnt)
            if 45 < area < (h_img * w_img * 0.12):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / max(h, 1)
                if aspect > 7.0 and h < 6:
                    continue
                tamper_zones.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "score": 0.96,
                    "signal": "Synthetic Digital Ink / Markup Scribble"
                })

        # ---------------------------------------------------------------------
        # 2. CROSS-TEXT STRIKE-THROUGH STROKES
        # ---------------------------------------------------------------------
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        for qx, qy, qw, qh in all_qr_boxes:
            pad = 12
            cv2.rectangle(binary, (max(0, qx - pad), max(0, qy - pad)),
                          (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)), 0, -1)

        k_strike = cv2.getStructuringElement(cv2.MORPH_RECT, (38, 3))
        thick_strikes = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_strike)
        cnts_strike, _ = cv2.findContours(thick_strikes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts_strike:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > int(w_img * 0.50):
                continue
            if 36 <= w <= int(w_img * 0.40) and 3 <= h <= 14:
                tamper_zones.append({
                    "bbox": [int(x), int(y - 2), int(w), int(h + 4)],
                    "score": 0.95,
                    "signal": "Strike-Through Defacement Stroke"
                })

        return {
            "tamper_zones": tamper_zones,
            "count": len(tamper_zones),
            "mean_variance": 0.0
        }
