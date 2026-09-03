import cv2
import numpy as np

class TextureFlatnessAnalyzer:
    """Detects digital markup, brush scribbles, and blackouts via Chromatic Zero-Entropy Assay."""

    @classmethod
    def detect_digital_strokes(cls, img_cv: np.ndarray, qr_bbox: list = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "mean_variance": 0.0}

        h_img, w_img = img_cv.shape[:2]
        total_area = h_img * w_img

        b, g, r = cv2.split(img_cv)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 1. Chromatic Deviation: Real camera capture of ink has paper reflection (|R-G| + |G-B| > 6)
        # Mobile gallery markup / digital pen has pure synthetic neutral values (|R-G| + |G-B| <= 5)
        chroma_diff = np.abs(r.astype(np.int16) - g.astype(np.int16)) + \
                      np.abs(g.astype(np.int16) - b.astype(np.int16)) + \
                      np.abs(b.astype(np.int16) - r.astype(np.int16))

        # 2. Local Texture Entropy / Flatness
        gray_f = gray.astype(np.float32)
        local_mean = cv2.blur(gray_f, (3, 3))
        local_sq = cv2.blur(gray_f ** 2, (3, 3))
        local_std = np.sqrt(np.maximum(local_sq - (local_mean ** 2), 0.0))

        # Digital markup mask: Dark pixel + perfectly neutral color + unnaturally flat local texture
        is_dark = gray < 65
        is_synthetic_neutral = chroma_diff <= 6
        is_flat_surface = local_std < 4.2

        markup_pixels = np.logical_and(is_dark, np.logical_and(is_synthetic_neutral, is_flat_surface))
        raw_mask = markup_pixels.astype(np.uint8) * 255

        # Mask QR Code matrix
        if qr_bbox:
            qx, qy, qw, qh = qr_bbox
            pad = 8
            cv2.rectangle(raw_mask, (max(0, qx - pad), max(0, qy - pad)),
                          (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)), 0, -1)

        # Suppress standard government emblems (Ashoka pillar & UIDAI logos)
        cv2.rectangle(raw_mask, (0, 0), (int(w_img * 0.18), int(h_img * 0.16)), 0, -1)
        cv2.rectangle(raw_mask, (int(w_img * 0.75), 0), (w_img, int(h_img * 0.16)), 0, -1)

        # Filter isolated 1px halftone noise and bridge scribble paths
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_clean)
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_bridge)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tamper_zones = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Catches small scribbles like '1014' (> 28px) up to large strike-through strokes
            if 28 < area < (total_area * 0.18):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / max(h, 1)

                # Ignore long thin dividing template rules
                if aspect > 7.0 and h < 6:
                    continue

                tamper_zones.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "score": 0.94,
                    "signal": "Synthetic Digital Ink / Markup Stroke"
                })

        return {
            "tamper_zones": tamper_zones,
            "count": len(tamper_zones),
            "mean_variance": round(float(np.mean(local_std)), 2)
        }
