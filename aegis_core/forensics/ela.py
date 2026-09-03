import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

class DifferentialELAAnalyzer:
    """Calibrated Error Level Analysis with text-ringing suppression."""

    @classmethod
    def analyze(cls, orig_pil: Image.Image, img_cv: np.ndarray, qr_bbox: list = None, quality: int = 90) -> dict:
        if orig_pil is None or img_cv is None:
            return {"suspicious_zones": [], "ela_enhanced": orig_pil, "ela_variance": 0.0}

        rgb_pil = orig_pil.convert("RGB")
        buffered = io.BytesIO()
        rgb_pil.save(buffered, format="JPEG", quality=quality)
        buffered.seek(0)
        resaved = Image.open(buffered)

        ela_im = ImageChops.difference(rgb_pil, resaved)
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1.0
        ela_enhanced = ImageEnhance.Brightness(ela_im).enhance(scale)

        diff_np = np.array(ela_im).astype(np.float32)
        channel_max = np.max(diff_np, axis=2).astype(np.uint8)

        h_img, w_img = channel_max.shape[:2]
        total_area = h_img * w_img

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        doc_mask = gray > 35
        valid_pixels = channel_max[doc_mask] if np.sum(doc_mask) > 500 else channel_max

        mean_val = float(np.mean(valid_pixels))
        std_val = float(np.std(valid_pixels))

        # Dynamic threshold tuned for re-saved compression artifacts
        dynamic_thresh = max(48, min(int(mean_val + (1.75 * std_val)), 78))
        blur_ela = cv2.GaussianBlur(channel_max, (3, 3), 0)
        _, thresh_ela = cv2.threshold(blur_ela, dynamic_thresh, 255, cv2.THRESH_BINARY)
        thresh_ela = cv2.bitwise_and(thresh_ela, thresh_ela, mask=doc_mask.astype(np.uint8) * 255)

        if qr_bbox:
            qx, qy, qw, qh = qr_bbox
            pad = 8
            cv2.rectangle(thresh_ela, (max(0, qx - pad), max(0, qy - pad)),
                          (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)), 0, -1)

        # Filter single character glyphs (< 65px)
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh_ela, cv2.MORPH_OPEN, kernel_clean)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        suspicious_zones = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 65 < area < (total_area * 0.12):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / max(h, 1)

                if aspect > 6.0 and h < 8:
                    continue

                suspicious_zones.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "score": 0.88,
                    "signal": "High Differential Compression (ELA)"
                })

        return {
            "suspicious_zones": suspicious_zones,
            "ela_enhanced": ela_enhanced,
            "ela_variance": round(std_val, 2)
        }
