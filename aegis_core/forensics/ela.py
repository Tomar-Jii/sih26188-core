import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from aegis_core.config import CONFIG

class DifferentialELAAnalyzer:
    """Multi-scale Error Level Analysis with dynamic baseline thresholding."""

    @staticmethod
    def analyze(orig_pil: Image.Image, img_cv: np.ndarray, qr_bbox: list = None) -> dict:
        rgb_pil = orig_pil.convert("RGB")
        buffered = io.BytesIO()
        rgb_pil.save(buffered, format="JPEG", quality=CONFIG.ELA_JPEG_QUALITY)
        buffered.seek(0)
        resaved = Image.open(buffered)

        # Compute absolute difference
        ela_diff = ImageChops.difference(rgb_pil, resaved)
        extrema = ela_diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1.0
        ela_enhanced = ImageEnhance.Brightness(ela_diff).enhance(scale)
        
        ela_cv = cv2.cvtColor(np.array(ela_enhanced), cv2.COLOR_RGB2BGR)
        gray_ela = cv2.cvtColor(ela_cv, cv2.COLOR_BGR2GRAY)

        # Restrict statistical baseline to card surface
        orig_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        doc_mask = orig_gray > 40
        valid_pixels = gray_ela[doc_mask] if np.sum(doc_mask) > 400 else gray_ela

        mean_val = float(np.mean(valid_pixels))
        std_val = float(np.std(valid_pixels))
        
        dynamic_thresh = max(
            CONFIG.ELA_DYNAMIC_THRESH_FLOOR,
            min(int(mean_val + (1.85 * std_val)), CONFIG.ELA_DYNAMIC_THRESH_CEIL)
        )

        blur = cv2.GaussianBlur(gray_ela, (3, 3), 0)
        _, thresh = cv2.threshold(blur, dynamic_thresh, 255, cv2.THRESH_BINARY)

        # Mask QR region to eliminate checkerboard ringing
        if qr_bbox:
            qx, qy, qw, qh = qr_bbox
            pad = 8
            cv2.rectangle(
                thresh,
                (max(0, qx - pad), max(0, qy - pad)),
                (min(thresh.shape[1], qx + qw + pad), min(thresh.shape[0], qy + qh + pad)),
                0, -1
            )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = gray_ela.shape
        total_area = h_img * w_img

        suspicious_zones = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if CONFIG.MIN_TAMPER_CONTOUR_AREA < area < (total_area * CONFIG.MAX_TAMPER_CONTOUR_AREA_RATIO):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / max(h, 1)

                # Filter out long divider lines
                if aspect > 5.5 and h < 12:
                    continue

                suspicious_zones.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area": int(area),
                    "signal": "Compression Discontinuity (ELA)"
                })

        return {
            "ela_enhanced": ela_enhanced,
            "ela_variance": round(std_val, 2),
            "dynamic_threshold": dynamic_thresh,
            "suspicious_zones": suspicious_zones
        }
