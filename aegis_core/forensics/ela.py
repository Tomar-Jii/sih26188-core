import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

class DifferentialELAAnalyzer:
    """Calibrated Error Level Analysis Engine (High-fidelity Heatmap & Global Variance)."""

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

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        doc_mask = gray > 35
        valid_pixels = channel_max[doc_mask] if np.sum(doc_mask) > 500 else channel_max

        std_val = float(np.std(valid_pixels))

        # ELA generates high-energy ringing around all legitimate printed text edges.
        # It serves as an optical visual heatmap for analysts; spatial bounding boxes
        # are reserved for structural and defacement signals to prevent text false positives.
        return {
            "suspicious_zones": [],
            "ela_enhanced": ela_enhanced,
            "ela_variance": round(std_val, 2)
        }
