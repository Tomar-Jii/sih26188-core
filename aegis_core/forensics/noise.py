import cv2
import numpy as np

class LocalNoiseAnalyzer:
    """Evaluates spatial sensor noise consistency (PRNU approximation) to detect spliced patches and digital inpainting."""

    @classmethod
    def audit(cls, img_cv: np.ndarray, qr_bbox: list = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "global_variance": 0.0}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        h_img, w_img = gray.shape[:2]
        total_area = h_img * w_img

        # 1. High-Pass Noise Residual Extraction
        denoised = cv2.medianBlur(gray, 3)
        residual = cv2.absdiff(gray, denoised).astype(np.float32)

        # 2. Document Surface Masking (Ignore dark ambient background)
        doc_mask = gray > 40
        valid_res = residual[doc_mask] if np.sum(doc_mask) > 500 else residual
        mean_noise = float(np.mean(valid_res))
        std_noise = float(np.std(valid_res))

        # 3. Local Windowed Variance Mapping (9x9 Kernel)
        mean_local = cv2.blur(residual, (9, 9))
        sq_local = cv2.blur(residual ** 2, (9, 9))
        local_var = np.maximum(sq_local - (mean_local ** 2), 0.0)

        # 4. Inconsistency Thresholding
        # Detect patches with significant noise deviation (spliced from different cameras or denoised)
        var_norm = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dynamic_thresh = max(65, min(int(mean_noise + (2.1 * std_noise)), 110))
        _, anomaly_mask = cv2.threshold(var_norm, dynamic_thresh, 255, cv2.THRESH_BINARY)
        anomaly_mask = cv2.bitwise_and(anomaly_mask, anomaly_mask, mask=doc_mask.astype(np.uint8) * 255)

        # 5. Mask Out Known Structured Regions (QR Matrix)
        if qr_bbox:
            qx, qy, qw, qh = qr_bbox
            pad = 8
            cv2.rectangle(
                anomaly_mask,
                (max(0, qx - pad), max(0, qy - pad)),
                (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)),
                0, -1
            )

        # 6. Filter Extreme Horizontal Dividers
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        detected_lines = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel_h)
        clean_mask = cv2.bitwise_and(anomaly_mask, cv2.bitwise_not(detected_lines))

        # 7. Morphological Grouping
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        closed = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tamper_zones = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Area bounds: 20px to 10% of document area
            if 20 < area < (total_area * 0.10):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / max(h, 1)

                if aspect > 5.5 and h < 7:
                    continue

                tamper_zones.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "score": 0.87,
                    "signal": "Sensor Noise Inconsistency (PRNU Residual)"
                })

        return {
            "tamper_zones": tamper_zones,
            "count": len(tamper_zones),
            "global_variance": round(float(np.var(residual)), 2)
        }
