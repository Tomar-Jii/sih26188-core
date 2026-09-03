import cv2
import numpy as np

class LocalNoiseAnalyzer:
    """Evaluates spatial sensor noise consistency (PRNU approximation)."""

    @staticmethod
    def extract_residual_map(img_cv: np.ndarray) -> tuple:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        denoised = cv2.medianBlur(gray, 3)
        residual = cv2.absdiff(gray, denoised)
        global_variance = float(np.var(residual))
        norm_map = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
        return norm_map, round(global_variance, 2)

    @staticmethod
    def evaluate_region_noise(img_cv: np.ndarray, bbox: list) -> float:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        x, y, w, h = bbox
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            return 1.0
        denoised_roi = cv2.medianBlur(roi, 3)
        roi_noise = float(np.var(cv2.absdiff(roi, denoised_roi)))
        denoised_global = cv2.medianBlur(gray, 3)
        global_noise = float(np.var(cv2.absdiff(gray, denoised_global))) + 1e-5
        return round(roi_noise / global_noise, 3)

    @classmethod
    def audit(cls, img_cv: np.ndarray, qr_bbox: list = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "global_variance": 0.0}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        h_img, w_img = gray.shape[:2]
        total_area = h_img * w_img

        denoised = cv2.medianBlur(gray, 3)
        residual = cv2.absdiff(gray, denoised).astype(np.float32)

        doc_mask = gray > 40
        valid_res = residual[doc_mask] if np.sum(doc_mask) > 500 else residual
        mean_noise = float(np.mean(valid_res))
        std_noise = float(np.std(valid_res))

        mean_local = cv2.blur(residual, (9, 9))
        sq_local = cv2.blur(residual ** 2, (9, 9))
        local_var = np.maximum(sq_local - (mean_local ** 2), 0.0)

        var_norm = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dynamic_thresh = max(65, min(int(mean_noise + (2.1 * std_noise)), 110))
        _, anomaly_mask = cv2.threshold(var_norm, dynamic_thresh, 255, cv2.THRESH_BINARY)
        anomaly_mask = cv2.bitwise_and(anomaly_mask, anomaly_mask, mask=doc_mask.astype(np.uint8) * 255)

        if qr_bbox:
            qx, qy, qw, qh = qr_bbox
            pad = 8
            cv2.rectangle(anomaly_mask, (max(0, qx - pad), max(0, qy - pad)),
                          (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)), 0, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        closed = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tamper_zones = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < (total_area * 0.10):
                x, y, w, h = cv2.boundingRect(cnt)
                if float(w) / max(h, 1) > 5.5 and h < 7:
                    continue
                tamper_zones.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "score": 0.87,
                    "signal": "Sensor Noise Inconsistency (PRNU Residual)"
                })

        return {"tamper_zones": tamper_zones, "count": len(tamper_zones), "global_variance": round(float(np.var(residual)), 2)}
