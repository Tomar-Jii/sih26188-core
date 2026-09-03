import cv2
import numpy as np

class LocalNoiseAnalyzer:
    """Evaluates spatial sensor noise consistency strictly on substrate (excluding text edges)."""

    @classmethod
    def audit(cls, img_cv: np.ndarray, qr_bbox: list = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "global_variance": 0.0}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        h_img, w_img = gray.shape[:2]
        total_area = h_img * w_img

        denoised = cv2.medianBlur(gray, 3)
        residual = cv2.absdiff(gray, denoised).astype(np.float32)

        # MASK OUT TEXT EDGES: Text edges naturally have high median residual
        edges = cv2.Canny(gray, 50, 150)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_edges = cv2.dilate(edges, kernel_edge)
        residual[dilated_edges > 0] = 0.0

        doc_mask = gray > 40
        valid_res = residual[doc_mask] if np.sum(doc_mask) > 500 else residual
        mean_noise = float(np.mean(valid_res))
        std_noise = float(np.std(valid_res))

        mean_local = cv2.blur(residual, (15, 15))
        sq_local = cv2.blur(residual ** 2, (15, 15))
        local_var = np.maximum(sq_local - (mean_local ** 2), 0.0)

        var_norm = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dynamic_thresh = max(80, min(int(mean_noise + (2.5 * std_noise)), 130))
        _, anomaly_mask = cv2.threshold(var_norm, dynamic_thresh, 255, cv2.THRESH_BINARY)
        anomaly_mask = cv2.bitwise_and(anomaly_mask, anomaly_mask, mask=doc_mask.astype(np.uint8) * 255)

        if qr_bbox:
            qx, qy, qw, qh = qr_bbox
            pad = 8
            cv2.rectangle(anomaly_mask, (max(0, qx - pad), max(0, qy - pad)),
                          (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)), 0, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tamper_zones = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Only large spliced substrate patches (> 150px), never letter-sized boxes
            if 150 < area < (total_area * 0.10):
                x, y, w, h = cv2.boundingRect(cnt)
                tamper_zones.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "score": 0.86,
                    "signal": "Sensor Noise Inconsistency (PRNU Residual)"
                })

        return {"tamper_zones": tamper_zones, "count": len(tamper_zones), "global_variance": round(float(np.var(residual)), 2)}
