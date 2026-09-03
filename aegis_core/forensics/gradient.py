import cv2
import numpy as np

class EdgeDiscontinuityAnalyzer:
    """Detects cut-and-paste boundary halos, sharp clipping borders, and splicing edge steps."""

    @classmethod
    def audit(cls, img_cv: np.ndarray, qr_bbox: list = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "mean_gradient": 0.0}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        h_img, w_img = gray.shape[:2]
        total_area = h_img * w_img

        # 1. Compute 2D Sobel Derivatives
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # 2. Local Contrast Normalization (Suppresses uniform lighting gradients)
        grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 3. Discontinuity Boundary Isolation
        # Paste halos and digital brush boundaries produce sharp, high-intensity transition spikes
        mean_g = float(np.mean(grad_norm))
        std_g = float(np.std(grad_norm))
        high_grad_thresh = max(90, min(int(mean_g + (2.2 * std_g)), 140))

        _, raw_edge_mask = cv2.threshold(grad_norm, high_grad_thresh, 255, cv2.THRESH_BINARY)

        # 4. Filter Document-Level Dividing Rules & Structural Lines
        # Real lines span > 55% of width and have thickness <= 5px
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        detected_lines = cv2.morphologyEx(raw_edge_mask, cv2.MORPH_OPEN, kernel_h)
        clean_edge_mask = cv2.bitwise_and(raw_edge_mask, cv2.bitwise_not(detected_lines))

        # 5. Mask Out QR Code Region
        if qr_bbox:
            qx, qy, qw, qh = qr_bbox
            pad = 6
            cv2.rectangle(
                clean_edge_mask,
                (max(0, qx - pad), max(0, qy - pad)),
                (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)),
                0, -1
            )

        # 6. Morphological Halo Bridging
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bridged = cv2.morphologyEx(clean_edge_mask, cv2.MORPH_CLOSE, kernel_bridge)

        contours, _ = cv2.findContours(bridged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tamper_zones = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Area floor 16px captures single edited digits and character clipping boxes
            if 16 < area < (total_area * 0.08):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / max(h, 1)

                # Skip extreme line artifacts
                if aspect > 6.0 and h < 6:
                    continue

                tamper_zones.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "score": 0.89,
                    "signal": "Boundary Gradient Step / Splicing Halo"
                })

        return {
            "tamper_zones": tamper_zones,
            "count": len(tamper_zones),
            "mean_gradient": round(mean_g, 2)
        }
