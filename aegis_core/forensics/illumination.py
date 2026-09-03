import cv2
import numpy as np

class ChromaticIlluminationAnalyzer:
    """Evaluates spatial lighting continuity and CIE-Lab chromatic gradients to detect lighting-inconsistent spliced elements."""

    @classmethod
    def audit(cls, img_cv: np.ndarray, qr_bbox: list = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "mean_divergence": 0.0}

        h_img, w_img = img_cv.shape[:2]
        total_area = h_img * w_img

        # 1. Convert to CIE-Lab color space for luminance-chrominance decoupling
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        # 2. Approximate Illumination Surface via Low-Pass Gaussian Retinex
        illumination_field = cv2.GaussianBlur(l_chan.astype(np.float32), (31, 31), 0)

        # 3. Document Surface Mask (excluding dark background and specular flash glare)
        doc_mask = np.logical_and(l_chan > 35, l_chan < 245)

        # 4. Compute 2D Vector Gradients of the Illumination Surface
        grad_x = cv2.Sobel(illumination_field, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(illumination_field, cv2.CV_64F, 0, 1, ksize=5)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 5. Tile-Based Local Directional Consistency Evaluation (24x24 tiles)
        tile_size = 24
        anomaly_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        divergence_scores = []

        valid_mags = grad_mag[doc_mask]
        global_mean_mag = float(np.mean(valid_mags)) if valid_mags.size > 0 else 1.0
        global_std_mag = float(np.std(valid_mags)) if valid_mags.size > 0 else 1.0

        for y in range(0, h_img - tile_size, tile_size):
            for x in range(0, w_img - tile_size, tile_size):
                patch_mask = doc_mask[y:y+tile_size, x:x+tile_size]
                if np.sum(patch_mask) < (tile_size * tile_size * 0.45):
                    continue

                patch_mag = grad_mag[y:y+tile_size, x:x+tile_size]
                local_mean = float(np.mean(patch_mag))

                # Also audit chromatic divergence in a*, b* channels
                patch_a = a_chan[y:y+tile_size, x:x+tile_size].astype(np.float32)
                patch_b = b_chan[y:y+tile_size, x:x+tile_size].astype(np.float32)
                chroma_var = float(np.std(patch_a) + np.std(patch_b))

                # Identify abrupt illumination flux or localized color temperature discrepancy
                z_score = abs(local_mean - global_mean_mag) / max(global_std_mag, 1e-4)

                if z_score > 2.3 and chroma_var > 6.5:
                    anomaly_mask[y:y+tile_size, x:x+tile_size] = 255
                    divergence_scores.append(z_score)

        # 6. Quarantine QR Matrix Region
        if qr_bbox:
            qx, qy, qw, qh = qr_bbox
            pad = 8
            cv2.rectangle(
                anomaly_mask,
                (max(0, qx - pad), max(0, qy - pad)),
                (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)),
                0, -1
            )

        # 7. Morphological Cleanup & Region Extraction
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tamper_zones = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Retain plausible composite element areas (between 40px and 12% of card)
            if 40 < area < (total_area * 0.12):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / max(h, 1)

                if aspect > 5.5 and h < 8:
                    continue

                tamper_zones.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "score": 0.86,
                    "signal": "Illumination & Chromatic Gradient Divergence"
                })

        mean_div = round(float(np.mean(divergence_scores)), 2) if divergence_scores else 0.0

        return {
            "tamper_zones": tamper_zones,
            "count": len(tamper_zones),
            "mean_divergence": mean_div
        }
