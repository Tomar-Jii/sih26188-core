import cv2
import numpy as np

class TextureFlatnessAnalyzer:
    """Detects physical defacement: Hair scribbles, digit obscurations, and strike-throughs."""

    @classmethod
    def detect_digital_strokes(cls, img_cv: np.ndarray, qr_bbox: list = None, **kwargs) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "mean_variance": 0.0}

        h_img, w_img = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        tamper_zones = []

        # Collect all QR boxes to mask
        all_qr_boxes = kwargs.get("qr_bboxes", [])
        if qr_bbox and qr_bbox not in all_qr_boxes:
            all_qr_boxes.append(qr_bbox)

        # ---------------------------------------------------------------------
        # 1. BIOMETRIC HEAD/HAIR DOODLE SCANNER (ENTROPY-VERIFIED)
        # ---------------------------------------------------------------------
        # Target portrait regions across standard ID layouts
        px1, py1 = int(w_img * 0.05), int(h_img * 0.25)
        px2, py2 = int(w_img * 0.45), int(h_img * 0.75)
        face_roi = gray[py1:py2, px1:px2]

        if face_roi.size > 0:
            _, dark_hair = cv2.threshold(face_roi, 36, 255, cv2.THRESH_BINARY_INV)
            k_hair = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned_hair = cv2.morphologyEx(dark_hair, cv2.MORPH_OPEN, k_hair)
            cnts_face, _ = cv2.findContours(cleaned_hair, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in cnts_face:
                area = cv2.contourArea(cnt)
                if 45 < area < 900:
                    x, y, w, h = cv2.boundingRect(cnt)
                    patch = face_roi[y:y+h, x:x+w]
                    
                    # ENTROPY CHECK:
                    # Real photographic hair has high texture variance (std > 5.5) due to camera sensor grain.
                    # A digital pen/brush scribble has artificial zero-grain flatness (std < 2.5).
                    patch_std = float(np.std(patch))
                    if patch_std < 3.2:
                        tamper_zones.append({
                            "bbox": [int(px1 + x), int(py1 + y), int(w), int(h)],
                            "score": 0.98,
                            "signal": "Biometric Portrait Defacement (Synthetic Hair Doodle)"
                        })

        # ---------------------------------------------------------------------
        # 2. OBSCURED / DEFACED DIGIT CLUSTERS (e.g. Cut numbers)
        # ---------------------------------------------------------------------
        num_y1, num_y2 = int(h_img * 0.50), int(h_img * 0.92)
        num_band = gray[num_y1:num_y2, :]

        if num_band.size > 0:
            _, dark_num = cv2.threshold(num_band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Mask ALL QR matrices out of number audit
            for qx, qy, qw, qh in all_qr_boxes:
                pad = 8
                ry1 = max(0, qy - num_y1 - pad)
                ry2 = min(num_band.shape[0], qy + qh - num_y1 + pad)
                rx1 = max(0, qx - pad)
                rx2 = min(w_img, qx + qw + pad)
                if ry2 > ry1 and rx2 > rx1:
                    dark_num[ry1:ry2, rx1:rx2] = 0

            cnts_num, _ = cv2.findContours(dark_num, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts_num:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / max(h, 1)

                hull = cv2.convexHull(cnt)
                solidity = float(area) / max(cv2.contourArea(hull), 1)

                # Flag multi-digit merged scribbles (high area, dense solidity, not standard text)
                if (area > 340 and w >= 28 and solidity > 0.48 and 0.45 <= aspect <= 3.0):
                    tamper_zones.append({
                        "bbox": [int(x), int(num_y1 + y), int(w), int(h)],
                        "score": 0.96,
                        "signal": "Obscured Digit Cluster (Number Defacement)"
                    })

        # ---------------------------------------------------------------------
        # 3. STRIKE-THROUGH LINE DETECTOR (Across Text)
        # ---------------------------------------------------------------------
        _, binary = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)
        
        # Mask ALL QR matrices
        for qx, qy, qw, qh in all_qr_boxes:
            pad = 8
            cv2.rectangle(binary, (max(0, qx - pad), max(0, qy - pad)),
                          (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)), 0, -1)

        k_thick_line = cv2.getStructuringElement(cv2.MORPH_RECT, (34, 3))
        thick_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_thick_line)

        cnts_lines, _ = cv2.findContours(thick_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_lines:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > int(w_img * 0.52):
                continue
            if 36 <= w <= int(w_img * 0.42) and 3 <= h <= 14:
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
