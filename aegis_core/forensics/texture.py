import cv2
import numpy as np

class TextureFlatnessAnalyzer:
    """Enterprise Defacement Engine: Detects strike-throughs, face doodles, and obscured digit clusters."""

    @classmethod
    def detect_digital_strokes(cls, img_cv: np.ndarray, qr_bbox: list = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "mean_variance": 0.0}

        h_img, w_img = img_cv.shape[:2]
        total_area = h_img * w_img
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv

        tamper_zones = []

        # ---------------------------------------------------------------------
        # 1. BIOMETRIC PORTRAIT DEFACEMENT SCANNER (Doodles on Face/Hair)
        # ---------------------------------------------------------------------
        # Left quadrant portrait area (x: 8% to 42%, y: 30% to 75%)
        px1, py1, px2, py2 = int(w_img * 0.08), int(h_img * 0.28), int(w_img * 0.40), int(h_img * 0.72)
        face_roi = gray[py1:py2, px1:px2]

        if face_roi.size > 0:
            # Detect unnaturally deep solid black strokes in the portrait area
            _, dark_hair_marks = cv2.threshold(face_roi, 42, 255, cv2.THRESH_BINARY_INV)
            
            # Clean tiny natural hair texture
            k_hair = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned_hair = cv2.morphologyEx(dark_hair_marks, cv2.MORPH_OPEN, k_hair)
            cnts_face, _ = cv2.findContours(cleaned_hair, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in cnts_face:
                area = cv2.contourArea(cnt)
                # Doodles on hair/face have solid area between 35px and 1200px
                if 35 < area < 1200:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Exclude the entire head outline; target specific scribbled clusters
                    if w < (px2 - px1) * 0.70 and h < (py2 - py1) * 0.50:
                        tamper_zones.append({
                            "bbox": [int(px1 + x), int(py1 + y), int(w), int(h)],
                            "score": 0.96,
                            "signal": "Biometric Portrait Defacement (Hair/Face Scribble)"
                        })

        # ---------------------------------------------------------------------
        # 2. CROSS-WORD STRIKE-THROUGH DETECTOR (Horizontal defacing strokes)
        # ---------------------------------------------------------------------
        _, binary_all = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Isolate horizontal strokes spanning across words (width 35px - 45% of card)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (28, 2))
        horizontal_strokes = cv2.morphologyEx(binary_all, cv2.MORPH_OPEN, h_kernel)

        cnts_strike, _ = cv2.findContours(horizontal_strokes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_strike:
            x, y, w, h = cv2.boundingRect(cnt)
            # Legitimate dividing lines span > 55% of card width
            if w > (w_img * 0.55):
                continue
            # Strike-through lines that cross over text words
            if 35 <= w <= (w_img * 0.45) and 2 <= h <= 18:
                tamper_zones.append({
                    "bbox": [int(x), int(y - 2), int(w), int(h + 4)],
                    "score": 0.94,
                    "signal": "Strike-Through Defacement Stroke"
                })

        # ---------------------------------------------------------------------
        # 3. OBSCURED / SCRIBBLED DIGIT CLUSTER DETECTOR (e.g. Tampered "1014")
        # ---------------------------------------------------------------------
        # In Aadhaar cards, the 12 digits sit in the lower half (y > 55% of card)
        num_band_y1, num_band_y2 = int(h_img * 0.58), int(h_img * 0.88)
        num_roi = binary_all[num_band_y1:num_band_y2, :]

        # Exclude QR Code area
        if qr_bbox:
            qx, qy, qw, qh = qr_bbox
            pad = 6
            ry1 = max(0, qy - num_band_y1 - pad)
            ry2 = min(num_roi.shape[0], qy + qh - num_band_y1 + pad)
            rx1 = max(0, qx - pad)
            rx2 = min(w_img, qx + qw + pad)
            if ry2 > ry1 and rx2 > rx1:
                num_roi[ry1:ry2, rx1:rx2] = 0

        cnts_num, _ = cv2.findContours(num_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_num:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / max(h, 1)

            # Normal single Aadhaar digits: width 8-28px, height 18-42px, area 60-320px
            # A scribbled/obscured digit cluster (like '1014') forms an amorphous blob:
            # Area > 340px OR width > 32px with high solidity, merging multiple digits
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / max(hull_area, 1)

            is_defaced_digit_cluster = (area > 340 and w > 28 and solidity > 0.45 and h < 55)
            is_messy_digit_scribble = (w >= 25 and h >= 20 and area > 280 and solidity > 0.50)

            # Do not flag legitimate long template bottom lines
            if aspect > 5.0 and h < 6:
                continue

            if is_defaced_digit_cluster or is_messy_digit_scribble:
                tamper_zones.append({
                    "bbox": [int(x), int(num_band_y1 + y), int(w), int(h)],
                    "score": 0.95,
                    "signal": "Obscured Digit Cluster (Number Tampering)"
                })

        return {
            "tamper_zones": tamper_zones,
            "count": len(tamper_zones),
            "mean_variance": 0.0
        }
