import cv2
import numpy as np

class TextureFlatnessAnalyzer:
    """Pinpoints true physical defacement: Hair scribbles, digit obscurations, and strike-throughs."""

    @classmethod
    def detect_digital_strokes(cls, img_cv: np.ndarray, qr_bbox: list = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "mean_variance": 0.0}

        h_img, w_img = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        tamper_zones = []

        # ---------------------------------------------------------------------
        # 1. BIOMETRIC HEAD/HAIR DOODLE SCANNER
        # ---------------------------------------------------------------------
        px1, py1 = int(w_img * 0.08), int(h_img * 0.28)
        px2, py2 = int(w_img * 0.38), int(h_img * 0.65)
        face_roi = gray[py1:py2, px1:px2]

        if face_roi.size > 0:
            # Unnatural dark digital brush in the hair region
            _, dark_hair = cv2.threshold(face_roi, 38, 255, cv2.THRESH_BINARY_INV)
            k_hair = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned_hair = cv2.morphologyEx(dark_hair, cv2.MORPH_OPEN, k_hair)
            cnts_face, _ = cv2.findContours(cleaned_hair, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in cnts_face:
                area = cv2.contourArea(cnt)
                if 40 < area < 1000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Upper quadrant of the head
                    if y < int((py2 - py1) * 0.45):
                        tamper_zones.append({
                            "bbox": [int(px1 + x), int(py1 + y), int(w), int(h)],
                            "score": 0.98,
                            "signal": "Biometric Portrait Defacement (Hair Scribble)"
                        })

        # ---------------------------------------------------------------------
        # 2. OBSCURED / DEFACED DIGIT CLUSTER (e.g. Cut '1014')
        # ---------------------------------------------------------------------
        # Aadhaar numbers sit between 58% and 85% height
        num_y1, num_y2 = int(h_img * 0.58), int(h_img * 0.85)
        num_band = gray[num_y1:num_y2, :]

        if num_band.size > 0:
            _, dark_num = cv2.threshold(num_band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Mask QR code if overlapping
            if qr_bbox:
                qx, qy, qw, qh = qr_bbox
                ry1 = max(0, qy - num_y1 - 6)
                ry2 = min(num_band.shape[0], qy + qh - num_y1 + 6)
                rx1 = max(0, qx - 6)
                rx2 = min(w_img, qx + qw + 6)
                if ry2 > ry1 and rx2 > rx1:
                    dark_num[ry1:ry2, rx1:rx2] = 0

            cnts_num, _ = cv2.findContours(dark_num, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts_num:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / max(h, 1)

                # Aadhaar single digits have area < 280px and aspect < 1.1
                # An obscured/scribbled cluster (like the crossed-out '1014') has merged thickness and high area
                hull = cv2.convexHull(cnt)
                solidity = float(area) / max(cv2.contourArea(hull), 1)

                if (area > 320 and w >= 26 and solidity > 0.45 and aspect < 3.2):
                    tamper_zones.append({
                        "bbox": [int(x), int(num_y1 + y), int(w), int(h)],
                        "score": 0.96,
                        "signal": "Obscured Digit Cluster (Number Defacement)"
                    })

        # ---------------------------------------------------------------------
        # 3. HEAVY STRIKE-THROUGH LINE (Shirorekha-Safe)
        # ---------------------------------------------------------------------
        # Shirorekha sits at the TOP of letters and is thin (height 1-2px).
        # A strike-through cuts across text with thickness >= 3px.
        _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        # Search specifically for thick horizontal cuts
        k_thick_line = cv2.getStructuringElement(cv2.MORPH_RECT, (32, 3))
        thick_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_thick_line)

        cnts_lines, _ = cv2.findContours(thick_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_lines:
            x, y, w, h = cv2.boundingRect(cnt)
            # Exclude full width template divider rules
            if w > int(w_img * 0.55):
                continue
            if 35 <= w <= int(w_img * 0.45) and 3 <= h <= 15:
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
