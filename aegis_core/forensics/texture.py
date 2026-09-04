import cv2
import numpy as np

class TextureFlatnessAnalyzer:
    """Enterprise Defacement Engine: Detects strike-throughs, obscured digit clusters, and portrait doodles."""

    @classmethod
    def detect_digital_strokes(cls, img_cv: np.ndarray, qr_bbox: list = None, face_bbox: list = None, **kwargs) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "mean_variance": 0.0}

        h_img, w_img = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        tamper_zones = []

        all_qr_boxes = list(kwargs.get("qr_bboxes", []))
        if qr_bbox and qr_bbox not in all_qr_boxes:
            all_qr_boxes.append(qr_bbox)

        # ---------------------------------------------------------------------
        # 0. SUBSTRATE MARGIN INSET (Eliminates desk border / table drop shadows)
        # ---------------------------------------------------------------------
        pad_y = int(h_img * 0.04)
        pad_x = int(w_img * 0.03)
        inner_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        inner_mask[pad_y:h_img - pad_y, pad_x:w_img - pad_x] = 255

        # Adaptive Otsu foreground extraction bounded to card interior
        _, raw_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        doc_ink = cv2.bitwise_and(raw_otsu, raw_otsu, mask=inner_mask)

        # Quarantine QR matrices from all texture scans
        for qx, qy, qw, qh in all_qr_boxes:
            pad = 12
            cv2.rectangle(
                doc_ink,
                (max(0, qx - pad), max(0, qy - pad)),
                (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)),
                0, -1
            )

        # Quarantine standard government logos (Ashoka emblem & UIDAI fingerprint)
        cv2.rectangle(doc_ink, (0, 0), (int(w_img * 0.22), int(h_img * 0.22)), 0, -1)
        cv2.rectangle(doc_ink, (int(w_img * 0.76), 0), (w_img, int(h_img * 0.22)), 0, -1)

        # ---------------------------------------------------------------------
        # 1. BIOMETRIC PORTRAIT DEFACEMENT (Hair / Face Scribble)
        # ---------------------------------------------------------------------
        if face_bbox:
            fx, fy, fw, fh = face_bbox
            # Focus strictly on the upper crown and hair zone
            crown_y1 = max(pad_y, fy - int(fh * 0.20))
            crown_y2 = min(h_img - pad_y, fy + int(fh * 0.35))
            crown_x1 = max(pad_x, fx)
            crown_x2 = min(w_img - pad_x, fx + fw)

            crown_roi = gray[crown_y1:crown_y2, crown_x1:crown_x2]
            if crown_roi.size > 0:
                roi_min = int(np.min(crown_roi))
                # Isolate high-contrast drawn marker / pen strokes over hair
                _, dark_doodle = cv2.threshold(crown_roi, min(48, roi_min + 22), 255, cv2.THRESH_BINARY_INV)
                k_doodle = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                cleaned_doodle = cv2.morphologyEx(dark_doodle, cv2.MORPH_OPEN, k_doodle)

                cnts_doodle, _ = cv2.findContours(cleaned_doodle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts_doodle:
                    area = cv2.contourArea(cnt)
                    dx, dy, dw, dh = cv2.boundingRect(cnt)
                    # Scribbles form bounded, concentrated markings
                    if 35 < area < 900 and dw < (fw * 0.65) and dh < (fh * 0.40):
                        tamper_zones.append({
                            "bbox": [int(crown_x1 + dx), int(crown_y1 + dy), int(dw), int(dh)],
                            "score": 0.98,
                            "signal": "Biometric Portrait Defacement (Hair Scribble)"
                        })

        # ---------------------------------------------------------------------
        # 2. CROSS-TEXT STRIKE-THROUGH STROKES
        # ---------------------------------------------------------------------
        ink_no_face = doc_ink.copy()
        if face_bbox:
            fx, fy, fw, fh = face_bbox
            cv2.rectangle(ink_no_face, (fx, fy), (fx + fw, fy + fh), 0, -1)

        # Horizontal kernel to bridge text strikes while filtering single glyphs
        k_strike = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 2))
        strike_layer = cv2.morphologyEx(ink_no_face, cv2.MORPH_OPEN, k_strike)

        cnts_strike, _ = cv2.findContours(strike_layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_strike:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter full-width divider rules and marginal lines
            if w > int(w_img * 0.48) or w < 26:
                continue
            if 2 <= h <= 16:
                # Vertical constraint: strike strokes cut through demographic text bands
                if int(h_img * 0.22) <= y <= int(h_img * 0.78):
                    tamper_zones.append({
                        "bbox": [int(x), int(y - 2), int(w), int(h + 4)],
                        "score": 0.96,
                        "signal": "Strike-Through Defacement Stroke"
                    })

        # ---------------------------------------------------------------------
        # 3. OBSCURED / DEFACED DIGIT CLUSTERS (Altered Numbers)
        # ---------------------------------------------------------------------
        num_y1, num_y2 = int(h_img * 0.52), int(h_img * 0.86)
        num_roi = doc_ink[num_y1:num_y2, :]

        cnts_num, _ = cv2.findContours(num_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_num:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / max(h, 1)

            # Legitimate single digits: w <= 24px, area <= 240px
            # Scribbled digits form merged multi-digit clusters
            hull = cv2.convexHull(cnt)
            solidity = float(area) / max(cv2.contourArea(hull), 1)

            if area > 260 and w >= 24 and solidity > 0.42 and 0.40 <= aspect <= 3.4 and 12 <= h <= 48:
                if aspect > 4.5 and h < 8:
                    continue
                tamper_zones.append({
                    "bbox": [int(x), int(num_y1 + y), int(w), int(h)],
                    "score": 0.97,
                    "signal": "Obscured Digit Cluster (Number Defacement)"
                })

        return {
            "tamper_zones": tamper_zones,
            "count": len(tamper_zones),
            "mean_variance": 0.0
        }
