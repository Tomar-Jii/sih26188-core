import cv2
import numpy as np

class FontDisparityAnalyzer:
    """Audits typographic baseline alignment, character x-height variance, and glyph aspect ratio anomalies."""

    @classmethod
    def audit(cls, img_cv: np.ndarray, qr_bbox: list = None, face_bbox: list = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "anomalous_rows": 0}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        h_img, w_img = gray.shape[:2]

        # 1. High-Pass Contrast Binarization for Text Glyphs
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Quarantine Known Non-Text Regions (QR Code & Biometric Portrait)
        if qr_bbox:
            qx, qy, qw, qh = qr_bbox
            pad = 8
            cv2.rectangle(thresh, (max(0, qx - pad), max(0, qy - pad)),
                          (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)), 0, -1)

        if face_bbox:
            fx, fy, fw, fh = face_bbox
            pad = 6
            cv2.rectangle(thresh, (max(0, fx - pad), max(0, fy - pad)),
                          (min(w_img, fx + fw + pad), min(h_img, fy + fh + pad)), 0, -1)

        # 3. Extract Character-Scale Component Candidates
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        glyphs = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / max(h, 1)

            # Restrict to individual character scale (height 8px - 45px, reasonable glyph width)
            if 15 < area < 1200 and 8 <= h <= 45 and 0.15 <= aspect <= 2.2:
                glyphs.append({
                    "bbox": (x, y, w, h),
                    "baseline_y": y + h,
                    "height": h,
                    "width": w,
                    "aspect": aspect,
                    "cx": x + (w // 2),
                    "cy": y + (h // 2)
                })

        if len(glyphs) < 6:
            return {"tamper_zones": [], "count": 0, "anomalous_rows": 0}

        # 4. Cluster Glyphs into Horizontal Text Rows
        # Sort by vertical center position
        glyphs_sorted = sorted(glyphs, key=lambda g: g["cy"])
        rows = []
        current_row = [glyphs_sorted[0]]

        for g in glyphs_sorted[1:]:
            # If vertical center is within 6px, consider it part of the same text line
            if abs(g["cy"] - current_row[-1]["cy"]) <= 7:
                current_row.append(g)
            else:
                if len(current_row) >= 4:
                    rows.append(current_row)
                current_row = [g]

        if len(current_row) >= 4:
            rows.append(current_row)

        tamper_zones = []
        anomalous_rows = 0

        # 5. Typographic Consistency Inspection Per Row
        for row in rows:
            # Sort row characters left-to-right
            row = sorted(row, key=lambda g: g["bbox"][0])

            baselines = [g["baseline_y"] for g in row]
            heights = [g["height"] for g in row]

            median_base = float(np.median(baselines))
            median_h = float(np.median(heights))
            std_h = float(np.std(heights))

            row_flagged = False

            for g in row:
                base_drift = abs(g["baseline_y"] - median_base)
                height_drift = abs(g["height"] - median_h)

                # Flag glyphs with abrupt baseline jump (>= 4px) or significant height mismatch
                is_baseline_outlier = base_drift >= 4.0
                is_height_outlier = (std_h > 1.2) and (height_drift >= max(4.0, 1.9 * std_h))

                if is_baseline_outlier or is_height_outlier:
                    x, y, w, h = g["bbox"]
                    tamper_zones.append({
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "score": 0.88,
                        "signal": "Typographic Disparity (Baseline/Glyph Height Anomaly)"
                    })
                    row_flagged = True

            if row_flagged:
                anomalous_rows += 1

        return {
            "tamper_zones": tamper_zones,
            "count": len(tamper_zones),
            "anomalous_rows": anomalous_rows
        }
