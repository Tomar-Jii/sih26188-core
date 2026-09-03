import cv2
import numpy as np

class FontDisparityAnalyzer:
    """Audits typographic baseline alignment with strict filtering for multi-script (Indic/Latin) documents."""

    @classmethod
    def audit(cls, img_cv: np.ndarray, qr_bbox: list = None, face_bbox: list = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"tamper_zones": [], "count": 0, "anomalous_rows": 0}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        h_img, w_img = gray.shape[:2]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Quarantine QR Code & Face
        if qr_bbox:
            qx, qy, qw, qh = qr_bbox
            pad = 10
            cv2.rectangle(thresh, (max(0, qx - pad), max(0, qy - pad)),
                          (min(w_img, qx + qw + pad), min(h_img, qy + qh + pad)), 0, -1)

        if face_bbox:
            fx, fy, fw, fh = face_bbox
            pad = 8
            cv2.rectangle(thresh, (max(0, fx - pad), max(0, fy - pad)),
                          (min(w_img, fx + fw + pad), min(h_img, fy + fh + pad)), 0, -1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        glyphs = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / max(h, 1)

            # Strict character bounds
            if 25 < area < 800 and 10 <= h <= 36 and 0.20 <= aspect <= 1.8:
                glyphs.append({
                    "bbox": (x, y, w, h),
                    "baseline_y": y + h,
                    "height": h,
                    "cy": y + (h // 2)
                })

        if len(glyphs) < 8:
            return {"tamper_zones": [], "count": 0, "anomalous_rows": 0}

        glyphs_sorted = sorted(glyphs, key=lambda g: g["cy"])
        rows = []
        current_row = [glyphs_sorted[0]]

        for g in glyphs_sorted[1:]:
            if abs(g["cy"] - current_row[-1]["cy"]) <= 6:
                current_row.append(g)
            else:
                if len(current_row) >= 5:
                    rows.append(current_row)
                current_row = [g]

        if len(current_row) >= 5:
            rows.append(current_row)

        tamper_zones = []
        anomalous_rows = 0

        for row in rows:
            baselines = [g["baseline_y"] for g in row]
            heights = [g["height"] for g in row]

            median_base = float(np.median(baselines))
            median_h = float(np.median(heights))
            std_h = float(np.std(heights))

            # Indic scripts naturally have std_h > 3.0 due to matras; skip checking baseline on Hindi rows
            if std_h > 3.5:
                continue

            row_flagged = False
            for g in row:
                base_drift = abs(g["baseline_y"] - median_base)
                height_drift = abs(g["height"] - median_h)

                # Extreme outlier check (spliced foreign digits in numbers/dates)
                if base_drift >= 7.0 or height_drift >= max(6.0, 2.5 * std_h):
                    x, y, w, h = g["bbox"]
                    tamper_zones.append({
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "score": 0.85,
                        "signal": "Typographic Disparity (Altered Character)"
                    })
                    row_flagged = True

            if row_flagged:
                anomalous_rows += 1

        return {
            "tamper_zones": tamper_zones,
            "count": len(tamper_zones),
            "anomalous_rows": anomalous_rows
        }
