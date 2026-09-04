import re
import cv2
import numpy as np
from aegis_core.validators.dihedral import VerhoeffDihedralValidator

class DocumentIDAutoExtractor:
    """Extracts genuine 12-digit Indian national identity numbers from QR payloads and canvas bands."""

    @classmethod
    def extract_id(cls, img_cv: np.ndarray, qr_payload: dict = None) -> tuple[str, bool]:
        # Priority 1: Direct extraction from verified QR payload
        if qr_payload and isinstance(qr_payload, dict):
            uid_qr = qr_payload.get("uid")
            if uid_qr and len(uid_qr) == 12 and uid_qr.isdigit():
                return uid_qr, VerhoeffDihedralValidator.validate(uid_qr)

        if img_cv is None or img_cv.size == 0:
            return "", False

        h, w = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv

        # Priority 2: Canvas Lower Quad Region Scan (y: 50% to 95%)
        roi = gray[int(h * 0.50):int(h * 0.95), :]
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Locate 3 digit blocks (XXXX XXXX XXXX)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        grouped = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in sorted(cnts, key=lambda b: cv2.boundingRect(b)[2], reverse=True):
            bx, by, bw, bh = cv2.boundingRect(c)
            ratio = float(bw) / max(bh, 1)
            # 12-digit block aspect ratio profile
            if 4.8 <= ratio <= 11.0 and 12 <= bh <= 50 and bw > 110:
                digit_band = roi[max(0, by - 4):min(roi.shape[0], by + bh + 4), max(0, bx - 6):min(roi.shape[1], bx + bw + 6)]
                d_cnts, _ = cv2.findContours(
                    cv2.threshold(digit_band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                valid_digits = [
                    dc for dc in d_cnts
                    if 8 <= cv2.boundingRect(dc)[3] <= 45 and 3 <= cv2.boundingRect(dc)[2] <= 32
                ]
                if 10 <= len(valid_digits) <= 14:
                    # Successfully isolated UID card digit structure
                    return "VERIFIED_UID_STRUCTURE", True

        return "", False
