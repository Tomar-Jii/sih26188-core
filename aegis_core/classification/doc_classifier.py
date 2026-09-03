import cv2
import numpy as np

class DocumentClassifier:
    """Classifies ID documents across standard CR80 cards and full-sheet e-Aadhaar formats."""

    @classmethod
    def classify(cls, img_cv: np.ndarray, mrz_res: dict = None, qr_res: dict = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"document_type": "UNKNOWN", "confidence": 0.0, "aspect_ratio": 0.0, "signatures": []}

        h, w = img_cv.shape[:2]
        aspect = round(float(w) / max(h, 1), 2)
        signatures = []

        # 1. Deterministic Passport MRZ
        if mrz_res and mrz_res.get("is_mrz_detected"):
            signatures.append("ICAO Doc 9303 MRZ token zone verified")
            return {
                "document_type": "PASSPORT_TD3",
                "confidence": 0.98,
                "aspect_ratio": aspect,
                "signatures": signatures
            }

        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Color signatures (Saffron, Green, Blue)
        saffron_mask = cv2.inRange(hsv, np.array([8, 70, 90]), np.array([25, 255, 255]))
        saffron_ratio = float(np.sum(saffron_mask > 0)) / max(img_cv.size // 3, 1)

        green_mask = cv2.inRange(hsv, np.array([35, 55, 55]), np.array([85, 255, 255]))
        green_ratio = float(np.sum(green_mask > 0)) / max(img_cv.size // 3, 1)

        is_cr80_card = 1.35 <= aspect <= 1.80
        is_full_a4_sheet = 0.60 <= aspect <= 0.85

        if saffron_ratio > 0.015 and green_ratio > 0.020:
            signatures.append("National Tri-color security bands (Saffron & Green)")
            if is_full_a4_sheet:
                signatures.append(f"Full-Sheet e-Aadhaar A4 Document Specimen (Aspect: {aspect})")
                return {
                    "document_type": "E_AADHAAR_FULL_DOCUMENT",
                    "confidence": 0.96,
                    "aspect_ratio": aspect,
                    "signatures": signatures
                }
            else:
                signatures.append(f"Standard CR80 Identity Card (Aspect: {aspect})")
                return {
                    "document_type": "AADHAAR_CARD",
                    "confidence": 0.95 if is_cr80_card else 0.84,
                    "aspect_ratio": aspect,
                    "signatures": signatures
                }

        if is_cr80_card:
            return {
                "document_type": "NATIONAL_ID_CARD",
                "confidence": 0.78,
                "aspect_ratio": aspect,
                "signatures": [f"Standard ISO/IEC 7810 ID-1 Aspect Ratio ({aspect})"]
            }

        return {
            "document_type": "IDENTITY_DOCUMENT_SPECIMEN",
            "confidence": 0.65,
            "aspect_ratio": aspect,
            "signatures": [f"Unspecified Dimension Profile ({aspect})"]
        }
