import cv2
import numpy as np

class DocumentClassifier:
    """Classifies identity documents using geometric aspect ratios, color-space bands, and structural anchors."""

    @classmethod
    def classify(cls, img_cv: np.ndarray, mrz_res: dict = None, qr_res: dict = None) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {
                "document_type": "UNKNOWN",
                "confidence": 0.0,
                "aspect_ratio": 0.0,
                "signatures": [],
                "expected_fields": []
            }

        h, w = img_cv.shape[:2]
        aspect = round(float(w) / max(h, 1), 2)
        signatures = []
        doc_type = "GENERIC_IDENTITY_DOCUMENT"
        confidence = 0.50

        # 1. Deterministic ICAO MRZ Presence Override (Passport TD3)
        if mrz_res and mrz_res.get("is_mrz_detected"):
            signatures.append("ICAO Doc 9303 MRZ token zone verified")
            return {
                "document_type": "PASSPORT_TD3",
                "confidence": 0.98,
                "aspect_ratio": aspect,
                "signatures": signatures,
                "expected_fields": ["passport_number", "dob", "expiry", "nationality", "mrz"]
            }

        # 2. Aspect Ratio Profiling (Standard ISO/IEC 7810 ID-1 / CR80 is 85.6mm x 53.98mm = 1.586)
        is_cr80_ratio = 1.38 <= aspect <= 1.78
        if is_cr80_ratio:
            signatures.append(f"Standard CR80 ID-1 Aspect Ratio ({aspect})")

        # 3. HSV Color-Space Security Feature Inspection
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Analyze top header band (upper 25% of document)
        top_header = hsv[0:int(h * 0.25), :]
        
        # Saffron / Orange hue mask (Aadhaar header) -> H: 8-25
        saffron_mask = cv2.inRange(top_header, np.array([8, 70, 90]), np.array([25, 255, 255]))
        saffron_ratio = float(np.sum(saffron_mask > 0)) / max(top_header.size // 3, 1)

        # Green hue mask (Aadhaar header / bottom band) -> H: 35-85
        green_mask = cv2.inRange(hsv, np.array([35, 55, 55]), np.array([85, 255, 255]))
        green_ratio = float(np.sum(green_mask > 0)) / max(hsv.size // 3, 1)

        # Blue hue mask (PAN Card / Driving License header ribbon) -> H: 95-130
        blue_mask = cv2.inRange(top_header, np.array([95, 75, 75]), np.array([130, 255, 255]))
        blue_ratio = float(np.sum(blue_mask > 0)) / max(top_header.size // 3, 1)

        # 4. Feature Fusion
        if saffron_ratio > 0.035 and green_ratio > 0.04:
            signatures.append("National Tri-color header band signatures (Saffron & Green)")
            doc_type = "AADHAAR_CARD"
            confidence = 0.94 if is_cr80_ratio else 0.82
        elif blue_ratio > 0.10 and is_cr80_ratio:
            signatures.append("Government Blue security banner (PAN / State DL profile)")
            doc_type = "PAN_CARD"
            confidence = 0.86
        elif is_cr80_ratio:
            doc_type = "NATIONAL_ID_CARD"
            confidence = 0.74
        else:
            doc_type = "IDENTITY_DOCUMENT_SPECIMEN"
            confidence = 0.60

        # QR Matrix confirmation boost
        if qr_res and qr_res.get("detected"):
            signatures.append(f"Security Matrix QR verified ({qr_res.get('status')})")
            confidence = min(0.98, confidence + 0.05)

        expected_fields = []
        if doc_type == "AADHAAR_CARD":
            expected_fields = ["uid_number", "holder_name", "dob_or_yob", "gender", "qr_matrix"]
        elif doc_type == "PAN_CARD":
            expected_fields = ["pan_number", "holder_name", "father_name", "dob"]
        elif doc_type == "PASSPORT_TD3":
            expected_fields = ["passport_number", "surname", "given_names", "nationality", "mrz"]
        else:
            expected_fields = ["document_id", "holder_name", "validity"]

        return {
            "document_type": doc_type,
            "confidence": round(confidence, 2),
            "aspect_ratio": aspect,
            "signatures": signatures,
            "expected_fields": expected_fields
        }
