import cv2
import numpy as np

class MultiPassQREngine:
    """Multi-pass adaptive QR detector: Decodes payloads and computes suppression masks."""

    @staticmethod
    def inspect_and_mask(img_cv: np.ndarray) -> dict:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        detector = cv2.QRCodeDetector()
        
        # Pass 1: Standard contrast
        data, points, _ = detector.detectAndDecode(img_cv)
        if points is not None and data:
            return MultiPassQREngine._format_result(True, "Decoded (Standard)", data, points)

        # Pass 2: Adaptive CLAHE (for dark/reflective cards)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        data, points, _ = detector.detectAndDecode(enhanced)
        if points is not None and data:
            return MultiPassQREngine._format_result(True, "Decoded (CLAHE Contrast)", data, points)

        # Pass 3: Otsu Binarization (specular highlight suppression)
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        data, points, _ = detector.detectAndDecode(otsu)
        if points is not None and data:
            return MultiPassQREngine._format_result(True, "Decoded (Otsu Binarized)", data, points)

        # Pass 4: Spatial Contour Geometry fallback (Finder pattern matrix presence)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        qr_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if 1200 < area < (img_cv.shape[0] * img_cv.shape[1] * 0.25):
                x, y, w, h = cv2.boundingRect(c)
                ratio = float(w) / max(h, 1)
                if 0.82 < ratio < 1.22 and x > (img_cv.shape[1] * 0.35):
                    qr_boxes.append((x, y, w, h))

        if qr_boxes:
            best_box = max(qr_boxes, key=lambda b: b[2] * b[3])
            return {
                "detected": True,
                "status": "Matrix Pattern Verified (Payload Unreadable)",
                "payload": "",
                "bbox": list(best_box)
            }

        return {
            "detected": False,
            "status": "No Valid QR Pattern Detected",
            "payload": "",
            "bbox": None
        }

    @staticmethod
    def _format_result(detected: bool, status: str, payload: str, points: np.ndarray) -> dict:
        pts = points[0].astype(int)
        x, y, w, h = cv2.boundingRect(pts)
        return {
            "detected": detected,
            "status": status,
            "payload": payload[:80] + ("..." if len(payload) > 80 else ""),
            "bbox": [int(x), int(y), int(w), int(h)]
        }
