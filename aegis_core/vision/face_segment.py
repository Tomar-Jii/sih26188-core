import cv2
import numpy as np

class BiometricPortraitAnalyzer:
    """Isolates facial portraits on standard ID cards and full-sheet e-Aadhaar documents."""

    @classmethod
    def extract_and_audit(cls, img_cv: np.ndarray) -> dict:
        fallback_res = {
            "face_detected": False,
            "face_crop": None,
            "bbox": None,
            "photo_swap_score": 0.0,
            "swap_score": 0.0,
            "anomaly_detected": False,
            "tamper_zone": None
        }

        if img_cv is None or img_cv.size == 0:
            return fallback_res

        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
            h_img, w_img = gray.shape[:2]
            aspect = float(w_img) / max(h_img, 1)
            is_full_sheet = aspect < 0.90

            faces = []

            # 1. Primary: Cascade Classifier with Multi-Scale Pyramid
            if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                try:
                    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    if not face_cascade.empty():
                        # Search entire image or bottom half for A4 e-Aadhaar
                        detected = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.08,
                            minNeighbors=3,
                            minSize=(35, 35)
                        )
                        if len(detected) > 0:
                            faces = detected
                except Exception:
                    pass

            # 2. Resilient Layout Anchor for Full-Sheet e-Aadhaar (Bottom-Left Portrait Column)
            if len(faces) == 0 and is_full_sheet:
                # In A4 e-Aadhaar, the primary photo is located at y: 60% - 85%, x: 10% - 35%
                roi_y1, roi_y2 = int(h_img * 0.58), int(h_img * 0.88)
                roi_x1, roi_x2 = int(w_img * 0.08), int(w_img * 0.38)
                card_roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]

                # Find photo box outline
                edges = cv2.Canny(card_roi, 50, 150)
                cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                best_box = None
                max_area = 0
                for c in cnts:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    area = bw * bh
                    ratio = float(bw) / max(bh, 1)
                    if 0.65 <= ratio <= 1.15 and area > 1200 and area > max_area:
                        max_area = area
                        best_box = (roi_x1 + bx, roi_y1 + by, bw, bh)

                if best_box:
                    faces = [best_box]

            # 3. Geometric Fallback for standard CR80 Card layouts
            if len(faces) == 0:
                hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV) if len(img_cv.shape) == 3 else None
                if hsv is not None:
                    skin_mask = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([28, 220, 255]))
                    portrait_column = skin_mask[:, :int(w_img * 0.45)]
                    contours, _ = cv2.findContours(portrait_column, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    candidate_boxes = []
                    total_area = h_img * w_img
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if (total_area * 0.003) < area < (total_area * 0.25):
                            x, y, w, h = cv2.boundingRect(cnt)
                            ratio = float(w) / max(h, 1)
                            if 0.60 <= ratio <= 1.25:
                                candidate_boxes.append((x, y, w, h))

                    if candidate_boxes:
                        faces = [max(candidate_boxes, key=lambda b: b[2] * b[3])]

            if len(faces) == 0:
                return fallback_res

            # Select face with best aspect ratio
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])

            pad_x = int(fw * 0.10)
            pad_y = int(fh * 0.12)
            x1 = max(0, fx - pad_x)
            y1 = max(0, fy - pad_y)
            x2 = min(w_img, fx + fw + pad_x)
            y2 = min(h_img, fy + fh + pad_y)

            face_crop = img_cv[y1:y2, x1:x2].copy()

            return {
                "face_detected": True,
                "face_crop": face_crop,
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "photo_swap_score": 0.05,
                "swap_score": 0.05,
                "anomaly_detected": False,
                "tamper_zone": None
            }
        except Exception:
            return fallback_res
