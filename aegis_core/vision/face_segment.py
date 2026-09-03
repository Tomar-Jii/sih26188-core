import cv2
import numpy as np

class BiometricPortraitAnalyzer:
    """Isolates facial portraits and audits boundary continuity for photo-replacement/swap signatures."""

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
            faces = []

            # 1. Primary Path: OpenCV CascadeClassifier (if available in environment)
            if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                try:
                    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    if not face_cascade.empty():
                        detected = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=4,
                            minSize=(int(min(h_img, w_img) * 0.12), int(min(h_img, w_img) * 0.12))
                        )
                        if len(detected) > 0:
                            faces = detected
                except Exception:
                    pass

            # 2. Resilient Fallback: Document Portrait Box Geometric & Hue Profiling
            # ID cards (Aadhaar/Passports) always place the portrait in the left quadrant (x < 42% width)
            if len(faces) == 0:
                hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV) if len(img_cv.shape) == 3 else None
                if hsv is not None:
                    # Skin tone & portrait zone mask
                    skin_mask = cv2.inRange(hsv, np.array([0, 25, 45]), np.array([28, 200, 250]))
                    # Restrict candidate search to standard portrait column (left 45% of card)
                    portrait_column = skin_mask[:, :int(w_img * 0.45)]
                    contours, _ = cv2.findContours(portrait_column, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    candidate_boxes = []
                    total_area = h_img * w_img
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if (total_area * 0.02) < area < (total_area * 0.35):
                            x, y, w, h = cv2.boundingRect(cnt)
                            ratio = float(w) / max(h, 1)
                            # Portrait aspect ratio (width:height ~ 0.65 to 1.15)
                            if 0.60 <= ratio <= 1.20 and y > (h_img * 0.12):
                                candidate_boxes.append((x, y, w, h))

                    if candidate_boxes:
                        faces = [max(candidate_boxes, key=lambda b: b[2] * b[3])]

            if len(faces) == 0:
                return fallback_res

            # Select the primary portrait candidate
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])

            pad_x = int(fw * 0.15)
            pad_y = int(fh * 0.20)
            x1 = max(0, fx - pad_x)
            y1 = max(0, fy - pad_y)
            x2 = min(w_img, fx + fw + pad_x)
            y2 = min(h_img, fy + fh + pad_y)

            face_crop = img_cv[y1:y2, x1:x2].copy()

            # Boundary perimeter gradient step
            pad = 4
            bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
            bx2, by2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
            perimeter = gray[by1:by2, bx1:bx2]

            sobel_x = cv2.Sobel(perimeter, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(perimeter, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            mean_border_grad = float(np.mean(grad_mag)) if perimeter.size > 0 else 0.0

            swap_score = min(1.0, round(mean_border_grad / 140.0, 2))
            anomaly_detected = swap_score > 0.74

            tamper_zone = None
            if anomaly_detected:
                tamper_zone = {
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "score": round(swap_score, 2),
                    "signal": "Biometric Photo-Swap Boundary Step"
                }

            return {
                "face_detected": True,
                "face_crop": face_crop,
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "photo_swap_score": swap_score,
                "swap_score": swap_score,
                "anomaly_detected": anomaly_detected,
                "tamper_zone": tamper_zone
            }

        except Exception:
            return fallback_res
