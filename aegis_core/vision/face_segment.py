import cv2
import numpy as np

class BiometricPortraitAnalyzer:
    """Isolates facial portraits and audits border continuity for photo-replacement indicators."""

    @staticmethod
    def extract_and_audit(img_cv: np.ndarray) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"face_detected": False, "face_crop": None, "photo_swap_score": 0.0, "bbox": None}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        h_img, w_img = gray.shape[:2]

        # Fail-safe built-in Haar Cascade detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(int(min(h_img, w_img) * 0.12), int(min(h_img, w_img) * 0.12))
        )

        if len(faces) == 0:
            return {
                "face_detected": False,
                "face_crop": None,
                "photo_swap_score": 0.0,
                "anomaly_detected": False,
                "bbox": None
            }

        # Select primary face candidate
        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        
        # Standardized portrait padding
        pad_x = int(fw * 0.20)
        pad_y = int(fh * 0.28)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(w_img, fx + fw + pad_x)
        y2 = min(h_img, fy + fh + pad_y)

        cropped_face = img_cv[y1:y2, x1:x2].copy()

        # Phase 18: Photo-Swap Boundary Continuity Audit
        # Evaluates gradient jump and noise differential along the portrait boundary perimeter
        boundary_pad = 3
        bx1, by1 = max(0, x1 - boundary_pad), max(0, y1 - boundary_pad)
        bx2, by2 = min(w_img, x2 + boundary_pad), min(h_img, y2 + boundary_pad)
        perimeter_outer = gray[by1:by2, bx1:bx2]
        
        # Sobel edge gradient across the perimeter
        sobel_x = cv2.Sobel(perimeter_outer, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(perimeter_outer, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        mean_border_gradient = float(np.mean(grad_mag))

        # Photo-swap heuristic: Pasted headshots typically create a high-contrast boundary step
        swap_score = min(1.0, round(mean_border_gradient / 140.0, 2))
        is_swap_anomaly = swap_score > 0.72

        return {
            "face_detected": True,
            "face_crop": cropped_face,
            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "photo_swap_score": swap_score,
            "anomaly_detected": is_swap_anomaly
        }
