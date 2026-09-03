import cv2
import numpy as np

class BiometricPortraitAnalyzer:
    """Isolates facial portraits and audits boundary continuity for photo-replacement/swap signatures."""

    @classmethod
    def extract_and_audit(cls, img_cv: np.ndarray) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {
                "face_detected": False,
                "face_crop": None,
                "bbox": None,
                "swap_score": 0.0,
                "anomaly_detected": False,
                "tamper_zone": None
            }

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        h_img, w_img = gray.shape[:2]

        # Fail-safe built-in OpenCV Haar Cascade detector
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
                "bbox": None,
                "swap_score": 0.0,
                "anomaly_detected": False,
                "tamper_zone": None
            }

        # Select the dominant face candidate
        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])

        # Standardized biometric portrait padding
        pad_x = int(fw * 0.22)
        pad_y = int(fh * 0.30)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(w_img, fx + fw + pad_x)
        y2 = min(h_img, fy + fh + pad_y)

        face_crop = img_cv[y1:y2, x1:x2].copy()

        # Audit boundary perimeter gradient step
        # Pasted headshots create an unnatural high-frequency step boundary where toner patterns break
        pad = 4
        bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
        bx2, by2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
        perimeter = gray[by1:by2, bx1:bx2]

        sobel_x = cv2.Sobel(perimeter, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(perimeter, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        mean_border_grad = float(np.mean(grad_mag))

        swap_score = min(1.0, round(mean_border_grad / 135.0, 2))
        anomaly_detected = swap_score > 0.72

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
            "swap_score": swap_score,
            "anomaly_detected": anomaly_detected,
            "tamper_zone": tamper_zone
        }
