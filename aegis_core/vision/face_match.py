import cv2
import numpy as np

class BiometricFaceMatcher:
    """Compares card canvas portrait against cryptographically signed QR avatar and live captured selfies."""

    @staticmethod
    def _preprocess(img_cv: np.ndarray, target_size=(112, 112)) -> np.ndarray:
        if img_cv is None or img_cv.size == 0:
            return None
        return cv2.resize(img_cv, target_size, interpolation=cv2.INTER_AREA)

    @classmethod
    def _compute_similarity(cls, c_face: np.ndarray, q_face: np.ndarray) -> float:
        """Core multi-space biometric similarity algorithm (Structural + Lab Color + Sobel Cosine)."""
        c_gray = cv2.cvtColor(c_face, cv2.COLOR_BGR2GRAY)
        q_gray = cv2.cvtColor(q_face, cv2.COLOR_BGR2GRAY)

        # 1. Structural Luminance Correlation
        c_norm = (c_gray.astype(np.float32) - np.mean(c_gray)) / (np.std(c_gray) + 1e-5)
        q_norm = (q_gray.astype(np.float32) - np.mean(q_gray)) / (np.std(q_gray) + 1e-5)
        structural_corr = float(np.mean(c_norm * q_norm))

        # 2. CIE-Lab Color & Skin Tone Correlation
        c_lab = cv2.cvtColor(c_face, cv2.COLOR_BGR2LAB)
        q_lab = cv2.cvtColor(q_face, cv2.COLOR_BGR2LAB)

        hist_c = cv2.calcHist([c_lab], [1, 2], None, [16, 16], [0, 256, 0, 256])
        hist_q = cv2.calcHist([q_lab], [1, 2], None, [16, 16], [0, 256, 0, 256])
        cv2.normalize(hist_c, hist_c, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_q, hist_q, 0, 1, cv2.NORM_MINMAX)
        color_corr = float(cv2.compareHist(hist_c, hist_q, cv2.HISTCMP_CORREL))

        # 3. Gradient Vector Cosine Similarity
        c_sobelx = cv2.Sobel(c_gray, cv2.CV_32F, 1, 0, ksize=3)
        c_sobely = cv2.Sobel(c_gray, cv2.CV_32F, 0, 1, ksize=3)
        q_sobelx = cv2.Sobel(q_gray, cv2.CV_32F, 1, 0, ksize=3)
        q_sobely = cv2.Sobel(q_gray, cv2.CV_32F, 0, 1, ksize=3)

        c_grad_mag = np.sqrt(c_sobelx**2 + c_sobely**2) + 1e-5
        q_grad_mag = np.sqrt(q_sobelx**2 + q_sobely**2) + 1e-5

        cos_sim = (c_sobelx * q_sobelx + c_sobely * q_sobely) / (c_grad_mag * q_grad_mag)
        grad_alignment = float(np.mean(np.maximum(cos_sim, 0.0)))

        composite = (0.45 * max(0.0, structural_corr)) + (0.35 * max(0.0, color_corr)) + (0.20 * grad_alignment)
        return round(max(0.0, min(1.0, composite)), 2)

    @classmethod
    def compare_portraits(cls, card_face_cv: np.ndarray, qr_photo_bytes: bytes) -> dict:
        """Existing Feature: Compares card portrait against QR avatar bytes."""
        default_res = {
            "evaluated": False,
            "match_status": "SKIPPED_NO_AVATAR",
            "similarity_score": 1.0,
            "is_photo_swap": False,
            "detail": "No cryptographically signed avatar available for verification."
        }

        if card_face_cv is None or card_face_cv.size == 0 or not qr_photo_bytes:
            return default_res

        try:
            qr_arr = np.frombuffer(qr_photo_bytes, dtype=np.uint8)
            qr_face_cv = cv2.imdecode(qr_arr, cv2.IMREAD_COLOR)

            if qr_face_cv is None or qr_face_cv.size == 0:
                return default_res

            c_face = cls._preprocess(card_face_cv)
            q_face = cls._preprocess(qr_face_cv)

            composite_score = cls._compute_similarity(c_face, q_face)
            is_swap = composite_score < 0.40
            match_status = "PHOTO_SWAP_DETECTED" if is_swap else "BIOMETRIC_AVATAR_CONFIRMED"

            return {
                "evaluated": True,
                "match_status": match_status,
                "similarity_score": composite_score,
                "is_photo_swap": is_swap,
                "qr_avatar_cv": q_face,
                "detail": f"Biometric correlation: {composite_score * 100:.1f}% against signed QR avatar."
            }
        except Exception as e:
            return {
                "evaluated": False,
                "match_status": "ERROR",
                "similarity_score": 0.0,
                "is_photo_swap": False,
                "detail": f"Biometric evaluation fault: {str(e)}"
            }

    @classmethod
    def compare_live_face(cls, card_face_cv: np.ndarray, live_img_cv: np.ndarray) -> dict:
        """New Feature: Extracts face from a live camera capture/selfie and compares with card photo."""
        default_res = {
            "evaluated": False,
            "live_face_detected": False,
            "match_status": "SKIPPED_NO_LIVE_FACE",
            "similarity_score": 0.0,
            "is_match": False,
            "live_face_crop": None,
            "detail": "No live camera frame provided for face matching."
        }

        if card_face_cv is None or card_face_cv.size == 0 or live_img_cv is None or live_img_cv.size == 0:
            return default_res

        try:
            live_gray = cv2.cvtColor(live_img_cv, cv2.COLOR_BGR2GRAY) if len(live_img_cv.shape) == 3 else live_img_cv
            h_l, w_l = live_gray.shape[:2]

            # Detect face in live image
            faces = []
            if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                try:
                    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    if not face_cascade.empty():
                        detected = face_cascade.detectMultiScale(live_gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                        if len(detected) > 0:
                            faces = detected
                except Exception:
                    pass

            if len(faces) == 0:
                # Fallback: Central bounding crop for selfies
                cx, cy = w_l // 2, h_l // 2
                box_sz = int(min(w_l, h_l) * 0.55)
                live_crop = live_img_cv[max(0, cy - box_sz//2):min(h_l, cy + box_sz//2),
                                        max(0, cx - box_sz//2):min(w_l, cx + box_sz//2)]
            else:
                fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                pad_x = int(fw * 0.15)
                pad_y = int(fh * 0.20)
                y1, y2 = max(0, fy - pad_y), min(h_l, fy + fh + pad_y)
                x1, x2 = max(0, fx - pad_x), min(w_l, fx + fw + pad_x)
                live_crop = live_img_cv[y1:y2, x1:x2]

            c_face = cls._preprocess(card_face_cv)
            l_face = cls._preprocess(live_crop)

            sim_score = cls._compute_similarity(c_face, l_face)
            is_match = sim_score >= 0.42
            match_status = "LIVE_FACE_MATCHED" if is_match else "LIVE_FACE_MISMATCH"

            return {
                "evaluated": True,
                "live_face_detected": True,
                "match_status": match_status,
                "similarity_score": sim_score,
                "is_match": is_match,
                "live_face_crop": l_face,
                "detail": f"Live selfie to card match score: {sim_score * 100:.1f}% ({'PASSED' if is_match else 'FAILED'})"
            }
        except Exception as e:
            return {
                "evaluated": False,
                "live_face_detected": False,
                "match_status": "ERROR",
                "similarity_score": 0.0,
                "is_match": False,
                "live_face_crop": None,
                "detail": f"Live selfie verification error: {str(e)}"
            }
