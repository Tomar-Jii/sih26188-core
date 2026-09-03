import io
import re
import cv2
import hashlib
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS

class VerhoeffAlgorithm:
    d_table = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
        [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
        [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    p_table = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
    ]

    @classmethod
    def validate(cls, num_str: str) -> bool:
        clean = re.sub(r'\D', '', str(num_str))
        if len(clean) != 12:
            return False
        c = 0
        digits = [int(x) for x in clean]
        for i, item in enumerate(reversed(digits)):
            c = cls.d_table[c][cls.p_table[i % 8][item]]
        return c == 0

class DocumentForensicSuite:

    @staticmethod
    def compute_sha256(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    @staticmethod
    def validate_id_document_structure(img_cv: np.ndarray) -> dict:
        h, w = img_cv.shape[:2]
        ratio = max(w, h) / min(w, h)
        is_mobile_screenshot = (h > w) and (ratio > 1.95)
        
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        val = hsv[:, :, 2]
        is_dark_screen = np.mean(val) < 40

        is_valid = not is_mobile_screenshot and not is_dark_screen
        reason = "Valid Card Layout"
        if is_mobile_screenshot:
            reason = "Mobile UI Screenshot Aspect Ratio (Not a physical ID card)"
        elif is_dark_screen:
            reason = "Dark mode UI / Low-light screen capture"

        return {"is_valid_id": is_valid, "reason": reason}

    @staticmethod
    def audit_exif_metadata(image: Image.Image) -> dict:
        suspicious_tags = []
        editing_tools = ["photoshop", "gimp", "canva", "picsart", "coreldraw", "lightroom", "snapseed"]
        info = image.getexif()
        if info:
            for tag_id, value in info.items():
                val_str = str(value).lower()
                for tool in editing_tools:
                    if tool in val_str:
                        suspicious_tags.append(f"Trace of editor '{tool.upper()}' detected in metadata")
        return {"software_traces": suspicious_tags}

    @staticmethod
    def analyze_moire_frequency(img_cv: np.ndarray) -> dict:
        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (512, 512))
            dft = np.fft.fft2(gray)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-9)
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30] = 0
            papr = float(np.percentile(magnitude_spectrum, 99.8) / (np.mean(magnitude_spectrum) + 1e-5))
            return {"papr_score": round(papr, 3), "is_screen_recapture": papr > 3.85}
        except Exception:
            return {"papr_score": 1.0, "is_screen_recapture": False}

    @staticmethod
    def localize_tampering(orig_img: Image.Image, quality: int = 90) -> tuple:
        buffered = io.BytesIO()
        orig_img.save(buffered, format="JPEG", quality=quality)
        buffered.seek(0)
        resaved = Image.open(buffered)
        
        # High-res ELA computation
        ela_im = ImageChops.difference(orig_img.convert("RGB"), resaved.convert("RGB"))
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1
        ela_enhanced = ImageEnhance.Brightness(ela_im).enhance(scale)
        
        ela_cv = cv2.cvtColor(np.array(ela_enhanced), cv2.COLOR_RGB2BGR)
        gray_ela = cv2.cvtColor(ela_cv, cv2.COLOR_BGR2GRAY)
        
        # 1. ADAPTIVE STATISTICAL THRESHOLDING (Prevents clean text from triggering)
        mean_val = np.mean(gray_ela)
        std_val = np.std(gray_ela)
        adaptive_thresh_val = max(int(mean_val + (2.6 * std_val)), 70)
        
        blur = cv2.GaussianBlur(gray_ela, (7, 7), 0)
        _, thresh = cv2.threshold(blur, adaptive_thresh_val, 255, cv2.THRESH_BINARY)
        
        # 2. MORPHOLOGICAL CLUSTERING (Merges letters into single word bounding boxes)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated_cv = cv2.cvtColor(np.array(orig_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        img_h, img_w = gray_ela.shape
        total_img_area = img_h * img_w
        tamper_boxes = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise and entire document outlines
            if 350 < area < (total_img_area * 0.12):
                x, y, w, h = cv2.boundingRect(cnt)
                
                # 3. QR CODE FILTER: Ignore regular square QR code regions on the right side
                if area > 2800 and 0.82 < (w / float(h)) < 1.18 and x > (img_w * 0.45):
                    continue
                
                cv2.rectangle(annotated_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated_cv, "TAMPER", (x, max(y - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)
                tamper_boxes += 1
                
        return ela_enhanced, annotated_cv, round(std_val, 2), tamper_boxes

    @staticmethod
    def audit_qr_code(img_cv: np.ndarray) -> dict:
        try:
            detector = cv2.QRCodeDetector()
            data, points, _ = detector.detectAndDecode(img_cv)
            if points is not None and data:
                return {"detected": True, "status": "QR Decoded Successfully"}
        except Exception:
            pass
        return {"detected": False, "status": "No readable QR code found"}
