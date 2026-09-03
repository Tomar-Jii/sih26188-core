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
        """
        Fail-safe Gatekeeper: Separates mobile UI screenshots from actual card uploads.
        """
        h, w = img_cv.shape[:2]
        ratio = max(w, h) / min(w, h)
        
        # Phone screenshots are portrait 20:9 or 19:9 (ratio > 2.05)
        is_mobile_screenshot = (h > w) and (ratio > 1.95)
        
        # Color distribution check: Aadhaar/PAN have high white/cream card base
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        # check brightness / saturation
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        is_dark_screen = np.mean(val) < 45  # Dark mode phone screenshots

        is_valid = not is_mobile_screenshot and not is_dark_screen
        
        reason = "Valid Card Form Factor"
        if is_mobile_screenshot:
            reason = "Detected Mobile Device Screenshot (Aspect ratio matches phone screen, not an ID card)."
        elif is_dark_screen:
            reason = "Detected Dark Mode Interface / Non-Document Canvas."

        return {
            "is_valid_id": is_valid,
            "aspect_ratio": round(ratio, 2),
            "reason": reason
        }

    @staticmethod
    def audit_exif_metadata(image: Image.Image) -> dict:
        metadata = {}
        suspicious_tags = []
        editing_tools = ["photoshop", "gimp", "canva", "picsart", "coreldraw", "lightroom", "snapseed"]
        
        info = image.getexif()
        if info:
            for tag_id, value in info.items():
                tag_name = TAGS.get(tag_id, tag_id)
                val_str = str(value).lower()
                metadata[tag_name] = str(value)
                for tool in editing_tools:
                    if tool in val_str:
                        suspicious_tags.append(f"Editor footprint detected: '{tool.upper()}' in metadata")
        
        return {
            "has_exif": len(metadata) > 0,
            "software_traces": suspicious_tags,
            "raw_metadata": metadata
        }

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
            is_screen = papr > 3.65
            return {"papr_score": round(papr, 3), "is_screen_recapture": is_screen}
        except Exception:
            return {"papr_score": 1.0, "is_screen_recapture": False}

    @staticmethod
    def localize_tampering(orig_img: Image.Image, quality: int = 88) -> tuple:
        buffered = io.BytesIO()
        orig_img.save(buffered, format="JPEG", quality=quality)
        buffered.seek(0)
        resaved = Image.open(buffered)
        
        # High-precision Error Level difference
        ela_im = ImageChops.difference(orig_img.convert("RGB"), resaved.convert("RGB"))
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1
        ela_enhanced = ImageEnhance.Brightness(ela_im).enhance(scale)
        
        ela_cv = cv2.cvtColor(np.array(ela_enhanced), cv2.COLOR_RGB2BGR)
        gray_ela = cv2.cvtColor(ela_cv, cv2.COLOR_BGR2GRAY)
        
        # SENSITIVE THRESHOLD: Lowered from 145 to 42 to catch ink scribbles and markups
        blur = cv2.GaussianBlur(gray_ela, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 42, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated_cv = cv2.cvtColor(np.array(orig_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        tamper_boxes = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Catch scribbles, painted marks, edited numbers
            if 60 < area < 25000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated_cv, "TAMPER", (x, max(y - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                tamper_boxes += 1
                
        ela_variance = float(np.std(np.array(ela_enhanced)))
        return ela_enhanced, annotated_cv, round(ela_variance, 2), tamper_boxes

    @staticmethod
    def audit_qr_code(img_cv: np.ndarray) -> dict:
        try:
            detector = cv2.QRCodeDetector()
            data, points, _ = detector.detectAndDecode(img_cv)
            if points is not None and data:
                return {
                    "detected": True,
                    "payload_snippet": data[:60] + ("..." if len(data) > 60 else ""),
                    "status": "QR Decoded Successfully"
                }
        except Exception:
            pass
        return {"detected": False, "payload_snippet": "N/A", "status": "No readable QR code found"}
