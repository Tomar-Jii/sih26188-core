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
        Gatekeeper: Verifies if the image is actually an ID card or just a random screenshot.
        """
        h, w = img_cv.shape[:2]
        aspect_ratio = max(w, h) / min(w, h)
        
        # 1. Aspect Ratio Test (Mobile screenshots are usually > 2.0 or < 0.5)
        is_screenshot_ratio = aspect_ratio > 2.05

        # 2. Cardholder Face Detection using OpenCV's built-in Haar Cascade
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        has_face = len(faces) > 0

        # An ID card MUST have a portrait photo and sensible dimensions
        is_valid_id = has_face and not is_screenshot_ratio

        reason = "Valid ID Card Profile"
        if not has_face and is_screenshot_ratio:
            reason = "Device Screenshot Detected: Aspect ratio exceeds ID standard and no cardholder portrait found."
        elif not has_face:
            reason = "No cardholder portrait face detected in document."
        elif is_screenshot_ratio:
            reason = "Invalid aspect ratio for government card standard."

        return {
            "is_valid_id": is_valid_id,
            "has_face": has_face,
            "face_count": len(faces),
            "aspect_ratio": round(aspect_ratio, 2),
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
        
        return {
            "papr_score": round(papr, 3),
            "is_screen_recapture": is_screen
        }

    @staticmethod
    def localize_tampering(orig_img: Image.Image, quality: int = 90) -> tuple:
        buffered = io.BytesIO()
        orig_img.save(buffered, format="JPEG", quality=quality)
        buffered.seek(0)
        resaved = Image.open(buffered)
        
        ela_im = ImageChops.difference(orig_img.convert("RGB"), resaved.convert("RGB"))
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1
        ela_enhanced = ImageEnhance.Brightness(ela_im).enhance(scale)
        
        ela_cv = cv2.cvtColor(np.array(ela_enhanced), cv2.COLOR_RGB2BGR)
        gray_ela = cv2.cvtColor(ela_cv, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray_ela, 145, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated_cv = cv2.cvtColor(np.array(orig_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        tamper_boxes = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 35000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated_cv, "TAMPER", (x, max(y - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                tamper_boxes += 1
                
        ela_stat = np.array(ela_enhanced)
        ela_variance = float(np.std(ela_stat))
        
        return ela_enhanced, annotated_cv, round(ela_variance, 2), tamper_boxes

    @staticmethod
    def audit_qr_code(img_cv: np.ndarray) -> dict:
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(img_cv)
        if points is not None and data:
            return {
                "detected": True,
                "payload_snippet": data[:60] + ("..." if len(data) > 60 else ""),
                "status": "QR Decoded Successfully"
            }
        return {"detected": False, "payload_snippet": "N/A", "status": "No readable QR code found"}
