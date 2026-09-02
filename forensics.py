import io
import cv2
import hashlib
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS

class DocumentForensicSuite:

    @staticmethod
    def compute_sha256(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    @staticmethod
    def audit_exif_metadata(image: Image.Image) -> dict:
        metadata = {}
        suspicious_tags = []
        editing_tools = ["photoshop", "gimp", "canva", "picsart", "coreldraw", "lightroom"]
        
        info = image.getexif()
        if info:
            for tag_id, value in info.items():
                tag_name = TAGS.get(tag_id, tag_id)
                val_str = str(value).lower()
                metadata[tag_name] = str(value)
                for tool in editing_tools:
                    if tool in val_str:
                        suspicious_tags.append(f"Trace of editing software detected: '{tool.upper()}'")
        
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
        
        _, thresh = cv2.threshold(gray_ela, 150, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated_cv = cv2.cvtColor(np.array(orig_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        tamper_boxes = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 120 < area < 40000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated_cv, "ANOMALY", (x, max(y - 5, 15)),
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
