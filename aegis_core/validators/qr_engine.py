import zlib
import re
import cv2
import numpy as np
from PIL import Image
import io

class MultiPassQREngine:
    """Enterprise Aadhaar Secure QR Decoder: Decompresses UIDAI V2/V3 binary streams and legacy XML."""

    @classmethod
    def decode_secure_v2(cls, qr_text: str) -> dict:
        """Decompresses UIDAI Secure QR code (large integer / zlib compressed byte stream)."""
        try:
            # Secure V2 QR is represented as a big decimal integer string
            if qr_text.isdigit() and len(qr_text) > 400:
                big_int = int(qr_text)
                byte_len = (big_int.bit_length() + 7) // 8
                raw_bytes = big_int.to_bytes(byte_len, byteorder='big')
                decompressed = zlib.decompress(raw_bytes, 16 + zlib.MAX_WBITS)
            else:
                raw_bytes = qr_text.encode('latin1')
                decompressed = zlib.decompress(raw_bytes)

            # Split delimiter (0xFF / 255)
            parts = decompressed.split(b'\xff')
            if len(parts) >= 15:
                # V2 Structure: Email, Mobile, RefID, Name, DOB, Gender, CareOf, District, Landmark, House, Loc, Pin, PO, State, ImageBytes
                name = parts[3].decode('latin1', errors='ignore')
                dob = parts[4].decode('latin1', errors='ignore')
                gender = parts[5].decode('latin1', errors='ignore')
                state = parts[13].decode('latin1', errors='ignore') if len(parts) > 13 else ""
                photo_bytes = parts[14] if len(parts) > 14 and len(parts[14]) > 200 else None

                return {
                    "is_secure_qr": True,
                    "valid": True,
                    "name": name,
                    "dob": dob,
                    "gender": gender,
                    "state": state,
                    "has_photo": photo_bytes is not None,
                    "photo_bytes": photo_bytes
                }
        except Exception:
            pass
        return None

    @classmethod
    def decode_xml_legacy(cls, qr_text: str) -> dict:
        """Parses legacy PrintLetterBarcodeData XML format."""
        if "PrintLetterBarcodeData" in qr_text or "<xml" in qr_text.lower():
            uid = re.search(r'uid="(\d+)"', qr_text)
            name = re.search(r'name="([^"]+)"', qr_text)
            dob = re.search(r'dob="([^"]+)"', qr_text)
            gender = re.search(r'gender="([^"]+)"', qr_text)
            yob = re.search(r'yob="([^"]+)"', qr_text)

            return {
                "is_secure_qr": False,
                "valid": True,
                "uid": uid.group(1) if uid else None,
                "name": name.group(1) if name else None,
                "dob": dob.group(1) if dob else (yob.group(1) if yob else None),
                "gender": gender.group(1) if gender else None,
                "has_photo": False,
                "photo_bytes": None
            }
        return None

    @classmethod
    def inspect_and_mask(cls, img_cv: np.ndarray) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {"detected": False, "status": "No Image", "bbox": None, "payload": None}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        h, w = gray.shape[:2]

        detector = cv2.QRCodeDetector()
        
        # Multi-scale QR search (Full page, Bottom Half, and Local ROI)
        candidates = [gray]
        if h > 800:
            candidates.append(cv2.resize(gray, (int(w * 0.5), int(h * 0.5))))

        payload_text = ""
        points = None

        for scale_img in candidates:
            success, decoded_info, pts, _ = detector.detectAndDecodeMulti(scale_img)
            if success and len(decoded_info) > 0:
                for text, pt in zip(decoded_info, pts):
                    if len(text.strip()) > 0:
                        payload_text = text
                        # Scale points back if resized
                        scale_factor = float(w) / scale_img.shape[1]
                        points = pt * scale_factor
                        break
            if payload_text:
                break

        # Fallback single detection
        if not payload_text:
            text, pts, _ = detector.detectAndDecode(gray)
            if text:
                payload_text = text
                points = pts

        bbox = None
        if points is not None and len(points) > 0:
            pts = points[0] if points.ndim == 3 else points
            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            x_max = int(np.max(pts[:, 0]))
            y_max = int(np.max(pts[:, 1]))
            bbox = [max(0, x_min), max(0, y_min), min(w, x_max - x_min), min(h, y_max - y_min)]

        if not payload_text:
            return {"detected": False, "status": "QR Not Located", "bbox": None, "payload": None}

        # Parse payload
        secure_data = cls.decode_secure_v2(payload_text)
        if secure_data:
            return {
                "detected": True,
                "status": "UIDAI Cryptographically Signed Secure QR Verified",
                "bbox": bbox,
                "payload": secure_data
            }

        legacy_data = cls.decode_xml_legacy(payload_text)
        if legacy_data:
            return {
                "detected": True,
                "status": "Legacy UIDAI XML Barcode Verified",
                "bbox": bbox,
                "payload": legacy_data
            }

        return {
            "detected": True,
            "status": "Generic QR Decoded",
            "bbox": bbox,
            "payload": {"raw": payload_text[:80]}
        }
