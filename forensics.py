import io
import re
import cv2
import hashlib
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS

class ICAO9303Validator:
    WEIGHTS = [7, 3, 1]

    @classmethod
    def compute_check_digit(cls, data_str: str) -> int:
        total = 0
        for i, char in enumerate(data_str):
            if '0' <= char <= '9':
                val = int(char)
            elif 'A' <= char <= 'Z':
                val = ord(char) - 55
            else:
                val = 0
            total += val * cls.WEIGHTS[i % 3]
        return total % 10

    @classmethod
    def parse_mrz(cls, raw_text: str) -> dict:
        if not raw_text or not isinstance(raw_text, str):
            return {"is_mrz_detected": False, "checks": {}, "all_checks_passed": False}

        lines = [line.strip().replace(" ", "").upper() for line in raw_text.splitlines() if len(line.strip()) >= 30]
        mrz_lines = [l for l in lines if re.match(r'^[A-Z0-9<]+$', l)]
        
        if len(mrz_lines) >= 2 and len(mrz_lines[0]) == 44 and len(mrz_lines[1]) == 44:
            l1, l2 = mrz_lines[0], mrz_lines[1]
            doc_num = l2[0:9]
            doc_num_check = l2[9]
            dob = l2[13:19]
            dob_check = l2[19]
            expiry = l2[21:27]
            expiry_check = l2[27]
            composite = l2[0:10] + l2[13:20] + l2[21:43]
            composite_check = l2[43]

            doc_valid = str(cls.compute_check_digit(doc_num)) == doc_num_check
            dob_valid = str(cls.compute_check_digit(dob)) == dob_check
            expiry_valid = str(cls.compute_check_digit(expiry)) == expiry_check
            comp_valid = str(cls.compute_check_digit(composite)) == composite_check

            return {
                "is_mrz_detected": True,
                "type": "TD3_PASSPORT",
                "doc_number": doc_num.replace("<", ""),
                "dob": dob,
                "expiry": expiry,
                "nationality": l2[10:13].replace("<", ""),
                "checks": {
                    "doc_number": "PASS" if doc_valid else "FAIL",
                    "dob": "PASS" if dob_valid else "FAIL",
                    "expiry": "PASS" if expiry_valid else "FAIL",
                    "composite": "PASS" if comp_valid else "FAIL"
                },
                "all_checks_passed": bool(doc_valid and dob_valid and expiry_valid and comp_valid)
            }

        return {"is_mrz_detected": False, "checks": {}, "all_checks_passed": False}


class DocumentQualityEngine:
    @staticmethod
    def assess_quality(img_cv: np.ndarray) -> dict:
        if img_cv is None or img_cv.size == 0:
            return {
                "passed": False,
                "abstain_reason": "Zero-dimension image buffer",
                "metrics": {"sharpness_laplacian": 0.0, "blur_status": "Unusable", "glare_status": "Unknown", "resolution": "0x0"}
            }

        h, w = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_status = "Good" if lap_var > 45.0 else ("Acceptable" if lap_var > 10.0 else "Poor (Blurry)")

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_pixels = max(h * w, 1)
        pure_white_ratio = float(np.sum(hist[250:]) / total_pixels)
        glare_status = "High (Glare Detected)" if pure_white_ratio > 0.20 else "Normal"

        passed = bool(lap_var > 10.0 and pure_white_ratio <= 0.30 and w >= 150 and h >= 150)
        abstain_reason = None
        if not passed:
            reasons = []
            if lap_var <= 10.0: reasons.append("Extreme Motion Blur")
            if pure_white_ratio > 0.30: reasons.append("Severe Overexposure")
            abstain_reason = "; ".join(reasons)

        return {
            "passed": passed,
            "abstain_reason": abstain_reason,
            "metrics": {
                "sharpness_laplacian": round(lap_var, 1),
                "blur_status": blur_status,
                "glare_status": glare_status,
                "resolution": f"{w}x{h}"
            }
        }


class MultiSignalForensics:
    @staticmethod
    def compute_sha256(raw_bytes: bytes) -> str:
        return hashlib.sha256(raw_bytes).hexdigest()

    @staticmethod
    def audit_exif_metadata(image: Image.Image) -> dict:
        suspicious_tags = []
        editing_tools = ["photoshop", "gimp", "canva", "picsart", "coreldraw", "lightroom", "snapseed"]
        try:
            info = image.getexif()
            if info:
                for tag_id, value in info.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    val_str = str(value).lower()
                    for tool in editing_tools:
                        if tool in val_str:
                            suspicious_tags.append(f"Editor trace in {tag_name}: {tool.upper()}")
        except Exception:
            pass
        return {"has_exif": bool(suspicious_tags), "software_traces": suspicious_tags}

    @staticmethod
    def analyze_moire_frequency(img_cv: np.ndarray) -> dict:
        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
            gray = cv2.resize(gray, (512, 512))
            dft = np.fft.fft2(gray)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-9)
            
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            magnitude_spectrum[crow-25:crow+25, ccol-25:ccol+25] = 0
            
            papr = float(np.percentile(magnitude_spectrum, 99.8) / (np.mean(magnitude_spectrum) + 1e-5))
            return {"papr_score": round(papr, 3), "is_screen_recapture": bool(papr > 3.90)}
        except Exception:
            return {"papr_score": 1.0, "is_screen_recapture": False}

    @staticmethod
    def localize_tampering_multisignal(orig_img: Image.Image, img_cv: np.ndarray) -> dict:
        rgb_pil = orig_img.convert("RGB")
        buffered = io.BytesIO()
        rgb_pil.save(buffered, format="JPEG", quality=90)
        buffered.seek(0)
        resaved = Image.open(buffered)

        ela_im = ImageChops.difference(rgb_pil, resaved)
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1.0
        ela_enhanced = ImageEnhance.Brightness(ela_im).enhance(scale)
        ela_cv = cv2.cvtColor(np.array(ela_enhanced), cv2.COLOR_RGB2BGR)
        gray_ela = cv2.cvtColor(ela_cv, cv2.COLOR_BGR2GRAY)

        orig_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        doc_mask = orig_gray > 40
        valid_ela = gray_ela[doc_mask] if np.sum(doc_mask) > 500 else gray_ela
        mean_val = float(np.mean(valid_ela))
        std_val = float(np.std(valid_ela))

        # Precision threshold clamped to catch subtle marks without firing on clean cards
        dynamic_thresh = max(48, min(int(mean_val + (1.8 * std_val)), 66))

        blur_ela = cv2.GaussianBlur(gray_ela, (3, 3), 0)
        _, thresh = cv2.threshold(blur_ela, dynamic_thresh, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated_cv = img_cv.copy() if len(img_cv.shape) == 3 else cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        h_img, w_img = gray_ela.shape
        total_area = h_img * w_img

        suspicious_regions = []
        box_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Area floor 40px catches small scribbles, ink marks, and edited numbers
            if 40 < area < (total_area * 0.12):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / max(h, 1)

                if aspect > 5.5 and h < 12:
                    continue

                cv2.rectangle(annotated_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated_cv, "TAMPER", (x, max(y - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

                suspicious_regions.append({
                    "region_id": f"REG_{box_count + 1}",
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "anomaly_score": 0.85
                })
                box_count += 1

        return {
            "ela_enhanced": ela_enhanced,
            "annotated_cv": annotated_cv,
            "ela_variance": round(std_val, 2),
            "tamper_regions": suspicious_regions,
            "box_count": box_count
        }


class RiskFusionEngine:
    @classmethod
    def evaluate(cls, quality: dict, deterministic: dict, forensics: dict, metadata: dict, cross_field: dict) -> dict:
        risk_accum = 0.0
        explanations = []
        boxes = forensics.get("box_count", 0)

        # Smart Gating
        if not quality.get("passed", False) and boxes == 0:
            return {
                "verdict": "ABSTAIN: INSUFFICIENT EVIDENCE",
                "risk_score": 0,
                "risk_level": "UNDETERMINED",
                "confidence": 0.25,
                "breakdown": [f"Screening halted: {quality.get('abstain_reason', 'Low quality buffer')}"],
                "recommendation": "RE-ACQUIRE DOCUMENT UNDER PROPER ILLUMINATION WITHOUT MOTION BLUR"
            }

        # Tampering signals
        if boxes > 0:
            assigned = min(45 + (boxes * 15), 85)
            risk_accum += assigned
            explanations.append(f"+{assigned} Spatial pixel tampering localized in {boxes} distinct region(s).")
        elif forensics.get("ela_variance", 0) > 30.0:
            risk_accum += 10
            explanations.append("+10 Compression variance indicates potential re-saving artifacts.")

        mrz = deterministic.get("mrz", {})
        if mrz.get("is_mrz_detected") and not mrz.get("all_checks_passed"):
            risk_accum += 30
            explanations.append("+30 Check digit verification mismatch.")

        moire = forensics.get("moire", {})
        if moire.get("is_screen_recapture"):
            risk_accum += 25
            explanations.append(f"+25 Optical screen grid frequencies detected (PAPR: {moire.get('papr_score')}).")

        traces = metadata.get("software_traces", [])
        if traces:
            risk_accum += 20
            explanations.append(f"+20 Image editing software signature detected in metadata.")

        final_risk = int(min(max(risk_accum, 4), 98))
        level = "HIGH" if final_risk >= 50 else ("MEDIUM" if final_risk >= 25 else "LOW")
        verdict = "SUSPICIOUS / POTENTIAL FORGERY" if final_risk >= 50 else ("ANOMALIES DETECTED" if final_risk >= 25 else "AUTHENTIC / UNALTERED")
        rec = "MANDATORY INVESTIGATOR REVIEW" if final_risk >= 50 else ("SECONDARY SCREENING" if final_risk >= 25 else "PROCEED WITH STANDARD PROCESSING")

        return {
            "verdict": verdict,
            "risk_score": final_risk,
            "risk_level": level,
            "confidence": 0.88 if quality.get("passed", False) else 0.65,
            "breakdown": explanations if explanations else ["All structural, cryptographic, and spatial signals within normal baseline tolerances."],
            "recommendation": rec
        }
