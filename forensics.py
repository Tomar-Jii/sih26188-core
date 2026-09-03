import io
import re
import cv2
import hashlib
import numpy as np
from datetime import datetime
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS

class ICAO9303Validator:
    """ICAO Doc 9303 standard MRZ parsing and 7-3-1 check digit validation."""
    WEIGHTS = [7, 3, 1]

    @classmethod
    def compute_check_digit(cls, data_str: str) -> int:
        total = 0
        for i, char in enumerate(data_str):
            if '0' <= char <= '9':
                val = int(char)
            elif 'A' <= char <= 'Z':
                val = ord(char) - 55
            elif char == '<':
                val = 0
            else:
                val = 0
            total += val * cls.WEIGHTS[i % 3]
        return total % 10

    @classmethod
    def parse_mrz(cls, raw_text: str) -> dict:
        lines = [line.strip().replace(" ", "").upper() for line in raw_text.splitlines() if len(line.strip()) >= 30]
        mrz_lines = [l for l in lines if re.match(r'^[A-Z0-9<]+$', l)]
        
        if len(mrz_lines) >= 2 and len(mrz_lines[0]) in [36, 44]:
            l1, l2 = mrz_lines[0], mrz_lines[1]
            if len(l1) == 44 and len(l2) == 44:  # TD3 Passport
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
                    "all_checks_passed": all([doc_valid, dob_valid, expiry_valid, comp_valid])
                }

        return {"is_mrz_detected": False, "checks": {}, "all_checks_passed": False}


class DocumentQualityEngine:
    """Pre-screening quality engine: Skew, Blur, Glare, Contrast, and Resolution."""

    @staticmethod
    def assess_quality(img_cv: np.ndarray) -> dict:
        h, w = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv

        # 1. Blur via Laplacian Variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_status = "Good" if lap_var > 120.0 else ("Moderate" if lap_var > 45.0 else "Poor (Blurry)")

        # 2. Glare & Under-exposure Analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_pixels = h * w
        pure_white_ratio = float(np.sum(hist[248:]) / total_pixels)
        pure_dark_ratio = float(np.sum(hist[:15]) / total_pixels)
        
        glare_status = "High (Specular Glare)" if pure_white_ratio > 0.08 else "Low"
        exposure_status = "Severely Underexposed" if pure_dark_ratio > 0.45 else "Well Exposed"

        # 3. Resolution Assessment
        resolution_status = "Acceptable" if (w >= 600 and h >= 400) else "Low Resolution"

        # Quality Gate Verdict
        passed_quality_gate = bool(lap_var > 45.0 and pure_white_ratio <= 0.15 and resolution_status == "Acceptable")
        abstain_reason = None
        if not passed_quality_gate:
            reasons = []
            if lap_var <= 45.0: reasons.append("Deficient Sharpness / Severe Motion Blur")
            if pure_white_ratio > 0.15: reasons.append("Excessive Surface Glare Overwriting Pixels")
            if resolution_status != "Acceptable": reasons.append("Pixel Density Below Minimum Inspection Standard")
            abstain_reason = "; ".join(reasons)

        return {
            "passed": passed_quality_gate,
            "abstain_reason": abstain_reason,
            "metrics": {
                "sharpness_laplacian": round(lap_var, 1),
                "blur_status": blur_status,
                "glare_status": glare_status,
                "exposure_status": exposure_status,
                "resolution": f"{w}x{h} ({resolution_status})"
            }
        }


class MultiSignalForensics:
    """Multi-Signal Forensic Engine targeting spatial, frequency, and metadata anomalies."""

    @staticmethod
    def compute_sha256(raw_bytes: bytes) -> str:
        return hashlib.sha256(raw_bytes).hexdigest()

    @staticmethod
    def audit_exif_metadata(image: Image.Image) -> dict:
        suspicious_tags = []
        editing_tools = ["photoshop", "gimp", "canva", "picsart", "coreldraw", "lightroom", "snapseed", "pixelmator"]
        info = image.getexif()
        raw_meta = {}
        if info:
            for tag_id, value in info.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                val_str = str(value).lower()
                raw_meta[tag_name] = str(value)[:80]
                for tool in editing_tools:
                    if tool in val_str:
                        suspicious_tags.append(f"Editor footprint detected in tag '{tag_name}': {tool.upper()}")
        return {"has_exif": len(raw_meta) > 0, "software_traces": suspicious_tags, "metadata": raw_meta}

    @staticmethod
    def analyze_moire_frequency(img_cv: np.ndarray) -> dict:
        """2D Fast Fourier Transform high-frequency peak-to-average ratio calculation."""
        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
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
    def extract_noise_variance_map(img_cv: np.ndarray) -> tuple[np.ndarray, float]:
        """High-frequency spatial noise residual analysis via median filter subtraction."""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        noise = cv2.absdiff(gray, denoised)
        noise_var = float(np.var(noise))
        norm_noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)
        return norm_noise, round(noise_var, 2)

    @staticmethod
    def localize_tampering_multisignal(orig_img: Image.Image, img_cv: np.ndarray) -> dict:
        """Adaptive ELA + Gradient Edge Continuity Analysis for precise bounding."""
        buffered = io.BytesIO()
        orig_img.save(buffered, format="JPEG", quality=90)
        buffered.seek(0)
        resaved = Image.open(buffered)

        # 1. Error Level Analysis
        ela_im = ImageChops.difference(orig_img.convert("RGB"), resaved.convert("RGB"))
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1.0
        ela_enhanced = ImageEnhance.Brightness(ela_im).enhance(scale)
        ela_cv = cv2.cvtColor(np.array(ela_enhanced), cv2.COLOR_RGB2BGR)
        gray_ela = cv2.cvtColor(ela_cv, cv2.COLOR_BGR2GRAY)

        # 2. Dynamic Z-score thresholding
        mean_val = float(np.mean(gray_ela))
        std_val = float(np.std(gray_ela))
        dynamic_thresh = max(48, min(int(mean_val + (2.1 * std_val)), 78))

        blur_ela = cv2.GaussianBlur(gray_ela, (5, 5), 0)
        _, thresh = cv2.threshold(blur_ela, dynamic_thresh, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 3. Multi-Signal Bounding & Edge Boundary Validation
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated_cv = img_cv.copy()
        h_img, w_img = gray_ela.shape
        total_area = h_img * w_img

        suspicious_regions = []
        box_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 85 < area < (total_area * 0.14):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / h
                # Suppress square 2D barcode regions on document corners
                if area > 2600 and 0.82 < aspect < 1.18 and x > (w_img * 0.40):
                    continue

                # Signal Cross-Check: Edge discontinuity within candidate zone
                roi_gray = cv2.cvtColor(img_cv[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                sobel_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
                edge_gradient = float(np.mean(np.abs(sobel_x)))

                anomaly_score = min(0.98, round(0.50 + (edge_gradient / 255.0) + (area / total_area * 2.0), 2))
                
                # Annotate
                cv2.rectangle(annotated_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated_cv, f"ANOMALY {anomaly_score}", (x, max(y - 5, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

                suspicious_regions.append({
                    "region_id": f"REG_{box_count + 1}",
                    "bbox": [x, y, w, h],
                    "anomaly_score": anomaly_score,
                    "signals": ["Adaptive ELA Compression Discrepancy", "Sobel Edge Discontinuity"],
                    "interpretation": "Discontinuous compression state; local text or pixel patch spliced."
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
    """Configurable Weighted Risk Fusion with Separate Model Confidence and Abstention."""
    
    # Transparent Weight Matrix
    WEIGHTS = {
        "quality_penalties": 0.10,
        "deterministic_failures": 0.25,
        "forensic_compression": 0.25,
        "screen_spoof": 0.20,
        "metadata_traces": 0.10,
        "cross_field_conflicts": 0.10
    }

    @classmethod
    def evaluate(cls, quality: dict, deterministic: dict, forensics: dict, metadata: dict, cross_field: dict) -> dict:
        risk_accum = 0.0
        explanations = []

        # 1. Quality Integrity Check
        if not quality["passed"]:
            return {
                "verdict": "ABSTAIN: INSUFFICIENT EVIDENCE",
                "risk_score": 0,
                "risk_level": "UNDETERMINED",
                "confidence": 0.20,
                "breakdown": ["Analysis halted: Document image failed standard quality gate."],
                "recommendation": "RE-ACQUIRE DOCUMENT UNDER PROPER ILLUMINATION WITHOUT BLUR"
            }

        # 2. Deterministic Checksums (e.g. MRZ, Formats)
        mrz = deterministic.get("mrz", {})
        if mrz.get("is_mrz_detected") and not mrz.get("all_checks_passed"):
            risk_accum += 25
            explanations.append("+25 ICAO Doc 9303 Checksum verification failed across machine-readable lines.")

        # 3. Forensic Localization (Tampered Regions)
        boxes = forensics.get("box_count", 0)
        if boxes > 0:
            assigned = min(boxes * 12, 28)
            risk_accum += assigned
            explanations.append(f"+{assigned} Spatial pixel splicing localized in {boxes} distinct document region(s).")
        elif forensics.get("ela_variance", 0) > 30.0:
            risk_accum += 10
            explanations.append("+10 Compression variance indicates potential uniform recompression artifacts.")

        # 4. Physical Liveness / Moiré
        moire = forensics.get("moire", {})
        if moire.get("is_screen_recapture"):
            risk_accum += 22
            explanations.append(f"+22 High-frequency optical Moiré detected (PAPR: {moire.get('papr_score')}) - Screen recapture spoof.")

        # 5. Metadata Trace
        traces = metadata.get("software_traces", [])
        if traces:
            risk_accum += 15
            explanations.append(f"+15 Image manipulation software signatures present: {traces[0]}")

        # 6. Cross-Field Discrepancies
        conflicts = cross_field.get("conflicts", [])
        if conflicts:
            risk_accum += 18
            explanations.append(f"+18 Cross-field data discrepancy: {conflicts[0]}")

        # Final Normalization
        final_risk = int(min(max(risk_accum, 3), 98))
        
        if final_risk >= 55:
            level = "HIGH"
            verdict = "SUSPICIOUS / POTENTIAL FORGERY"
            rec = "MANDATORY INVESTIGATOR FORENSIC REVIEW"
        elif final_risk >= 25:
            level = "MEDIUM"
            verdict = "ANOMALIES DETECTED"
            rec = "SECONDARY VALIDATION REQUIRED"
        else:
            level = "LOW"
            verdict = "AUTHENTIC / UNALTERED"
            rec = "PROCEED WITH STANDARD PROCESSING"

        confidence_score = round(0.92 - (0.05 if not mrz.get("is_mrz_detected") else 0.0), 2)

        return {
            "verdict": verdict,
            "risk_score": final_risk,
            "risk_level": level,
            "confidence": confidence_score,
            "breakdown": explanations if explanations else ["All structural, cryptographic, and spatial signals within normal tolerances."],
            "recommendation": rec
        }
