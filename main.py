import io
import base64
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from forensics import DocumentForensicSuite, VerhoeffAlgorithm

app = FastAPI(title="AegisID - Defense Grade Forensic Screener")
templates = Jinja2Templates(directory="templates")
forensic_suite = DocumentForensicSuite()

def mat_to_base64(mat: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', mat)
    return base64.b64encode(buffer).decode('utf-8')

def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/audit")
async def audit_document(
    file: UploadFile = File(...),
    id_number: str = Form("")
):
    raw_bytes = await file.read()
    sha256_hash = forensic_suite.compute_sha256(raw_bytes)
    
    orig_pil = Image.open(io.BytesIO(raw_bytes))
    img_cv = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 1. GATEKEEPER CHECK: Is it actually an ID Card?
    structure_check = forensic_suite.validate_id_document_structure(img_cv)

    # Core Inspections
    exif_results = forensic_suite.audit_exif_metadata(orig_pil)
    moire_results = forensic_suite.analyze_moire_frequency(img_cv)
    ela_img, annotated_cv, ela_variance, tamper_boxes = forensic_suite.localize_tampering(orig_pil)
    qr_results = forensic_suite.audit_qr_code(img_cv)

    risk_score = 0
    flags = []

    # If it fails the ID Gatekeeper Check
    if not structure_check["is_valid_id"]:
        risk_score = 100
        flags.append(f"[GATEWAY REJECT] {structure_check['reason']}")
        verdict = "NOT AN ID DOCUMENT"
    else:
        # Verhoeff Aadhaar Math Check
        verhoeff_status = "N/A"
        clean_id = id_number.replace(" ", "").strip()
        if clean_id:
            if len(clean_id) == 12 and clean_id.isdigit():
                is_valid_aadhaar = VerhoeffAlgorithm.validate(clean_id)
                if is_valid_aadhaar:
                    verhoeff_status = "PASSED (Valid Dihedral D5)"
                else:
                    verhoeff_status = "FAILED (Mathematical Forgery)"
                    risk_score += 50
                    flags.append(f"Aadhaar Number [{clean_id}] failed Verhoeff Dihedral Checksum: Digit sequence is fabricated.")
            else:
                verhoeff_status = "INVALID FORMAT (Expected 12 Digits)"
                risk_score += 20
                flags.append("Specified ID number format does not comply with 12-digit standard.")

        if exif_results["software_traces"]:
            risk_score += 45
            flags.extend(exif_results["software_traces"])

        if moire_results["is_screen_recapture"]:
            risk_score += 35
            flags.append(f"Screen Optical Moiré detected (Score: {moire_results['papr_score']}) - Recaptured from electronic display.")

        if tamper_boxes > 0:
            risk_score += min(tamper_boxes * 15, 45)
            flags.append(f"Pixel Splicing Localized: {tamper_boxes} anomalous regions identified.")
        elif ela_variance > 32.0:
            risk_score += 20
            flags.append("High compression level variance detected across surface.")

        if not qr_results["detected"]:
            flags.append("Cryptographic QR Code missing or unreadable on document.")

        risk_score = min(max(risk_score, 4), 99)
        verdict = "FORGERY DETECTED" if risk_score >= 50 else ("SUSPICIOUS" if risk_score >= 25 else "GENUINE / AUTHENTIC")

    verhoeff_status = "SKIPPED (Invalid Doc)" if not structure_check["is_valid_id"] else locals().get('verhoeff_status', 'N/A')

    return {
        "verdict": verdict,
        "risk_score": risk_score,
        "sha256": sha256_hash,
        "flags": flags,
        "is_valid_id": structure_check["is_valid_id"],
        "verhoeff_status": verhoeff_status,
        "metrics": {
            "has_portrait": structure_check["has_face"],
            "ela_variance": ela_variance,
            "tamper_regions": tamper_boxes,
            "moire_papr": moire_results["papr_score"],
            "screen_spoof": moire_results["is_screen_recapture"],
            "qr_status": qr_results["status"]
        },
        "images": {
            "orig_b64": f"data:image/png;base64,{pil_to_base64(orig_pil)}",
            "ela_heatmap": f"data:image/png;base64,{pil_to_base64(ela_img)}",
            "annotated_tamper": f"data:image/png;base64,{mat_to_base64(annotated_cv)}"
        }
    }
