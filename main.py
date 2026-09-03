import io
import base64
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
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
    try:
        raw_bytes = await file.read()
        sha256_hash = forensic_suite.compute_sha256(raw_bytes)
        
        orig_pil = Image.open(io.BytesIO(raw_bytes))
        img_cv = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)

        # 1. Gatekeeper Check
        structure_check = forensic_suite.validate_id_document_structure(img_cv)

        # 2. Forensic Layers
        exif_results = forensic_suite.audit_exif_metadata(orig_pil)
        moire_results = forensic_suite.analyze_moire_frequency(img_cv)
        ela_img, annotated_cv, ela_variance, tamper_boxes = forensic_suite.localize_tampering(orig_pil)
        qr_results = forensic_suite.audit_qr_code(img_cv)

        risk_score = 0
        flags = []
        verhoeff_status = "Not Provided"

        if not structure_check["is_valid_id"]:
            risk_score = 100
            flags.append(f"[GATEWAY REJECT] {structure_check['reason']}")
            verdict = "NOT AN ID DOCUMENT"
            verhoeff_status = "SKIPPED"
        else:
            clean_id = id_number.replace(" ", "").strip()
            if clean_id:
                if len(clean_id) == 12 and clean_id.isdigit():
                    is_valid_aadhaar = VerhoeffAlgorithm.validate(clean_id)
                    if is_valid_aadhaar:
                        verhoeff_status = "PASSED (Valid Dihedral D5)"
                    else:
                        verhoeff_status = "FAILED (Mathematical Forgery)"
                        risk_score += 55
                        flags.append(f"Aadhaar Number [{clean_id}] failed Verhoeff Checksum: Digit sequence is fabricated.")
                else:
                    verhoeff_status = "INVALID FORMAT"
                    risk_score += 20
                    flags.append("Specified ID number format does not comply with 12-digit standard.")

            if tamper_boxes > 0:
                risk_score += min(tamper_boxes * 20, 60)
                flags.append(f"Pixel Splicing/Scribble Detected: {tamper_boxes} anomalous regions localized in red boxes.")
            elif ela_variance > 30.0:
                risk_score += 25
                flags.append("Compression level variance detected across document surface.")

            if exif_results["software_traces"]:
                risk_score += 40
                flags.extend(exif_results["software_traces"])

            if moire_results["is_screen_recapture"]:
                risk_score += 35
                flags.append(f"Screen Optical Moiré detected (PAPR: {moire_results['papr_score']}) - Screen photo, not physical card.")

            if not qr_results["detected"]:
                flags.append("No standard cryptographic QR code decoded.")

            risk_score = min(max(risk_score, 4), 99)
            verdict = "FORGERY DETECTED" if risk_score >= 50 else ("SUSPICIOUS" if risk_score >= 25 else "GENUINE / AUTHENTIC")

        return {
            "verdict": verdict,
            "risk_score": risk_score,
            "sha256": sha256_hash,
            "flags": flags,
            "is_valid_id": structure_check["is_valid_id"],
            "verhoeff_status": verhoeff_status,
            "metrics": {
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
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
