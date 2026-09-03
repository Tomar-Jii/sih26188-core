import io
import uuid
import base64
import cv2
import numpy as np
from datetime import datetime, timezone
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from forensics import (
    ICAO9303Validator,
    DocumentQualityEngine,
    MultiSignalForensics,
    RiskFusionEngine
)

APP_VERSION = "4.2.2-resilient"
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB limit for high-res camera captures
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp", 
    "image/bmp", "image/pjpeg", "image/x-png", "application/octet-stream"
}

app = FastAPI(
    title="AegisID - Defense Grade Forensic Screener (SSB/MHA)",
    version=APP_VERSION
)
templates = Jinja2Templates(directory="templates")

CASE_REGISTRY: dict = {}

def mat_to_base64(mat: np.ndarray) -> str:
    if mat is None or mat.size == 0:
        return ""
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
async def execute_complete_audit(
    file: UploadFile = File(...),
    case_id: str = Form(None),
    mrz_raw_input: str = Form("")
):
    # 1. Flexible Content-Type validation with fallback
    c_type = (file.content_type or "").lower()
    if c_type and c_type not in ALLOWED_MIME_TYPES and not c_type.startswith("image/"):
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={"error": "UnsupportedMediaType", "detail": f"Uploaded type '{c_type}' is not a valid image."}
        )

    # 2. File size bounds
    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "EmptyFilePayload", "detail": "Uploaded file is 0 bytes."}
        )
    if len(raw_bytes) > MAX_FILE_SIZE:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"error": "PayloadTooLarge", "detail": "File exceeds 25 MB limit."}
        )

    # 3. Safe image decoding
    try:
        orig_pil = Image.open(io.BytesIO(raw_bytes))
        orig_pil.load()
        rgb_pil = orig_pil.convert("RGB")
        img_cv = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_cv is None or img_cv.size == 0:
            raise ValueError("Corrupted raster")
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "UnprocessableImage", "detail": f"Cannot decode image: {str(e)}"}
        )

    # 4. Forensic execution pipeline
    try:
        sha256_hash = MultiSignalForensics.compute_sha256(raw_bytes)
        timestamp_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        quality_eval = DocumentQualityEngine.assess_quality(img_cv)
        metadata_eval = MultiSignalForensics.audit_exif_metadata(orig_pil)
        moire_eval = MultiSignalForensics.analyze_moire_frequency(img_cv)
        forensics_eval = MultiSignalForensics.localize_tampering_multisignal(rgb_pil, img_cv)
        mrz_eval = ICAO9303Validator.parse_mrz(mrz_raw_input)

        deterministic_pkg = {"mrz": mrz_eval}
        cross_field = {"conflicts": []}
        if mrz_eval.get("is_mrz_detected") and not mrz_eval.get("all_checks_passed"):
            cross_field["conflicts"].append("MRZ check digit validation failed.")

        fusion_input_forensics = {
            "box_count": forensics_eval.get("box_count", 0),
            "ela_variance": forensics_eval.get("ela_variance", 0.0),
            "moire": moire_eval
        }
        
        fusion_decision = RiskFusionEngine.evaluate(
            quality=quality_eval,
            deterministic=deterministic_pkg,
            forensics=fusion_input_forensics,
            metadata=metadata_eval,
            cross_field=cross_field
        )

        q_metrics = quality_eval.get("metrics", {})
        blur_txt = q_metrics.get("blur_status", "Normal")
        res_txt = q_metrics.get("resolution", "Standard")

        timeline = [
            {"time": timestamp_iso, "event": "Document Ingested & SHA-256 Registered", "status": "VERIFIED"},
            {"time": timestamp_iso, "event": f"Quality Gate: {blur_txt} ({res_txt})", "status": "PASS" if quality_eval.get("passed") else "FLAGGED"},
            {"time": timestamp_iso, "event": f"Optical Frequency Moiré: PAPR {moire_eval.get('papr_score', 1.0)}", "status": "SUSPICIOUS" if moire_eval.get("is_screen_recapture") else "PASS"},
            {"time": timestamp_iso, "event": f"Pixel Compression Analysis: {forensics_eval.get('box_count', 0)} Anomalous Region(s)", "status": "FLAGGED" if forensics_eval.get("box_count", 0) > 0 else "PASS"},
            {"time": timestamp_iso, "event": f"ICAO 9303 MRZ Engine: {'Valid' if mrz_eval.get('all_checks_passed') else ('Failed' if mrz_eval.get('is_mrz_detected') else 'Standby')}", "status": "PASS" if mrz_eval.get('all_checks_passed') else "STANDBY"},
            {"time": timestamp_iso, "event": f"Risk Engine: Score {fusion_decision.get('risk_score', 0)}/100 ({fusion_decision.get('risk_level', 'LOW')})", "status": "COMPLETED"}
        ]

        assigned_case = case_id if case_id else f"AEG-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

        result_payload = {
            "version": APP_VERSION,
            "case_id": assigned_case,
            "timestamp": timestamp_iso,
            "sha256": sha256_hash,
            "quality": quality_eval,
            "deterministic": deterministic_pkg,
            "forensics": {
                "box_count": forensics_eval.get("box_count", 0),
                "ela_variance": forensics_eval.get("ela_variance", 0.0),
                "regions": forensics_eval.get("tamper_regions", []),
                "moire": moire_eval
            },
            "metadata": metadata_eval,
            "cross_field": cross_field,
            "risk": fusion_decision,
            "timeline": timeline,
            "images": {
                "orig_b64": f"data:image/png;base64,{pil_to_base64(rgb_pil)}",
                "annotated_b64": f"data:image/png;base64,{mat_to_base64(forensics_eval.get('annotated_cv'))}",
                "ela_b64": f"data:image/png;base64,{pil_to_base64(forensics_eval.get('ela_enhanced'))}"
            }
        }

        CASE_REGISTRY[assigned_case] = result_payload
        return result_payload

    except Exception as err:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "InternalPipelineError", "detail": f"Forensic engine exception: {str(err)}"}
        )

@app.get("/api/cases/{case_id}")
async def fetch_case_record(case_id: str):
    record = CASE_REGISTRY.get(case_id)
    if not record:
        raise HTTPException(status_code=404, detail="Case identifier not found.")
    return record
