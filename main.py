import io
import uuid
import base64
import cv2
import numpy as np
from datetime import datetime, timezone
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from forensics import (
    ICAO9303Validator,
    DocumentQualityEngine,
    MultiSignalForensics,
    RiskFusionEngine
)

APP_VERSION = "1.0.0-phase1"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_UPLOAD_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}

app = FastAPI(title="AegisID - Forensic Screening Core")
templates = Jinja2Templates(directory="templates")

# In-Memory Case & Evidence Registry (Ephemeral & Private)
CASE_REGISTRY: dict = {}

def mat_to_base64(mat: np.ndarray) -> str:
    if mat is None:
        return ""
    _, buffer = cv2.imencode('.png', mat)
    return base64.b64encode(buffer).decode('utf-8')

def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def _error_response(status_code: int, code: str, message: str, case_id: str = "") -> JSONResponse:
    payload = {
        "error": {"code": code, "message": message},
        "system_state": "ERROR",
        "backend_version": APP_VERSION,
    }
    if case_id:
        payload["case_id"] = case_id
    return JSONResponse(status_code=status_code, content=payload)


def _decode_image_safely(raw_bytes: bytes, uploaded_content_type: str) -> tuple[Image.Image, np.ndarray]:
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=("EMPTY_UPLOAD", "Uploaded file is empty."))
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=("FILE_TOO_LARGE", f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit."),
        )

    if uploaded_content_type and uploaded_content_type not in ALLOWED_UPLOAD_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=("UNSUPPORTED_MEDIA_TYPE", "Unsupported file type. Please upload a standard image format."),
        )

    try:
        tmp = Image.open(io.BytesIO(raw_bytes))
        image_format = (tmp.format or "").upper()
        tmp.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=("MALFORMED_IMAGE", "Uploaded file is not a valid image or is corrupted."),
        )

    if image_format and image_format not in ALLOWED_PIL_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=("UNSUPPORTED_IMAGE_FORMAT", "Unsupported image encoding."),
        )

    try:
        orig_pil = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=("MALFORMED_IMAGE", "Failed to safely decode image content."),
        )

    img_cv = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if img_cv is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=("MALFORMED_IMAGE", "Failed to decode image pixels."),
        )

    if len(img_cv.shape) == 2:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    elif len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
    elif not (len(img_cv.shape) == 3 and img_cv.shape[2] == 3):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=("UNSUPPORTED_IMAGE_CHANNELS", "Unsupported image channel layout."),
        )

    return orig_pil, img_cv

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/audit")
async def execute_complete_audit(
    file: UploadFile = File(...),
    case_id: str = Form(None),
    doc_type_hint: str = Form("AUTO_DETECT"),
    mrz_raw_input: str = Form("")
):
    """
    Primary Unified Screening Endpoint executing the complete 8-stage forensic pipeline.
    """
    assigned_case = case_id if case_id else f"AEG-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    timestamp_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    CASE_REGISTRY[assigned_case] = {
        "case_id": assigned_case,
        "system_state": "ANALYZING",
        "timestamp": timestamp_iso,
        "backend_version": APP_VERSION,
    }

    try:
        raw_bytes = await file.read()
        sha256_hash = MultiSignalForensics.compute_sha256(raw_bytes)
        orig_pil, img_cv = _decode_image_safely(raw_bytes, (file.content_type or "").lower())

        # 1. Quality Assessment Gate
        quality_eval = DocumentQualityEngine.assess_quality(img_cv)

        # 2. Metadata Audit
        metadata_eval = MultiSignalForensics.audit_exif_metadata(orig_pil)

        # 3. Frequency Moiré Screen Recapture Analysis
        moire_eval = MultiSignalForensics.analyze_moire_frequency(img_cv)

        # 4. Multi-Signal Spatial Tampering & Boundary Localization
        forensics_eval = MultiSignalForensics.localize_tampering_multisignal(orig_pil, img_cv)

        # 5. Deterministic Validation (MRZ ICAO 9303)
        mrz_eval = ICAO9303Validator.parse_mrz(mrz_raw_input)

        # 6. Cross-Field Consistency Checks
        cross_field = {"conflicts": []}
        if mrz_eval["is_mrz_detected"] and not mrz_eval["all_checks_passed"]:
            cross_field["conflicts"].append("MRZ check digit validation failed. Potential string modification.")

        # 7. Risk Fusion & Explainable Synthesis
        deterministic_pkg = {"mrz": mrz_eval}
        fusion_input_forensics = {
            "box_count": forensics_eval["box_count"],
            "ela_variance": forensics_eval["ela_variance"],
            "moire": moire_eval
        }
        
        fusion_decision = RiskFusionEngine.evaluate(
            quality=quality_eval,
            deterministic=deterministic_pkg,
            forensics=fusion_input_forensics,
            metadata=metadata_eval,
            cross_field=cross_field
        )

        # Build Audit Trail Timeline
        timeline = [
            {"time": timestamp_iso, "event": "Document Ingestion & Cryptographic Hashing Completed", "status": "VERIFIED"},
            {"time": timestamp_iso, "event": f"Pre-Screening Quality Gate: {quality_eval['metrics']['blur_status']} Sharpness", "status": "PASS" if quality_eval["passed"] else "REJECT"},
            {"time": timestamp_iso, "event": f"Optical Frequency Spectrum Analysis: PAPR {moire_eval['papr_score']}", "status": "SUSPICIOUS" if moire_eval["is_screen_recapture"] else "PASS"},
            {"time": timestamp_iso, "event": f"Pixel Compression Analysis: {forensics_eval['box_count']} Discrepant Region(s)", "status": "FLAGGED" if forensics_eval["box_count"] > 0 else "PASS"},
            {"time": timestamp_iso, "event": f"Deterministic MRZ Validation: {'Detected & Verified' if mrz_eval['all_checks_passed'] else ('Detected with Checksum Errors' if mrz_eval['is_mrz_detected'] else 'No MRZ Pattern Presented')}", "status": "PASS" if mrz_eval["all_checks_passed"] else "STANDBY"},
            {"time": timestamp_iso, "event": f"Risk Engine Synthesis: Score {fusion_decision['risk_score']}/100 ({fusion_decision['risk_level']})", "status": "COMPLETED"}
        ]

        result_payload = {
            "case_id": assigned_case,
            "system_state": "COMPLETED",
            "backend_version": APP_VERSION,
            "timestamp": timestamp_iso,
            "sha256": sha256_hash,
            "quality": quality_eval,
            "deterministic": deterministic_pkg,
            "forensics": {
                "box_count": forensics_eval["box_count"],
                "ela_variance": forensics_eval["ela_variance"],
                "regions": forensics_eval["tamper_regions"],
                "moire": moire_eval
            },
            "metadata": metadata_eval,
            "cross_field": cross_field,
            "risk": fusion_decision,
            "timeline": timeline,
            "images": {
                "orig_b64": f"data:image/png;base64,{pil_to_base64(orig_pil)}",
                "annotated_b64": f"data:image/png;base64,{mat_to_base64(forensics_eval['annotated_cv'])}",
                "ela_b64": f"data:image/png;base64,{pil_to_base64(forensics_eval['ela_enhanced'])}"
            }
        }

        # Store in registry
        CASE_REGISTRY[assigned_case] = result_payload
        return result_payload

    except HTTPException as exc:
        err_code, err_msg = ("REQUEST_ERROR", "Unable to process request.")
        if isinstance(exc.detail, tuple) and len(exc.detail) == 2:
            err_code, err_msg = exc.detail
        CASE_REGISTRY[assigned_case] = {
            "case_id": assigned_case,
            "system_state": "ERROR",
            "backend_version": APP_VERSION,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            "error": {"code": err_code, "message": err_msg},
        }
        return _error_response(exc.status_code, err_code, err_msg, case_id=assigned_case)
    except Exception:
        CASE_REGISTRY[assigned_case] = {
            "case_id": assigned_case,
            "system_state": "ERROR",
            "backend_version": APP_VERSION,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            "error": {"code": "INSPECTION_PIPELINE_ERROR", "message": "Unexpected internal error during screening."},
        }
        return _error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INSPECTION_PIPELINE_ERROR",
            message="Unexpected internal error during screening.",
            case_id=assigned_case,
        )

@app.get("/api/cases/{case_id}")
async def fetch_case_record(case_id: str):
    record = CASE_REGISTRY.get(case_id)
    if not record:
        raise HTTPException(status_code=404, detail="Case record not found in ephemeral registry.")
    return record
