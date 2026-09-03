import io
import re
import uuid
import base64
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from forensics import (
    ICAO9303Validator,
    DocumentQualityEngine,
    MultiSignalForensics,
    RiskFusionEngine
)

app = FastAPI(title="AegisID - Defense Grade Forensic Screener (SSB/MHA)")
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
    try:
        raw_bytes = await file.read()
        sha256_hash = MultiSignalForensics.compute_sha256(raw_bytes)
        
        orig_pil = Image.open(io.BytesIO(raw_bytes))
        img_cv = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_cv is None:
            raise ValueError("Unable to decode input as a valid graphical document.")

        timestamp_iso = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

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

        assigned_case = case_id if case_id else f"AEG-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

        result_payload = {
            "case_id": assigned_case,
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

    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": "InspectionPipelineError", "details": str(exc)})

@app.get("/api/cases/{case_id}")
async def fetch_case_record(case_id: str):
    record = CASE_REGISTRY.get(case_id)
    if not record:
        raise HTTPException(status_code=404, detail="Case record not found in ephemeral registry.")
    return record
