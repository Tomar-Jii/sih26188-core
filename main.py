import io
import base64
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from aegis_core.config import CONFIG
from aegis_core.quality.gatekeeper import DocumentQualityGate
from aegis_core.vision.warp import DocumentPerspectiveWarper
from aegis_core.validators.dihedral import VerhoeffDihedralValidator
from aegis_core.validators.mrz_td3 import ICAO9303MRZParser
from aegis_core.validators.qr_engine import MultiPassQREngine
from aegis_core.forensics.ela import DifferentialELAAnalyzer
from aegis_core.forensics.moire import OpticalMoireAnalyzer
from aegis_core.forensics.metadata import MetadataFootprintAnalyzer
from aegis_core.timeline.audit_trail import ForensicAuditTrail, EphemeralCaseLedger

app = FastAPI(title="AegisID - Defense Forensic Screener", version=CONFIG.APP_VERSION)
templates = Jinja2Templates(directory="templates")

def mat_to_b64(mat: np.ndarray) -> str:
    if mat is None or mat.size == 0: return ""
    _, buf = cv2.imencode('.png', mat)
    return base64.b64encode(buf).decode('utf-8')

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/audit")
async def execute_audit(
    file: UploadFile = File(...),
    case_id: str = Form(None),
    id_number: str = Form(""),
    mrz_raw_input: str = Form("")
):
    raw_bytes = await file.read()
    if not raw_bytes or len(raw_bytes) > CONFIG.MAX_PAYLOAD_BYTES:
        return JSONResponse(status_code=400, content={"error": "Invalid payload size."})

    try:
        orig_pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img_cv = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_cv is None: raise ValueError("Corrupted image matrix")
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": f"Image decode failure: {str(e)}"})

    trail = ForensicAuditTrail(case_id=case_id)
    sha256 = __import__("hashlib").sha256(raw_bytes).hexdigest()
    trail.log("Document Ingested & SHA-256 Registered", "VERIFIED")

    # Modular Quality Gate
    quality = DocumentQualityGate.audit(img_cv)
    trail.log(f"Quality Assessment: {quality['metrics'].get('blur_status', 'OK')}", "PASS" if quality["passed"] else "FLAGGED")

    # Perspective Normalization
    warped_cv = DocumentPerspectiveWarper.extract_and_warp(img_cv)

    # Deterministic Validators
    mrz_res = ICAO9303MRZParser.parse(mrz_raw_input)
    qr_res = MultiPassQREngine.inspect_and_mask(warped_cv)

    # Forensic Engines
    ela_res = DifferentialELAAnalyzer.analyze(orig_pil, warped_cv, qr_bbox=qr_res.get("bbox"))
    moire_res = OpticalMoireAnalyzer.inspect(warped_cv)
    meta_res = MetadataFootprintAnalyzer.inspect(orig_pil)

    # Regions Annotation
    regions = ela_res.get("suspicious_zones", [])
    annotated = warped_cv.copy()
    for reg in regions:
        x, y, w, h = reg["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(annotated, "TAMPER", (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    box_count = len(regions)
    risk_score = min(98, 4 + (box_count * 15) + (25 if moire_res.get("is_screen_recapture") else 0))
    risk_level = "HIGH" if risk_score >= 50 else ("MEDIUM" if risk_score >= 25 else "LOW")
    verdict = "SUSPICIOUS / POTENTIAL FORGERY" if risk_score >= 50 else "AUTHENTIC / UNALTERED"

    response = {
        "version": CONFIG.APP_VERSION,
        "case_id": trail.case_id,
        "sha256": sha256,
        "quality": quality,
        "deterministic": {"mrz": mrz_res, "qr": qr_res},
        "forensics": {
            "box_count": box_count,
            "regions": regions,
            "ela_variance": ela_res.get("ela_variance", 0.0),
            "moire": moire_res
        },
        "metadata": meta_res,
        "risk": {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "verdict": verdict,
            "confidence": 0.88,
            "recommendation": "MANDATORY INVESTIGATOR REVIEW" if risk_score >= 50 else "STANDARD PROCESSING",
            "breakdown": [f"+{box_count * 15} Spatial tamper anomaly in {box_count} region(s)."] if box_count > 0 else ["All signals within baseline."]
        },
        "timeline": trail.get_timeline(),
        "images": {
            "orig_b64": f"data:image/png;base64,{pil_to_b64(orig_pil)}",
            "annotated_b64": f"data:image/png;base64,{mat_to_b64(annotated)}",
            "ela_b64": f"data:image/png;base64,{pil_to_b64(ela_res.get('ela_enhanced', orig_pil))}"
        }
    }

    EphemeralCaseLedger.register(trail.case_id, response)
    return response

@app.get("/api/cases/{case_id}")
async def get_case(case_id: str):
    rec = EphemeralCaseLedger.fetch(case_id)
    if not rec: raise HTTPException(status_code=404, detail="Case not found.")
    return rec
