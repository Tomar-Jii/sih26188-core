import io
import base64
import cv2
import hashlib
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from aegis_core.config import CONFIG
from aegis_core.quality.gatekeeper import DocumentQualityGate
from aegis_core.vision.warp import DocumentPerspectiveWarper
from aegis_core.vision.face_segment import BiometricPortraitAnalyzer
from aegis_core.validators.dihedral import VerhoeffDihedralValidator
from aegis_core.validators.mrz_td3 import ICAO9303MRZParser
from aegis_core.validators.qr_engine import MultiPassQREngine
from aegis_core.forensics.ela import DifferentialELAAnalyzer
from aegis_core.forensics.texture import TextureFlatnessAnalyzer
from aegis_core.forensics.moire import OpticalMoireAnalyzer
from aegis_core.forensics.metadata import MetadataFootprintAnalyzer
from aegis_core.fusion.cross_field import CrossFieldConsistencyEngine
from aegis_core.fusion.cluster import SpatialRegionMerger
from aegis_core.fusion.risk_engine import MultiSignalRiskEngine
from aegis_core.fusion.abstention import ConfidenceAbstentionGate
from aegis_core.timeline.audit_trail import ForensicAuditTrail, EphemeralCaseLedger
from aegis_core.reporting.bsa_dossier import BSADossierBuilder

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
    # 1. Ingestion Validation
    c_type = (file.content_type or "").lower()
    if c_type and c_type not in CONFIG.ALLOWED_MIME_TYPES and not c_type.startswith("image/"):
        return JSONResponse(status_code=415, content={"error": "UnsupportedMediaType", "detail": f"Allowed: {list(CONFIG.ALLOWED_MIME_TYPES)}"})

    raw_bytes = await file.read()
    if not raw_bytes or len(raw_bytes) > CONFIG.MAX_PAYLOAD_BYTES:
        return JSONResponse(status_code=400, content={"error": "InvalidPayload", "detail": f"File size must be >0 and <{CONFIG.MAX_PAYLOAD_BYTES // (1024*1024)}MB."})

    try:
        orig_pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img_cv = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_cv is None or img_cv.size == 0: raise ValueError("Corrupted pixel buffer")
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": "UnprocessableImage", "detail": str(e)})

    trail = ForensicAuditTrail(case_id=case_id)
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    trail.log("Document Ingested & Cryptographic Hash Registered", "VERIFIED")

    # 2. Phase 4 Quality Gate
    quality = DocumentQualityGate.audit(img_cv)
    q_status = quality["metrics"].get("blur_status", "Acceptable")
    trail.log(f"Pre-Screening Quality Gate: {q_status} Sharpness", "PASS" if quality["passed"] else "FLAGGED")

    # 3. Phase 3 Boundary Perspective Normalization
    warped_cv = DocumentPerspectiveWarper.extract_and_warp(img_cv)

    # 4. Phase 8 & 9 Deterministic Cryptographic Validators
    clean_id = id_number.replace(" ", "").strip()
    dihedral_valid = False
    if clean_id and len(clean_id) == 12 and clean_id.isdigit():
        dihedral_valid = VerhoeffDihedralValidator.validate(clean_id)
        trail.log(f"Dihedral D5 Checksum ({clean_id[:4]} **** {clean_id[-4:]}): {'PASS' if dihedral_valid else 'FAIL'}", "PASS" if dihedral_valid else "FAIL")

    mrz_res = ICAO9303MRZParser.parse(mrz_raw_input)
    if mrz_res["is_mrz_detected"]:
        trail.log(f"ICAO Doc 9303 MRZ Checksum: {'PASS' if mrz_res['all_checks_passed'] else 'FAIL'}", "PASS" if mrz_res['all_checks_passed'] else "FLAGGED")

    # 5. Phase 10 Multi-Pass QR Decoder & Masker
    qr_res = MultiPassQREngine.inspect_and_mask(warped_cv)

    # 6. Phase 7 Biometric Face Segmentation & Photo-Swap Audit
    face_res = BiometricPortraitAnalyzer.extract_and_audit(warped_cv)
    trail.log(f"Biometric Portrait: {'Extracted' if face_res['face_detected'] else 'No Face Detected'} (Swap Gradient: {face_res['swap_score']})",
              "FLAGGED" if face_res["anomaly_detected"] else "PASS")

    # 7. Phase 11, 14, 15, 16 Multi-Signal Spatial Scanners
    ela_res = DifferentialELAAnalyzer.analyze(orig_pil, warped_cv, qr_bbox=qr_res.get("bbox"))
    texture_res = TextureFlatnessAnalyzer.detect_digital_strokes(warped_cv, qr_bbox=qr_res.get("bbox"))
    moire_res = OpticalMoireAnalyzer.inspect(warped_cv)
    meta_res = MetadataFootprintAnalyzer.inspect(orig_pil)

    # 8. Phase 6 Spatial NMS Clustering
    raw_candidates = ela_res.get("suspicious_zones", []) + texture_res.get("tamper_zones", [])
    if face_res.get("tamper_zone"):
        raw_candidates.append(face_res["tamper_zone"])

    merged_regions = SpatialRegionMerger.merge_regions(raw_candidates, iou_thresh=0.18, max_dist=12)
    box_count = len(merged_regions)
    trail.log(f"Spatial Fusion (NMS): {box_count} Consolidated Tamper Zone(s)", "FLAGGED" if box_count > 0 else "PASS")

    # 9. Phase 19 Cross-Field Coherence Matrix
    cross_field_res = CrossFieldConsistencyEngine.audit_consistency(
        ocr_fields={"doc_number": clean_id},
        mrz_data=mrz_res,
        qr_data=qr_res
    )

    # 10. Phase 21 Weighted Risk Engine
    risk_data = MultiSignalRiskEngine.compute_risk(
        quality_result=quality,
        mrz_result=mrz_res,
        dihedral_valid=dihedral_valid,
        id_number_present=bool(clean_id),
        moire_result=moire_res,
        merged_regions=merged_regions,
        metadata_result=meta_res,
        photo_swap_result=face_res,
        cross_field_result=cross_field_res
    )

    # 11. Phase 22 Confidence & Abstention Gate
    final_verdict = ConfidenceAbstentionGate.evaluate(quality, merged_regions, risk_data)
    trail.log(f"Risk Fusion Synthesis: Score {final_verdict['risk_score']}/100 ({final_verdict['risk_level']})", "COMPLETED")

    # 12. Draw Annotations on Warped Canvas
    annotated = warped_cv.copy()
    for reg in merged_regions:
        x, y, w, h = reg["bbox"]
        tag = "TAMPER [MULTI]" if reg.get("multi_signal_verified") else "TAMPER"
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(annotated, tag, (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1)

    # 13. Phase 25 Court-Admissible BSA 65B Dossier Builder
    dossier = BSADossierBuilder.build_certificate(
        case_id=trail.case_id,
        sha256_hash=sha256,
        risk_decision=final_verdict,
        quality_data=quality,
        mrz_data=mrz_res,
        forensic_summary={"region_count": box_count, "moire_papr": moire_res.get("papr_score", 1.0)},
        metadata_summary=meta_res
    )

    response = {
        "version": CONFIG.APP_VERSION,
        "case_id": trail.case_id,
        "sha256": sha256,
        "quality": quality,
        "deterministic": {
            "mrz": mrz_res,
            "dihedral_valid": dihedral_valid,
            "qr": qr_res
        },
        "forensics": {
            "box_count": box_count,
            "regions": merged_regions,
            "face": {
                "detected": face_res["face_detected"],
                "swap_score": face_res["swap_score"],
                "anomaly_detected": face_res["anomaly_detected"]
            },
            "ela_variance": ela_res.get("ela_variance", 0.0),
            "texture_variance": texture_res.get("mean_variance", 0.0),
            "moire": moire_res
        },
        "metadata": meta_res,
        "cross_field": cross_field_res,
        "risk": final_verdict,
        "dossier": dossier,
        "timeline": trail.get_timeline(),
        "images": {
            "orig_b64": f"data:image/png;base64,{pil_to_b64(orig_pil)}",
            "annotated_b64": f"data:image/png;base64,{mat_to_b64(annotated)}",
            "ela_b64": f"data:image/png;base64,{pil_to_b64(ela_res.get('ela_enhanced', orig_pil))}",
            "face_b64": f"data:image/png;base64,{mat_to_b64(face_res['face_crop'])}" if face_res["face_detected"] else None
        }
    }

    EphemeralCaseLedger.register(trail.case_id, response)
    return response

@app.get("/api/cases/{case_id}")
async def get_case(case_id: str):
    rec = EphemeralCaseLedger.fetch(case_id)
    if not rec: raise HTTPException(status_code=404, detail="Case record not found.")
    return rec
