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
from aegis_core.vision.face_segment import BiometricPortraitAnalyzer
from aegis_core.validators.dihedral import VerhoeffDihedralValidator
from aegis_core.validators.mrz_td3 import ICAO9303MRZParser
from aegis_core.validators.qr_engine import MultiPassQREngine
from aegis_core.forensics.ela import DifferentialELAAnalyzer
from aegis_core.forensics.noise import LocalNoiseAnalyzer
from aegis_core.forensics.texture import TextureFlatnessAnalyzer
from aegis_core.forensics.gradient import EdgeDiscontinuityAnalyzer
from aegis_core.forensics.moire import OpticalMoireAnalyzer
from aegis_core.forensics.metadata import MetadataFootprintAnalyzer
from aegis_core.fusion.cross_field import CrossFieldConsistencyEngine
from aegis_core.fusion.cluster import SpatialRegionMerger
from aegis_core.fusion.risk_engine import MultiSignalRiskEngine
from aegis_core.fusion.abstention import ConfidenceAbstentionGate
from aegis_core.timeline.audit_trail import ForensicAuditTrail, EphemeralCaseLedger
from aegis_core.reporting.bsa_dossier import BSADossierBuilder

app = FastAPI(
    title="AegisID - Defense Forensic Screener (SSB/MHA)",
    version=CONFIG.APP_VERSION
)
templates = Jinja2Templates(directory="templates")

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
    id_number: str = Form(""),
    mrz_raw_input: str = Form("")
):
    # 1. Ingestion Validation
    if file.content_type not in CONFIG.ALLOWED_MIME_TYPES:
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={"error": "UnsupportedMediaType", "detail": f"Permitted formats: {list(CONFIG.ALLOWED_MIME_TYPES)}"}
        )

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "EmptyFilePayload", "detail": "Uploaded file contains 0 bytes."}
        )
    if len(raw_bytes) > CONFIG.MAX_PAYLOAD_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"error": "PayloadTooLarge", "detail": f"File exceeds maximum size of {CONFIG.MAX_PAYLOAD_BYTES // (1024*1024)} MB."}
        )

    try:
        orig_pil = Image.open(io.BytesIO(raw_bytes))
        orig_pil.load()
        rgb_pil = orig_pil.convert("RGB")
        img_cv = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_cv is None or img_cv.size == 0:
            raise ValueError("Corrupted byte matrix")
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "UnprocessableImage", "detail": "Image header or pixel raster is unreadable."}
        )

    try:
        trail = ForensicAuditTrail(case_id=case_id)
        sha256_hash = io.hashlib.sha256(raw_bytes).hexdigest() if hasattr(io, "hashlib") else __import__("hashlib").sha256(raw_bytes).hexdigest()
        trail.log("Document Ingestion & Cryptographic Registration Completed", "VERIFIED")

        # 2. Quality Assessment
        quality_eval = DocumentQualityGate.audit(img_cv)
        trail.log(f"Pre-Screening Quality Gate: {quality_eval['metrics'].get('blur_status', 'Unknown')} Sharpness", 
                  "PASS" if quality_eval["passed"] else "FLAGGED")

        # 3. Perspective Homography
        warped_cv = DocumentPerspectiveWarper.extract_and_warp(img_cv)

        # 4. QR Audit & Spatial Mask Extraction
        qr_eval = MultiPassQREngine.inspect_and_mask(warped_cv)

        # 5. Deterministic Validators
        mrz_eval = ICAO9303MRZParser.parse(mrz_raw_input)
        
        dihedral_valid = False
        clean_id = id_number.replace(" ", "").strip()
        if clean_id and len(clean_id) == 12 and clean_id.isdigit():
            dihedral_valid = VerhoeffDihedralValidator.validate(clean_id)
            trail.log(f"Dihedral D5 Checksum: {'PASS' if dihedral_valid else 'FAIL'}", "PASS" if dihedral_valid else "FAIL")

        # 6. Multi-Signal Spatial Forensics
        ela_eval = DifferentialELAAnalyzer.analyze(rgb_pil, warped_cv, qr_bbox=qr_eval.get("bbox"))
        moire_eval = OpticalMoireAnalyzer.inspect(warped_cv)
        metadata_eval = MetadataFootprintAnalyzer.inspect(orig_pil)
        face_eval = BiometricPortraitAnalyzer.extract_and_audit(warped_cv)

        # Cross-signal candidate collection
        candidate_zones = []
        for zone in ela_eval.get("suspicious_zones", []):
            bbox = zone["bbox"]
            grad_jump = EdgeDiscontinuityAnalyzer.evaluate_boundary_gradient(warped_cv, bbox)
            flatness = TextureFlatnessAnalyzer.audit_patch_flatness(warped_cv, bbox)
            
            score = 0.60 + (grad_jump * 0.20) + (0.15 if flatness["is_unnaturally_flat"] else 0.0)
            candidate_zones.append({
                "bbox": bbox,
                "score": min(0.96, score),
                "signal": "Compression & Texture Discontinuity"
            })

        # NMS Spatial Clustering
        merged_regions = SpatialRegionMerger.merge_regions(candidate_zones, iou_threshold=0.25)
        trail.log(f"Spatial Pixel Analysis: {len(merged_regions)} Altered Region(s) Localized", 
                  "FLAGGED" if len(merged_regions) > 0 else "PASS")

        # 7. Cross-Field Consistency
        cross_field_eval = CrossFieldConsistencyEngine.audit_consistency(
            ocr_fields={"doc_number": clean_id},
            mrz_data=mrz_eval,
            qr_data=qr_eval
        )

        # 8. Risk Fusion & Abstention
        risk_data = MultiSignalRiskEngine.compute_risk(
            quality_result=quality_eval,
            mrz_result=mrz_eval,
            dihedral_valid=dihedral_valid,
            id_number_present=bool(clean_id),
            moire_result=moire_eval,
            merged_regions=merged_regions,
            metadata_result=metadata_eval,
            photo_swap_result=face_eval,
            cross_field_result=cross_field_eval
        )

        final_verdict = ConfidenceAbstentionGate.evaluate(quality_eval, merged_regions, risk_data)
        trail.log(f"Risk Fusion Synthesis: {final_verdict['risk_score']}/100 ({final_verdict['risk_level']})", "COMPLETED")

        # 9. Annotate Image
        annotated_cv = warped_cv.copy()
        for idx, reg in enumerate(merged_regions):
            rx, ry, rw, rh = reg["bbox"]
            cv2.rectangle(annotated_cv, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
            cv2.putText(annotated_cv, f"TAMPER {reg['score']}", (rx, max(ry - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # 10. Construct Section 65B Dossier
        dossier = BSADossierBuilder.build_certificate(
            case_id=trail.case_id,
            sha256_hash=sha256_hash,
            risk_decision=final_verdict,
            quality_data=quality_eval,
            mrz_data=mrz_eval,
            forensic_summary={"region_count": len(merged_regions), "moire_papr": moire_eval["papr_score"]},
            metadata_summary=metadata_eval
        )

        response_payload = {
            "version": CONFIG.APP_VERSION,
            "case_id": trail.case_id,
            "timestamp": dossier["attestation_timestamp"],
            "sha256": sha256_hash,
            "quality": quality_eval,
            "deterministic": {
                "mrz": mrz_eval,
                "dihedral_valid": dihedral_valid,
                "qr": qr_eval
            },
            "forensics": {
                "box_count": len(merged_regions),
                "regions": merged_regions,
                "ela_variance": ela_eval["ela_variance"],
                "moire": moire_eval,
                "photo_swap": face_eval
            },
            "metadata": metadata_eval,
            "cross_field": cross_field_eval,
            "risk": final_verdict,
            "dossier": dossier,
            "timeline": trail.get_timeline(),
            "images": {
                "orig_b64": f"data:image/png;base64,{pil_to_base64(rgb_pil)}",
                "annotated_b64": f"data:image/png;base64,{mat_to_base64(annotated_cv)}",
                "ela_b64": f"data:image/png;base64,{pil_to_base64(ela_eval['ela_enhanced'])}",
                "face_b64": f"data:image/png;base64,{mat_to_base64(face_eval['face_crop'])}" if face_eval["face_detected"] else None
            }
        }

        EphemeralCaseLedger.register(trail.case_id, response_payload)
        return response_payload

    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "PipelineExecutionFault", "detail": "An internal error occurred during forensic screening."}
        )

@app.get("/api/cases/{case_id}")
async def fetch_case_record(case_id: str):
    record = EphemeralCaseLedger.fetch(case_id)
    if not record:
        raise HTTPException(status_code=404, detail="Case record not found in active session memory.")
    return record
