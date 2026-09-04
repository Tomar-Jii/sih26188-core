import io
import base64
import cv2
import hashlib
import traceback
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from aegis_core.config import CONFIG
from aegis_core.ingestion.handler import StreamIngestionHandler, IngestionSecurityError
from aegis_core.quality.gatekeeper import DocumentQualityGate
from aegis_core.vision.warp import DocumentPerspectiveWarper
from aegis_core.vision.face_segment import BiometricPortraitAnalyzer
from aegis_core.vision.face_match import BiometricFaceMatcher
from aegis_core.vision.id_extractor import DocumentIDAutoExtractor
from aegis_core.vision.font_audit import FontDisparityAnalyzer
from aegis_core.classification.doc_classifier import DocumentClassifier
from aegis_core.validators.dihedral import VerhoeffDihedralValidator
from aegis_core.validators.mrz_td3 import ICAO9303MRZParser
from aegis_core.validators.qr_engine import MultiPassQREngine
from aegis_core.forensics.ela import DifferentialELAAnalyzer
from aegis_core.forensics.texture import TextureFlatnessAnalyzer
from aegis_core.forensics.gradient import EdgeDiscontinuityAnalyzer
from aegis_core.forensics.noise import LocalNoiseAnalyzer
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

def clean_for_json(data):
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items() if not isinstance(v, (np.ndarray, bytes))}
    elif isinstance(data, list):
        return [clean_for_json(item) for item in data if not isinstance(item, (np.ndarray, bytes))]
    elif isinstance(data, (np.integer, np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64, np.float32)):
        return float(data)
    elif isinstance(data, (np.bool_, bool)):
        return bool(data)
    elif isinstance(data, (bytes, bytearray, np.ndarray)):
        return None
    return data

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/audit")
async def execute_audit(
    file: UploadFile = File(...),
    live_face: UploadFile = File(None),
    case_id: str = Form(None),
    id_number: str = Form(""),
    mrz_raw_input: str = Form("")
):
    try:
        raw_bytes, orig_pil, img_cv, verified_mime = await StreamIngestionHandler.ingest_and_sanitize(file)
    except IngestionSecurityError as sec_err:
        return JSONResponse(status_code=sec_err.status_code, content={"detail": sec_err.message})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"detail": f"Stream ingestion fault: {str(exc)}"})

    live_cv = None
    if live_face is not None:
        try:
            live_bytes = await live_face.read()
            if len(live_bytes) > 0:
                live_pil = Image.open(io.BytesIO(live_bytes))
                live_pil = ImageOps.exif_transpose(live_pil).convert("RGB")
                live_cv = cv2.cvtColor(np.array(live_pil), cv2.COLOR_RGB2BGR)
        except Exception:
            live_cv = None

    try:
        trail = ForensicAuditTrail(case_id=case_id)
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        trail.log(f"Stream Ingested & Quarantined ({verified_mime})", "VERIFIED")

        # 1. Quality Gate
        quality = DocumentQualityGate.audit(img_cv)
        q_status = quality.get("metrics", {}).get("blur_status", "Acceptable")
        trail.log(f"Pre-Screening Quality Gate: {q_status} Sharpness", "PASS" if quality.get("passed") else "FLAGGED")

        # 2. Perspective Normalization
        warped_cv = DocumentPerspectiveWarper.extract_and_warp(img_cv)
        if warped_cv is None or warped_cv.size == 0:
            warped_cv = img_cv.copy()
        warped_pil = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))

        # 3. Multi-Pass Secure QR Decoder
        qr_res, qr_photo_bytes = MultiPassQREngine.inspect_and_mask(warped_cv)
        all_qr_boxes = qr_res.get("bboxes", [])
        if qr_res.get("bbox") and qr_res.get("bbox") not in all_qr_boxes:
            all_qr_boxes.append(qr_res["bbox"])

        if qr_res.get("detected"):
            trail.log(f"Cryptographic Ground-Truth: {qr_res.get('status')}", "PASS")

        # 4. Auto ID Extractor & Dihedral Checksum
        clean_id = id_number.replace(" ", "").strip()
        detected_id = clean_id
        dihedral_valid = None

        if not detected_id:
            auto_id, is_valid = DocumentIDAutoExtractor.extract_id(warped_cv, qr_res.get("payload"))
            if auto_id:
                detected_id = auto_id
                dihedral_valid = is_valid
                trail.log(f"Document Identifier Verified: {detected_id}", "PASS")

        if clean_id and len(clean_id) == 12 and clean_id.isdigit():
            dihedral_valid = VerhoeffDihedralValidator.validate(clean_id)
            trail.log(f"Dihedral D5 Checksum: {'PASS' if dihedral_valid else 'FAIL'}", "PASS" if dihedral_valid else "FAIL")
        elif qr_res.get("detected"):
            dihedral_valid = True

        mrz_res = ICAO9303MRZParser.parse(mrz_raw_input)
        if mrz_res.get("is_mrz_detected"):
            trail.log(f"ICAO Doc 9303 MRZ Checksum: {'PASS' if mrz_res.get('all_checks_passed') else 'FAIL'}", "PASS" if mrz_res.get('all_checks_passed') else "FLAGGED")

        # 5. Document Classification
        classification_res = DocumentClassifier.classify(warped_cv, mrz_res=mrz_res, qr_res=qr_res)
        trail.log(f"Document Classification: {classification_res['document_type']}", "PASS")

        # 6. Biometric Portrait Extraction with Full-Crown Envelope
        face_res = BiometricPortraitAnalyzer.extract_and_audit(warped_cv)
        exclusion_envelope = face_res.get("envelope_bbox", face_res.get("bbox"))

        # 7. Biometric Avatar Match
        face_match_res = {"evaluated": False, "match_status": "SKIPPED", "similarity_score": 1.0, "is_photo_swap": False}
        if qr_photo_bytes and face_res.get("face_detected"):
            face_match_res = BiometricFaceMatcher.compare_portraits(face_res["face_crop"], qr_photo_bytes)
            if face_match_res.get("evaluated"):
                log_status = "PASS" if not face_match_res.get("is_photo_swap") else "FLAGGED"
                trail.log(f"Biometric Avatar Audit: {face_match_res['match_status']} (Corr: {int(face_match_res['similarity_score']*100)}%)", log_status)

        # 8. Live Selfie Face Match
        live_match_res = {"evaluated": False, "match_status": "SKIPPED", "similarity_score": 0.0, "is_match": True}
        live_crop_mat = None
        if live_cv is not None and face_res.get("face_detected"):
            live_match_res, live_crop_mat = BiometricFaceMatcher.compare_live_face(face_res["face_crop"], live_cv)
            if live_match_res.get("evaluated"):
                log_status = "PASS" if live_match_res.get("is_match") else "FLAGGED"
                trail.log(f"Live 1:1 Face Verification: {live_match_res['match_status']} ({int(live_match_res['similarity_score']*100)}% Similarity)", log_status)

        # 9. Spatial Defacement Scanners with Full Portrait & QR Immunity
        texture_res = TextureFlatnessAnalyzer.detect_digital_strokes(
            warped_cv,
            qr_bbox=qr_res.get("bbox"),
            face_bbox=exclusion_envelope,
            qr_bboxes=all_qr_boxes
        )
        ela_res = DifferentialELAAnalyzer.analyze(warped_pil, warped_cv, qr_bbox=qr_res.get("bbox"))
        gradient_res = EdgeDiscontinuityAnalyzer.audit(warped_cv, qr_bbox=qr_res.get("bbox"))
        noise_res = LocalNoiseAnalyzer.audit(warped_cv, qr_bbox=qr_res.get("bbox"))
        moire_res = OpticalMoireAnalyzer.inspect(warped_cv)
        meta_res = MetadataFootprintAnalyzer.inspect(orig_pil)

        # 10. Spatial Region Merger (NMS)
        raw_candidates = texture_res.get("tamper_zones", [])
        if face_match_res.get("is_photo_swap") and face_res.get("bbox"):
            fx, fy, fw, fh = face_res["bbox"]
            raw_candidates.append({
                "bbox": [int(fx), int(fy), int(fw), int(fh)],
                "score": 0.99,
                "signal": "Biometric Avatar Mismatch (Cryptographic Photo-Swap)"
            })

        merged_regions = SpatialRegionMerger.merge_regions(raw_candidates, iou_thresh=0.15)
        box_count = len(merged_regions)
        trail.log(f"Spatial Fusion (NMS): {box_count} Defacement Zone(s) Isolated", "FLAGGED" if box_count > 0 else "PASS")

        # 11. Cross-Field Verification
        cross_field_res = CrossFieldConsistencyEngine.audit_consistency(
            ocr_fields={"doc_number": detected_id or ""},
            mrz_data=mrz_res,
            qr_data=qr_res
        )

        # 12. Risk Engine Synthesis
        photo_swap_data = {
            "face_detected": face_res.get("face_detected", False),
            "anomaly_detected": face_match_res.get("is_photo_swap", False) or (not live_match_res.get("is_match") if live_match_res.get("evaluated") else False),
            "swap_score": 0.95 if (face_match_res.get("is_photo_swap") or (not live_match_res.get("is_match") if live_match_res.get("evaluated") else False)) else 0.05
        }

        risk_data = MultiSignalRiskEngine.compute_risk(
            quality_result=quality,
            mrz_result=mrz_res,
            dihedral_valid=bool(dihedral_valid),
            id_number_present=bool(detected_id),
            moire_result=moire_res,
            merged_regions=merged_regions,
            metadata_result=meta_res,
            photo_swap_result=photo_swap_data,
            cross_field_result=cross_field_res
        )

        # 13. Confidence & Abstention Gate
        final_verdict = ConfidenceAbstentionGate.evaluate(quality, merged_regions, risk_data)
        trail.log(f"Risk Fusion Synthesis: Score {final_verdict['risk_score']}/100 ({final_verdict['risk_level']})", "COMPLETED")

        # 14. Annotate Canvas
        annotated = warped_cv.copy()
        for reg in merged_regions:
            x, y, w, h = reg["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated, "TAMPER", (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # 15. BSA 65B Dossier Builder
        dossier = BSADossierBuilder.build_certificate(
            case_id=trail.case_id,
            sha256_hash=sha256,
            risk_decision=final_verdict,
            quality_data=quality,
            mrz_data=mrz_res,
            forensic_summary={"region_count": box_count, "moire_papr": moire_res.get("papr_score", 1.0)},
            metadata_summary=meta_res
        )

        card_face_crop_b64 = mat_to_b64(face_res['face_crop']) if face_res.get("face_detected") else None
        live_face_crop_b64 = mat_to_b64(live_crop_mat) if live_crop_mat is not None else None

        raw_response = {
            "version": CONFIG.APP_VERSION,
            "case_id": trail.case_id,
            "sha256": sha256,
            "classification": classification_res,
            "quality": quality,
            "deterministic": {
                "mrz": mrz_res,
                "dihedral_valid": dihedral_valid,
                "detected_id": detected_id,
                "eval_id_present": bool(detected_id),
                "qr": qr_res,
                "live_face_verification": live_match_res
            },
            "forensics": {
                "box_count": box_count,
                "regions": merged_regions,
                "face": {
                    "detected": face_res.get("face_detected", False),
                    "swap_score": photo_swap_data["swap_score"],
                    "anomaly_detected": photo_swap_data["anomaly_detected"],
                    "qr_avatar_match": face_match_res,
                    "live_selfie_match": live_match_res
                },
                "ela_variance": ela_res.get("ela_variance", 0.0),
                "texture_variance": texture_res.get("mean_variance", 0.0),
                "gradient_mean": gradient_res.get("mean_gradient", 0.0),
                "noise_variance": noise_res.get("global_variance", 0.0),
                "moire": moire_res
            },
            "metadata": meta_res,
            "cross_field": cross_field_res,
            "risk": final_verdict,
            "dossier": dossier,
            "timeline": trail.get_timeline(),
            "images": {
                "orig_b64": f"data:image/png;base64,{mat_to_b64(warped_cv)}",
                "annotated_b64": f"data:image/png;base64,{mat_to_b64(annotated)}",
                "ela_b64": f"data:image/png;base64,{pil_to_b64(ela_res.get('ela_enhanced', orig_pil))}",
                "face_b64": f"data:image/png;base64,{card_face_crop_b64}" if card_face_crop_b64 else None,
                "live_face_b64": f"data:image/png;base64,{live_face_crop_b64}" if live_face_crop_b64 else None
            }
        }

        sanitized_response = clean_for_json(raw_response)
        EphemeralCaseLedger.register(trail.case_id, sanitized_response)
        return JSONResponse(status_code=200, content=sanitized_response)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"{type(e).__name__}: {str(e)}"})

@app.get("/api/cases/{case_id}")
async def get_case(case_id: str):
    rec = EphemeralCaseLedger.fetch(case_id)
    if not rec: raise HTTPException(status_code=404, detail="Case record not found.")
    return rec
