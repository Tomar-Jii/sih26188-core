import io
import base64
import cv2
import hashlib
import traceback
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from aegis_core.config import CONFIG
from aegis_core.ingestion.handler import StreamIngestionHandler, IngestionSecurityError
from aegis_core.quality.gatekeeper import DocumentQualityGate
from aegis_core.vision.warp import DocumentPerspectiveWarper
from aegis_core.vision.face_segment import BiometricPortraitAnalyzer
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
    try:
        raw_bytes, orig_pil, img_cv, verified_mime = await StreamIngestionHandler.ingest_and_sanitize(file)
    except IngestionSecurityError as sec_err:
        return JSONResponse(status_code=sec_err.status_code, content={"detail": sec_err.message})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"detail": f"Stream ingestion fault: {str(exc)}"})

    try:
        trail = ForensicAuditTrail(case_id=case_id)
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        trail.log(f"Stream Ingested & Quarantined (Verified Binary: {verified_mime})", "VERIFIED")

        # 1. Quality Gate
        quality = DocumentQualityGate.audit(img_cv)
        q_status = quality.get("metrics", {}).get("blur_status", "Acceptable")
        trail.log(f"Pre-Screening Quality Gate: {q_status} Sharpness", "PASS" if quality.get("passed") else "FLAGGED")

        # 2. Boundary Perspective Normalization
        warped_cv = DocumentPerspectiveWarper.extract_and_warp(img_cv)
        if warped_cv is None or warped_cv.size == 0:
            warped_cv = img_cv.copy()
        warped_pil = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))

        # 3. Deterministic Cryptographic Validators
        clean_id = id_number.replace(" ", "").strip()
        dihedral_valid = False
        if clean_id and len(clean_id) == 12 and clean_id.isdigit():
            dihedral_valid = VerhoeffDihedralValidator.validate(clean_id)
            trail.log(f"Dihedral D5 Checksum ({clean_id[:4]} **** {clean_id[-4:]}): {'PASS' if dihedral_valid else 'FAIL'}", "PASS" if dihedral_valid else "FAIL")

        mrz_res = ICAO9303MRZParser.parse(mrz_raw_input)
        if mrz_res.get("is_mrz_detected"):
            trail.log(f"ICAO Doc 9303 MRZ Checksum: {'PASS' if mrz_res.get('all_checks_passed') else 'FAIL'}", "PASS" if mrz_res.get('all_checks_passed') else "FLAGGED")

        # 4. Multi-Pass QR Decoder & Masker
        qr_res = MultiPassQREngine.inspect_and_mask(warped_cv)

        # 5. Multi-Modal Document Classification
        classification_res = DocumentClassifier.classify(warped_cv, mrz_res=mrz_res, qr_res=qr_res)
        trail.log(f"Document Classification: {classification_res['document_type']} (Confidence: {int(classification_res['confidence']*100)}%)", "PASS")

        # 6. Biometric Portrait Segmentation & Photo-Swap Audit
        face_res = BiometricPortraitAnalyzer.extract_and_audit(warped_cv)
        swap_score_val = face_res.get("swap_score", face_res.get("photo_swap_score", 0.0))
        trail.log(f"Biometric Portrait: {'Extracted' if face_res.get('face_detected') else 'No Face'} (Swap Score: {swap_score_val})",
                  "FLAGGED" if face_res.get("anomaly_detected") else "PASS")

        # 7. Typographic Font Disparity
        font_res = {"tamper_zones": [], "count": 0, "anomalous_rows": 0}
        try:
            font_res = FontDisparityAnalyzer.audit(warped_cv, qr_bbox=qr_res.get("bbox"), face_bbox=face_res.get("bbox"))
            if font_res.get("count", 0) > 0:
                trail.log(f"Typographic Audit: {font_res['count']} Glyph Anomalies in {font_res['anomalous_rows']} Row(s)", "FLAGGED")
        except Exception:
            pass

        # 8. Multi-Signal Spatial Scanners
        ela_res = DifferentialELAAnalyzer.analyze(warped_pil, warped_cv, qr_bbox=qr_res.get("bbox"))
        
        texture_res = {"tamper_zones": [], "count": 0, "mean_variance": 0.0}
        if hasattr(TextureFlatnessAnalyzer, "detect_digital_strokes"):
            texture_res = TextureFlatnessAnalyzer.detect_digital_strokes(warped_cv, qr_bbox=qr_res.get("bbox"))

        gradient_res = {"tamper_zones": [], "count": 0, "mean_gradient": 0.0}
        if hasattr(EdgeDiscontinuityAnalyzer, "audit"):
            gradient_res = EdgeDiscontinuityAnalyzer.audit(warped_cv, qr_bbox=qr_res.get("bbox"))

        noise_res = {"tamper_zones": [], "count": 0, "global_variance": 0.0}
        if hasattr(LocalNoiseAnalyzer, "audit"):
            noise_res = LocalNoiseAnalyzer.audit(warped_cv, qr_bbox=qr_res.get("bbox"))
        elif hasattr(LocalNoiseAnalyzer, "extract_residual_map"):
            _, n_var = LocalNoiseAnalyzer.extract_residual_map(warped_cv)
            noise_res["global_variance"] = n_var

        moire_res = OpticalMoireAnalyzer.inspect(warped_cv)
        meta_res = MetadataFootprintAnalyzer.inspect(orig_pil)

        # 9. Spatial NMS Clustering
        raw_candidates = (
            ela_res.get("suspicious_zones", []) +
            texture_res.get("tamper_zones", []) +
            gradient_res.get("tamper_zones", []) +
            noise_res.get("tamper_zones", []) +
            font_res.get("tamper_zones", [])
        )
        if face_res.get("tamper_zone"):
            raw_candidates.append(face_res["tamper_zone"])

        # Format normalization for clustering
        normalized_candidates = []
        for c in raw_candidates:
            normalized_candidates.append({
                "bbox": c["bbox"],
                "score": c.get("score", 0.85),
                "signal": c.get("signal", "Spatial Anomaly")
            })

        merged_regions = SpatialRegionMerger.merge_regions(normalized_candidates, iou_threshold=0.20) if hasattr(SpatialRegionMerger, "merge_regions") else []
        box_count = len(merged_regions)
        trail.log(f"Spatial Fusion (NMS): {box_count} Consolidated Tamper Zone(s)", "FLAGGED" if box_count > 0 else "PASS")

        # 10. Cross-Field Coherence Matrix
        cross_field_res = CrossFieldConsistencyEngine.audit_consistency(
            ocr_fields={"doc_number": clean_id},
            mrz_data=mrz_res,
            qr_data=qr_res
        )

        # 11. Weighted Risk Fusion Engine
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

        # 12. Confidence & Abstention Gate
        final_verdict = ConfidenceAbstentionGate.evaluate(quality, merged_regions, risk_data)
        trail.log(f"Risk Fusion Synthesis: Score {final_verdict['risk_score']}/100 ({final_verdict['risk_level']})", "COMPLETED")

        # 13. Annotate Warped Canvas
        annotated = warped_cv.copy()
        for reg in merged_regions:
            x, y, w, h = reg["bbox"]
            tag = "TAMPER [MULTI]" if reg.get("multi_signal_verified") else "TAMPER"
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated, tag, (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1)

        # 14. Section 65B Dossier Builder
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
            "classification": classification_res,
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
                    "detected": face_res.get("face_detected", False),
                    "swap_score": swap_score_val,
                    "anomaly_detected": face_res.get("anomaly_detected", False)
                },
                "font_audit": {
                    "anomalies_detected": font_res.get("count", 0),
                    "anomalous_rows": font_res.get("anomalous_rows", 0)
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
                "face_b64": f"data:image/png;base64,{mat_to_b64(face_res['face_crop'])}" if face_res.get("face_detected") else None
            }
        }

        EphemeralCaseLedger.register(trail.case_id, response)
        return response

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"{type(e).__name__}: {str(e)}"}
        )

@app.get("/api/cases/{case_id}")
async def get_case(case_id: str):
    rec = EphemeralCaseLedger.fetch(case_id)
    if not rec: raise HTTPException(status_code=404, detail="Case record not found.")
    return rec
