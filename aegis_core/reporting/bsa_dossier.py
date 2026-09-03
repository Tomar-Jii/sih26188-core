from datetime import datetime, timezone

class BSADossierBuilder:
    """Constructs court-admissible forensic screening dossiers under Section 65B of BSA 2023."""

    @staticmethod
    def build_certificate(
        case_id: str,
        sha256_hash: str,
        risk_decision: dict,
        quality_data: dict,
        mrz_data: dict,
        forensic_summary: dict,
        metadata_summary: dict
    ) -> dict:
        timestamp_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

        legal_attestation = (
            "This electronic forensic dossier is compiled pursuant to Section 63 and Section 65B "
            "of the Bharatiya Sakshya Adhiniyam, 2023. The diagnostic metrics herein are produced "
            "via deterministic checksum matrices, 2D Fast Fourier Transform frequency analysis, "
            "and differential quantization residual evaluation without intermediate manual tampering. "
            "This document constitutes automated screening intelligence and assists authorized personnel."
        )

        return {
            "legal_framework": "Bharatiya Sakshya Adhiniyam, 2023 (BSA) / Section 65B",
            "jurisdiction": "Ministry of Home Affairs // Sashastra Seema Bal (SSB)",
            "case_identifier": case_id,
            "attestation_timestamp": timestamp_now,
            "evidence_sha256": sha256_hash,
            "screening_verdict": risk_decision["verdict"],
            "risk_index": f"{risk_decision['risk_score']}/100 ({risk_decision['risk_level']})",
            "confidence_metric": f"{int(risk_decision['confidence'] * 100)}%",
            "recommended_action": risk_decision["recommendation"],
            "findings": {
                "quality_profile": quality_data.get("metrics", {}),
                "mrz_status": mrz_data.get("checks", {}),
                "spatial_tamper_regions": forensic_summary.get("region_count", 0),
                "screen_recapture_papr": forensic_summary.get("moire_papr", 1.0),
                "software_fingerprints": metadata_summary.get("software_traces", [])
            },
            "attestation_clause": legal_attestation,
            "certifying_officer": "SYSTEM_AUTOMATED_FORENSIC_EXAMINER"
        }
