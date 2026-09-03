from aegis_core.config import CONFIG

class MultiSignalRiskEngine:
    """Configurable weighted risk fusion engine providing explainable score breakdowns."""

    @classmethod
    def compute_risk(
        cls,
        quality_result: dict,
        mrz_result: dict,
        dihedral_valid: bool,
        id_number_present: bool,
        moire_result: dict,
        merged_regions: list,
        metadata_result: dict,
        photo_swap_result: dict,
        cross_field_result: dict
    ) -> dict:
        contributions = []
        accumulated_risk = 0.0

        # 1. Deterministic Failures (Checksums)
        if mrz_result.get("is_mrz_detected") and not mrz_result.get("all_checks_passed"):
            accumulated_risk += 35.0
            contributions.append("+35 ICAO Doc 9303 Checksum verification failure on machine-readable zone.")

        if id_number_present and not dihedral_valid:
            accumulated_risk += 45.0
            contributions.append("+45 Permutation Dihedral Checksum failed: Document identification string is mathematically forged.")

        # 2. Spatial Forensic Anomalies (Merged NMS Regions)
        region_count = len(merged_regions)
        if region_count > 0:
            assigned = min(region_count * 18.0, 50.0)
            accumulated_risk += assigned
            signals_found = set()
            for r in merged_regions:
                signals_found.update(r.get("signals", []))
            signals_str = ", ".join(signals_found)
            contributions.append(f"+{int(assigned)} Localized spatial tampering in {region_count} distinct zone(s) [{signals_str}].")

        # 3. Optical Surface Liveness (Screen Grid Moiré)
        if moire_result.get("is_screen_recapture"):
            accumulated_risk += 28.0
            contributions.append(f"+28 High-frequency optical Moiré detected (PAPR: {moire_result.get('papr_score')}) - Screen photo recapture.")

        # 4. Biometric Photo-Swap Boundary Step
        if photo_swap_result.get("anomaly_detected"):
            accumulated_risk += 30.0
            contributions.append(f"+30 Biometric portrait perimeter discontinuity: High boundary gradient step suggests headshot replacement.")

        # 5. Metadata Traces
        software_traces = metadata_result.get("software_traces", [])
        if software_traces:
            accumulated_risk += 18.0
            contributions.append(f"+18 Image processing tool signatures detected in metadata: {software_traces[0]}")

        # 6. Cross-Field Contradictions
        conflicts = cross_field_result.get("conflicts", [])
        if conflicts:
            accumulated_risk += 25.0
            contributions.append(f"+25 Cross-field mismatch: {conflicts[0]}")

        # Bounded Risk Normalization
        final_risk = int(min(max(accumulated_risk, 3.0), 99.0))

        if final_risk >= 52:
            level = "HIGH"
            verdict = "SUSPICIOUS / POTENTIAL FORGERY"
            recommendation = "MANDATORY INVESTIGATOR FORENSIC REVIEW"
        elif final_risk >= 24:
            level = "MEDIUM"
            verdict = "ANOMALIES DETECTED"
            recommendation = "SECONDARY SCRUTINY REQUIRED"
        else:
            level = "LOW"
            verdict = "AUTHENTIC / UNALTERED"
            recommendation = "PROCEED WITH STANDARD PROCESSING"

        return {
            "risk_score": final_risk,
            "risk_level": level,
            "verdict": verdict,
            "recommendation": recommendation,
            "contributions": contributions if contributions else ["All structural, cryptographic, and spatial signals within normal baseline tolerances."]
        }
