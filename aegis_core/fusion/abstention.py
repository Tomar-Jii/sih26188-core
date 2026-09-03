class ConfidenceAbstentionGate:
    """Evaluates whether forensic confidence is sufficient to issue a formal verdict."""

    @staticmethod
    def evaluate(quality_result: dict, merged_regions: list, risk_data: dict) -> dict:
        passed_quality = quality_result.get("passed", False)
        box_count = len(merged_regions)
        
        # Calculate decoupled model confidence (0.0 to 1.0)
        confidence = 0.90
        sharpness = quality_result.get("metrics", {}).get("sharpness_laplacian", 50.0)
        
        if sharpness < 20.0:
            confidence -= 0.25
        elif sharpness < 40.0:
            confidence -= 0.10

        confidence = round(max(0.20, min(confidence, 0.96)), 2)

        # Smart Abstention Condition:
        # Only abstain if the image fails quality AND no positive tamper signatures are localized.
        should_abstain = (not passed_quality) and (box_count == 0) and (risk_data["risk_score"] < 25)

        if should_abstain:
            return {
                "abstained": True,
                "confidence": confidence,
                "verdict": "ABSTAIN: INSUFFICIENT EVIDENCE",
                "risk_score": 0,
                "risk_level": "UNDETERMINED",
                "recommendation": "RE-ACQUIRE DOCUMENT UNDER PROPER ILLUMINATION WITHOUT BLUR",
                "reason": quality_result.get("abstain_reason", "Low input resolution or motion blur prevents reliable screening.")
            }

        return {
            "abstained": False,
            "confidence": confidence,
            "verdict": risk_data["verdict"],
            "risk_score": risk_data["risk_score"],
            "risk_level": risk_data["risk_level"],
            "recommendation": risk_data["recommendation"],
            "reason": None
        }
