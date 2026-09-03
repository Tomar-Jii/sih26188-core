class ConfidenceAbstentionGate:
    """Evaluates whether forensic confidence is sufficient to issue a formal verdict."""

    @staticmethod
    def evaluate(quality_result: dict, merged_regions: list, risk_data: dict) -> dict:
        passed_quality = quality_result.get("passed", False)
        box_count = len(merged_regions)

        confidence = 0.90
        sharpness = quality_result.get("metrics", {}).get("sharpness_laplacian", 50.0)

        if sharpness < 20.0:
            confidence -= 0.25
        elif sharpness < 40.0:
            confidence -= 0.10

        confidence = round(max(0.20, min(confidence, 0.96)), 2)

        # Retain explanation breakdown points under both keys for frontend safety
        breakdown_points = risk_data.get("contributions", []) or risk_data.get("breakdown", [])

        # Smart Abstention Condition:
        should_abstain = (not passed_quality) and (box_count == 0) and (risk_data.get("risk_score", 0) < 25)

        if should_abstain:
            return {
                "abstained": True,
                "confidence": confidence,
                "verdict": "ABSTAIN: INSUFFICIENT EVIDENCE",
                "risk_score": 0,
                "risk_level": "UNDETERMINED",
                "recommendation": "RE-ACQUIRE DOCUMENT UNDER PROPER ILLUMINATION WITHOUT BLUR",
                "reason": quality_result.get("abstain_reason", "Low input resolution or blur prevents reliable screening."),
                "breakdown": ["Screening halted: Input document failed optical quality baseline."],
                "contributions": ["Screening halted: Input document failed optical quality baseline."]
            }

        return {
            "abstained": False,
            "confidence": confidence,
            "verdict": risk_data.get("verdict", "AUTHENTIC / UNALTERED"),
            "risk_score": risk_data.get("risk_score", 4),
            "risk_level": risk_data.get("risk_level", "LOW"),
            "recommendation": risk_data.get("recommendation", "PROCEED WITH STANDARD PROCESSING"),
            "reason": None,
            "breakdown": breakdown_points,
            "contributions": breakdown_points
        }
