from forensics import ICAO9303Validator, RiskFusionEngine


def test_mrz_validator_valid_td3():
    mrz = (
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
    )
    result = ICAO9303Validator.parse_mrz(mrz)
    assert result["is_mrz_detected"] is True
    assert result["all_checks_passed"] is True


def test_mrz_validator_invalid_and_empty_and_malformed():
    invalid = (
        "P<UTOSPECIMEN<<JANE<CITIZEN<<<<<<<<<<<<<<<<<\n"
        "L898902C<9UTO9408124F2910236ZE184226B<<<<<14"
    )
    malformed = "!!@@## not mrz @@##"
    assert ICAO9303Validator.parse_mrz(invalid)["all_checks_passed"] is False
    assert ICAO9303Validator.parse_mrz(malformed)["is_mrz_detected"] is False
    assert ICAO9303Validator.parse_mrz("")["is_mrz_detected"] is False


def test_risk_engine_abstains_on_poor_quality_without_regions():
    result = RiskFusionEngine.evaluate(
        quality={"passed": False},
        deterministic={"mrz": {"is_mrz_detected": False, "all_checks_passed": False}},
        forensics={"box_count": 0, "ela_variance": 0, "moire": {"is_screen_recapture": False}},
        metadata={"software_traces": []},
        cross_field={"conflicts": []},
    )
    assert "ABSTAIN" in result["verdict"]
    assert result["risk_level"] == "UNDETERMINED"
