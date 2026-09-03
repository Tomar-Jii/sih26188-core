import re

class ICAO9303MRZParser:
    """ICAO Doc 9303 TD3 Machine Readable Zone parser with 7-3-1 weight algorithms."""
    WEIGHTS = [7, 3, 1]

    @classmethod
    def compute_check_digit(cls, data_str: str) -> int:
        total = 0
        for i, char in enumerate(data_str):
            if '0' <= char <= '9':
                val = int(char)
            elif 'A' <= char <= 'Z':
                val = ord(char) - 55
            else:
                val = 0
            total += val * cls.WEIGHTS[i % 3]
        return total % 10

    @classmethod
    def parse(cls, raw_text: str) -> dict:
        if not raw_text or not isinstance(raw_text, str):
            return {"is_mrz_detected": False, "checks": {}, "all_checks_passed": False}

        lines = [l.strip().replace(" ", "").upper() for l in raw_text.splitlines() if len(l.strip()) >= 30]
        valid_lines = [l for l in lines if re.match(r'^[A-Z0-9<]+$', l)]

        if len(valid_lines) >= 2 and len(valid_lines[0]) == 44 and len(valid_lines[1]) == 44:
            l1, l2 = valid_lines[0], valid_lines[1]
            doc_num = l2[0:9]
            doc_check = l2[9]
            dob = l2[13:19]
            dob_check = l2[19]
            exp = l2[21:27]
            exp_check = l2[27]
            comp = l2[0:10] + l2[13:20] + l2[21:43]
            comp_check = l2[43]

            c1 = str(cls.compute_check_digit(doc_num)) == doc_check
            c2 = str(cls.compute_check_digit(dob)) == dob_check
            c3 = str(cls.compute_check_digit(exp)) == exp_check
            c4 = str(cls.compute_check_digit(comp)) == comp_check

            return {
                "is_mrz_detected": True,
                "type": "TD3_PASSPORT",
                "doc_number": doc_num.replace("<", ""),
                "dob": dob,
                "expiry": exp,
                "nationality": l2[10:13].replace("<", ""),
                "checks": {
                    "doc_number": "PASS" if c1 else "FAIL",
                    "dob": "PASS" if c2 else "FAIL",
                    "expiry": "PASS" if c3 else "FAIL",
                    "composite": "PASS" if c4 else "FAIL"
                },
                "all_checks_passed": bool(c1 and c2 and c3 and c4)
            }

        return {"is_mrz_detected": False, "checks": {}, "all_checks_passed": False}
