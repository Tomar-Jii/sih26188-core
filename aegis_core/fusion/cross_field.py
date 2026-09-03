import re

class CrossFieldConsistencyEngine:
    """Evaluates cross-source coherence across visible strings, MRZ tokens, and barcode payloads."""

    @staticmethod
    def audit_consistency(ocr_fields: dict, mrz_data: dict, qr_data: dict) -> dict:
        matrix = {}
        conflicts = []

        # 1. Document Identifier Cross-Check
        ocr_doc_num = ocr_fields.get("doc_number", "").replace(" ", "").upper()
        mrz_doc_num = mrz_data.get("doc_number", "").replace(" ", "").upper()

        if ocr_doc_num and mrz_doc_num:
            if ocr_doc_num == mrz_doc_num:
                matrix["document_number"] = {"status": "MATCH", "evidence": "OCR matches MRZ token"}
            else:
                matrix["document_number"] = {"status": "CONFLICT", "evidence": f"OCR ({ocr_doc_num}) != MRZ ({mrz_doc_num})"}
                conflicts.append("Document number discrepancy between OCR text and MRZ line.")
        elif mrz_doc_num:
            matrix["document_number"] = {"status": "VERIFIED_VIA_MRZ", "evidence": f"MRZ token: {mrz_doc_num}"}
        else:
            matrix["document_number"] = {"status": "NOT_AVAILABLE", "evidence": "Insufficient cross-validation sources"}

        # 2. Date of Birth Cross-Check
        ocr_dob = ocr_fields.get("dob", "").replace("/", "").replace("-", "")
        mrz_dob = mrz_data.get("dob", "")

        if ocr_dob and mrz_dob:
            # Check YYMMDD format alignment
            if mrz_dob in ocr_dob or ocr_dob in mrz_dob:
                matrix["date_of_birth"] = {"status": "MATCH", "evidence": "DOB components align"}
            else:
                matrix["date_of_birth"] = {"status": "CONFLICT", "evidence": f"OCR DOB does not match MRZ ({mrz_dob})"}
                conflicts.append("Date of birth contradiction between primary text and MRZ.")
        else:
            matrix["date_of_birth"] = {"status": "NOT_AVAILABLE", "evidence": "Missing cross-field DOB sources"}

        # 3. Barcode Payload Correlation
        qr_status = qr_data.get("detected", False)
        if qr_status and qr_data.get("payload"):
            payload = qr_data["payload"].upper()
            if mrz_doc_num and mrz_doc_num in payload:
                matrix["qr_correlation"] = {"status": "MATCH", "evidence": "Document token verified in QR payload"}
            else:
                matrix["qr_correlation"] = {"status": "INCONCLUSIVE", "evidence": "QR payload present but unmapped to visible fields"}
        else:
            matrix["qr_correlation"] = {"status": "NOT_AVAILABLE", "evidence": "No readable barcode payload decoded"}

        return {
            "matrix": matrix,
            "conflicts": conflicts,
            "has_critical_conflict": len(conflicts) > 0
        }
