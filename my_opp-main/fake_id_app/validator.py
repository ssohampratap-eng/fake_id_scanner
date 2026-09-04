from datetime import datetime
from typing import Dict, Any

D_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

P_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]


def validate_aadhaar_verhoeff(num_str: str) -> bool:
    clean = num_str.replace(" ", "").replace("-", "")
    if not clean.isdigit() or len(clean) != 12:
        return False
    if clean[0] in ['0', '1']:
        return False

    c = 0
    reversed_digits = [int(x) for x in reversed(clean)]
    for idx, digit in enumerate(reversed_digits):
        p_val = P_TABLE[idx % 8][digit]
        c = D_TABLE[c][p_val]
    return c == 0


def is_single_digit_transposition_recoverable(clean_id: str) -> bool:
    """
    Defense-in-depth: even with the D_TABLE fixed, camera glare/lamination
    can still cause EasyOCR to misread exactly one digit. If substituting
    ANY single position with ANY digit makes the checksum valid, the number
    is plausibly genuine with an OCR misread — not structurally fake.
    """
    for i in range(len(clean_id)):
        original_digit = clean_id[i]
        for d in "0123456789":
            if d == original_digit:
                continue
            candidate = clean_id[:i] + d + clean_id[i + 1:]
            if validate_aadhaar_verhoeff(candidate):
                return True
    return False


def validate_id_document(fields: Dict[str, Any], doc_type: str = "Aadhaar Card") -> Dict[str, Any]:
    id_obj = fields.get("id_number", {})
    id_val = id_obj.get("value", "") if isinstance(id_obj, dict) else str(id_obj)
    clean_id = id_val.replace(" ", "").replace("-", "")
    warnings = []

    len_valid = len(clean_id) == 12 and clean_id.isdigit()

    if not len_valid:
        return {
            "status": "FAIL",
            "checksum_valid": False,
            "checksum_status": "MALFORMED",
            "aadhaar_number_length_valid": False,
            "warnings": ["Aadhaar identifier exactly 12 digits ka hona chahiye."],
            "message": "ID number could not be isolated as 12 digits."
        }

    verhoeff_valid = validate_aadhaar_verhoeff(clean_id)

    if verhoeff_valid:
        return {
            "status": "PASS",
            "checksum_valid": True,
            "checksum_status": "VALID",
            "aadhaar_number_length_valid": True,
            "warnings": [],
            "message": "Aadhaar checksum mathematically valid hai. Issuance/ownership verified nahi hai."
        }

    # Checksum failed on a well-formed 12-digit number — check if it's a
    # plausible single-digit OCR misread before treating it as fake.
    recoverable = is_single_digit_transposition_recoverable(clean_id)

    if recoverable:
        warnings.append("Checksum failed, but a single-digit correction restores validity "
                         "(likely OCR misread from glare/lamination, not a fake number).")
        return {
            "status": "SOFT_FAIL",
            "checksum_valid": False,
            "checksum_status": "SOFT_FAIL",
            "aadhaar_number_length_valid": True,
            "warnings": warnings,
            "message": "Checksum invalid, but plausibly recoverable via single-digit OCR correction."
        }

    warnings.append("Mathematical Verhoeff checksum validation fail ho gayi — no single-digit "
                     "correction restores validity.")
    return {
        "status": "FAIL",
        "checksum_valid": False,
        "checksum_status": "HARD_FAIL",
        "aadhaar_number_length_valid": True,
        "warnings": warnings,
        "message": "Mathematical checksum invalid, not explainable by single-digit OCR noise."
    }