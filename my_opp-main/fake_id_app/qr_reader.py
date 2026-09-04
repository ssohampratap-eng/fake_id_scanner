import cv2
import numpy as np
from difflib import SequenceMatcher
from typing import Dict, Any, Optional

try:
    import zxingcpp
except Exception:
    zxingcpp = None

try:
    from secure_qr_scanner import decode_aadhaar_secure_qr
except ImportError:
    decode_aadhaar_secure_qr = None


def _try_decode(img) -> Optional[Any]:
    try:
        barcodes = zxingcpp.read_barcodes(img)
        return barcodes[0] if barcodes else None
    except Exception:
        return None


def _decode_with_fallbacks(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    h, w = gray.shape[:2]
    upscaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    sources = [img_bgr, gray, enhanced, upscaled]

    # Aadhaar QR is often small and placed on the right side of the card.
    for source in (img_bgr, gray, enhanced):
        source_h, source_w = source.shape[:2]
        sources.append(source[:, int(source_w * 0.35):])
        sources.append(source[int(source_h * 0.15):int(source_h * 0.90), int(source_w * 0.35):])

    for source in sources:
        bc = _try_decode(source)
        if bc:
            return bc

    return None


def extract_qr_details(image_path: str, clean_id: str = "", ocr_name: str = "") -> Dict[str, Any]:
    img = cv2.imread(image_path)
    if img is None:
        return {
            "qr_found": False,
            "is_matched": True,
            "raw_data": "",
            "warnings": ["Image unreadable."]
        }

    # 1. Try standard zxing-cpp decode with fallbacks
    bc = _decode_with_fallbacks(img) if zxingcpp is not None else None
    raw_text = bc.text if bc else ""

    # 2. Fallback to dedicated Secure Binary QR Parser if zxing-cpp fails or raw text is empty
    if not raw_text and decode_aadhaar_secure_qr is not None:
        secure_res = decode_aadhaar_secure_qr(image_path)
        if secure_res.get("success"):
            parsed_name = secure_res.get("name", "")
            match = True
            if ocr_name and parsed_name and ocr_name != "Not Detected":
                ratio = SequenceMatcher(None, ocr_name.upper(), parsed_name.upper()).ratio()
                match = ratio >= 0.50
            return {
                "qr_found": True,
                "is_matched": match,
                "raw_data": str(secure_res),
                "warnings": [] if match else ["Digital QR payload name does not match OCR extraction."]
            }

    if not raw_text:
        return {
            "qr_found": False,
            "is_matched": True,
            "raw_data": "",
            "warnings": ["Document par koi QR code detect nahi hua."]
        }

    upper_raw = raw_text.upper()
    warnings = []
    name_match = True
    id_match = True

    if clean_id and clean_id != "Not Detected" and len(clean_id) == 12:
        last4 = clean_id[-4:]
        if clean_id not in upper_raw and last4 not in upper_raw:
            id_match = False
            warnings.append("Visible ID number digital QR payload se match nahi karta.")

    if ocr_name and ocr_name != "Not Detected" and len(ocr_name) > 3:
        clean_n = ocr_name.upper().replace(" ", "")
        raw_n = upper_raw.replace(" ", "")
        if clean_n not in raw_n and raw_n not in clean_n:
            ratio = SequenceMatcher(None, clean_n, raw_n).ratio()
            if ratio < 0.55:
                name_match = False
                warnings.append("Visible Name digital QR payload se match nahi karta.")

    return {
        "qr_found": True,
        "is_matched": (name_match and id_match),
        "raw_data": raw_text,
        "warnings": warnings
    }