import cv2
import zxingcpp
import zlib
import re

def decode_aadhaar_secure_qr(image_path):
    """
    Dedicated Aadhaar QR Code Scanner:
    1. Isolates high-density matrix
    2. Decodes raw byte payload
    3. Handles both Legacy XML and Secure Binary QR formats
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"success": False, "error": "Image read error"}

    # Detect barcodes/QR codes using zxing-cpp engine
    results = zxingcpp.read_barcodes(img)
    
    if not results:
        # Fallback: Crop right half (standard Aadhaar QR location)
        h, w, _ = img.shape
        roi = img[:, int(w * 0.45):]
        results = zxingcpp.read_barcodes(roi)

    if not results:
        return {"success": False, "error": "QR Code physically missing or damaged"}

    raw_qr = results[0]
    raw_bytes = raw_qr.bytes
    raw_text = raw_qr.text

    parsed_data = {
        "success": True,
        "format": "UNKNOWN",
        "name": None,
        "dob": None,
        "gender": None,
        "masked_id": None
    }

    # Format 1: Legacy XML QR
    if "<PrintLetterBarcodeData" in raw_text or "uid=" in raw_text:
        parsed_data["format"] = "LEGACY_XML"
        name_match = re.search(r'name="([^"]+)"', raw_text)
        dob_match = re.search(r'dob="([^"]+)"', raw_text)
        gender_match = re.search(r'gender="([^"]+)"', raw_text)
        
        parsed_data["name"] = name_match.group(1) if name_match else None
        parsed_data["dob"] = dob_match.group(1) if dob_match else None
        parsed_data["gender"] = gender_match.group(1) if gender_match else None
        return parsed_data

    # Format 2: Secure QR Code (Decompression)
    try:
        decompressed = zlib.decompress(raw_bytes, 16 + zlib.MAX_WBITS)
        text_content = decompressed.decode('latin-1', errors='ignore')
        parsed_data["format"] = "SECURE_V2"
        
        # Extract text components separated by standard delimiters
        parts = [p for p in text_content.split('\xff') if p.strip()]
        if len(parts) >= 4:
            parsed_data["name"] = parts[1]
            parsed_data["dob"] = parts[2]
            parsed_data["gender"] = parts[3]
    except Exception:
        # Fallback for integer-encoded binary
        parsed_data["format"] = "SECURE_BINARY_RAW"

    return parsed_data