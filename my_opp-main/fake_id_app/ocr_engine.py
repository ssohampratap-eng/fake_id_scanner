import cv2
import re
import numpy as np
from typing import Dict, Any

try:
    import easyocr
    READER = easyocr.Reader(['en', 'hi'], gpu=False)
except Exception:
    READER = None

# Institutional / Header stopwords jo name candidate nahi ho sakte
GLOBAL_STOPWORDS = {
    "GOVERNMENT", "GOVT", "INDIA", "REPUBLIC", "OF", "UNIQUE",
    "IDENTIFICATION", "AUTHORITY", "AADHAAR", "MALE", "FEMALE", 
    "TRANSGENDER", "DOB", "YEAR", "BIRTH", "ISSUE", "ISSUED", 
    "DATE", "NAME", "ADDRESS", "VID", "HELP", "UIDAI", "WWW",
    "ENROLMENT", "MERAAADHAAR", "MERI", "PEHCHAN",
    "भारत", "सरकार", "भारतीय", "विशिष्ट", "पहचान", "प्राधिकरण",
    "आधार", "जन्म", "तारीख", "वर्ष", "लिंग", "पुरुष", "महिला", 
    "मेरा", "मेरी", "पिता", "पति"
}

def extract_document_fields(image_path: str) -> Dict[str, Any]:
    if READER is None:
        return {"extracted_text": "", "fields": {}, "warnings": ["EasyOCR module not initialized."]}

    img = cv2.imread(image_path)
    if img is None:
        return {"extracted_text": "", "fields": {}, "warnings": ["Image load failed from disk."]}

    results = READER.readtext(image_path)
    full_lines = []
    boxes_info = []

    for (bbox, text, prob) in results:
        clean_t = text.strip()
        if clean_t:
            full_lines.append(clean_t)
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            boxes_info.append({
                "text": clean_t,
                "box": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                "conf": float(prob)
            })

    full_text = " ".join(full_lines)
    fields = {}

    # -------------------------------------------------------------
    # 1. 12-DIGIT IDENTIFIER EXTRACTION (With OCR Noise Resilience)
    # -------------------------------------------------------------
    id_candidates = re.findall(r'[2-9]\d{3}\s?\d{4}\s?\d{4}', full_text)
    if id_candidates:
        scanned_id = id_candidates[-1].replace(" ", "")
        fields["id_number"] = {"value": scanned_id, "confidence": 0.90}
    else:
        clean_all = full_text.replace(" ", "")
        digits_only = re.findall(r'[2-9]\d{11}', clean_all)
        if digits_only:
            fields["id_number"] = {"value": digits_only[0], "confidence": 0.70}
        else:
            fields["id_number"] = {"value": "Not Detected", "confidence": 0.0}

    # -------------------------------------------------------------
    # 2. DOB EXTRACTION
    # -------------------------------------------------------------
    dob_match = re.search(r'\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b', full_text)
    if dob_match:
        fields["dob"] = {"value": dob_match.group(1).replace("-", "/").replace(".", "/"), "confidence": 0.90}
    else:
        yob_match = re.search(r'\b(19\d{2}|20\d{2})\b', full_text)
        if yob_match:
            fields["dob"] = {"value": f"01/01/{yob_match.group(1)}", "confidence": 0.70}
        else:
            fields["dob"] = {"value": "Not Detected", "confidence": 0.0}

    # -------------------------------------------------------------
    # 3. ISSUE DATE EXTRACTION (Margin text)
    # -------------------------------------------------------------
    issue_match = re.search(r'(?:issue|issued|जारी)[\s:]*(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})', full_text, re.IGNORECASE)
    if issue_match:
        fields["issue_date"] = {"value": issue_match.group(1).replace("-", "/").replace(".", "/"), "confidence": 0.85}
    else:
        fields["issue_date"] = {"value": "Not Detected", "confidence": 0.0}

    # -------------------------------------------------------------
    # 4. GENDER EXTRACTION
    # -------------------------------------------------------------
    if re.search(r'\b(FEMALE|महिला)\b', full_text, re.IGNORECASE):
        fields["gender"] = {"value": "FEMALE", "confidence": 0.95}
    elif re.search(r'\b(MALE|पुरुष)\b', full_text, re.IGNORECASE):
        fields["gender"] = {"value": "MALE", "confidence": 0.95}
    else:
        fields["gender"] = {"value": "Not Detected", "confidence": 0.0}

    # -------------------------------------------------------------
    # 5. SPATIAL GEOMETRIC NAME EXTRACTION (Add-on 1 Engine)
    # -------------------------------------------------------------
    def is_valid_name_candidate(raw_str: str) -> bool:
        # Numbers, dates, ya commas wale lines address/ID hoti hain, unhe filter karein
        if re.search(r'[\d,/\\:;=\-_]', raw_str):
            return False
        clean_words = [re.sub(r'[^a-zA-Z\u0900-\u097F]', '', w).upper() for w in raw_str.split()]
        clean_words = [w for w in clean_words if len(w) > 1]
        if not clean_words or len(clean_words) > 5:
            return False
        # Agar saare words institutional stoplist me hain, toh yeh name nahi hai
        if all(w in GLOBAL_STOPWORDS for w in clean_words):
            return False
        # Kam se kam ek word blacklist ke bahar hona chahiye
        non_stops = [w for w in clean_words if w not in GLOBAL_STOPWORDS]
        return len(non_stops) >= 1

    # Step A: DOB Anchor locate karein
    dob_anchor_box = None
    for b in boxes_info:
        t = b["text"].upper()
        if re.search(r'\b(DOB|जन्म|YEAR|BIRTH|\d{2}/\d{2}/\d{4})\b', t):
            dob_anchor_box = b["box"]
            break

    name_found = "Not Detected"
    name_conf = 0.0

    # Step B: Primary Extraction — DOB ke theek upar wala horizontal band scan karein
    if dob_anchor_box:
        dob_y_top = dob_anchor_box[1]
        dob_x_left = dob_anchor_box[0]
        
        candidates = []
        for b in boxes_info:
            b_x0, b_y0, b_x1, b_y1 = b["box"]
            # DOB line ke upar (vertical gap 5px se 140px ke andar)
            vertical_dist = dob_y_top - b_y1
            # Horizontal alignment DOB line ke lagbhag collinear honi chahiye
            horizontal_overlap = not (b_x1 < (dob_x_left - 150) or b_x0 > (dob_x_left + 450))

            if 2 < vertical_dist < 140 and horizontal_overlap:
                candidate_text = b["text"].strip()
                if is_valid_name_candidate(candidate_text):
                    candidates.append({
                        "text": candidate_text,
                        "conf": b["conf"],
                        "y_bottom": b_y1,
                        "dist": vertical_dist
                    })

        if candidates:
            # Jo line DOB ke sabse kareeb (sabse kam vertical distance par) ho, wahi name hai
            candidates.sort(key=lambda item: item["dist"])
            name_found = candidates[0]["text"]
            name_conf = candidates[0]["conf"]

    # Step C: Fallback (Agar DOB anchor missing ho ya detect na ho saka ho)
    if name_found == "Not Detected":
        # Reading order me sort karein: top-to-bottom
        sorted_by_y = sorted(boxes_info, key=lambda b: b["box"][1])
        h_img = img.shape[0]

        for b in sorted_by_y:
            y_top = b["box"][1]
            # Top 18% header area (jahan "भारत सरकार" hota hai) strictly chhod dein
            if y_top < (h_img * 0.18):
                continue
            # Lower 45% footer area (jahan numbers aur disclaimers hote hain) chhod dein
            if y_top > (h_img * 0.58):
                continue

            candidate_text = b["text"].strip()
            if is_valid_name_candidate(candidate_text):
                name_found = candidate_text
                name_conf = b["conf"]
                break

    fields["name"] = {"value": name_found, "confidence": round(name_conf, 2)}

    return {
        "extracted_text": full_text,
        "fields": fields,
        "boxes": boxes_info,
        "warnings": []
    }