import cv2
import numpy as np
import os
from PIL import Image, ImageChops, ImageEnhance
from typing import Dict, Any

def run_ela_detection(image_path: str, quality: int = 90, threshold: float = 8.5) -> Dict[str, Any]:
    temp_ela = "temp_ela.jpg"
    heatmap_path = "tamper_heatmap.jpg"
    try:
        im = Image.open(image_path).convert('RGB')
        im.save(temp_ela, 'JPEG', quality=quality)
        resaved = Image.open(temp_ela)

        diff = ImageChops.difference(im, resaved)
        scale = ImageEnhance.Brightness(diff)
        enhanced = scale.enhance(10.0)
        enhanced.save(heatmap_path)

        diff_arr = np.array(diff)
        tamper_score = float(np.mean(diff_arr))

        if os.path.exists(temp_ela):
            os.remove(temp_ela)

        return {
            "ela_available": True,
            "ela_score": round(tamper_score, 2),
            "is_anomaly": tamper_score > threshold,
            "heatmap_path": heatmap_path,
            "message": "Compression anomaly detected." if tamper_score > threshold else "No strong JPEG-compression anomaly observed.",
            "limitation": "ELA is an anomaly indicator, not proof of authenticity."
        }
    except Exception as e:
        return {
            "ela_available": False,
            "ela_score": 0.0,
            "is_anomaly": False,
            "heatmap_path": None,
            "message": f"ELA processing failed: {str(e)}",
            "limitation": "ELA skipped."
        }

def scan_metadata(image_path: str) -> Dict[str, Any]:
    software_found = None
    try:
        im = Image.open(image_path)
        exif = im.getexif()
        if exif:
            software_found = exif.get(305, None)
    except Exception:
        pass

    is_suspicious = False
    if software_found:
        s_upper = str(software_found).upper()
        if any(tool in s_upper for tool in ["PHOTOSHOP", "CANVA", "GIMP", "PICSART"]):
            is_suspicious = True

    return {
        "status": "WARNING" if is_suspicious else "PASS",
        "software": str(software_found) if software_found else "None/Stripped",
        "warning": f"Editing software ({software_found}) metadata me mila." if is_suspicious else "No suspicious editing tool metadata found."
    }