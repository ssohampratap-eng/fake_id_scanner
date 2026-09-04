import cv2
import numpy as np
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, Any


class ForensicEngine:
    def __init__(self, min_blur: float = 65.0, min_dim: int = 400):
        self.min_blur = min_blur
        self.min_dim = min_dim
        self.orb = cv2.ORB_create(nfeatures=1200)
        # Base moiré thresholds for clean, glare-free captures
        self.base_moire_high = 127.0
        self.base_moire_medium = 116.0
        self.glare_tolerance_offset = 12.0  # added to thresholds when glare detected

    def _detect_glare(self, gray: np.ndarray) -> bool:
        """Shared glare heuristic — blown-out highlights + sharp local contrast spikes."""
        bright_ratio = np.sum(gray > 240) / gray.size
        if bright_ratio > 0.03:
            return True
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var > 1500

    def check_quality(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Module 1: Image Quality Check (Blur, Glare, Resolution, Corners)."""
        if img_bgr is None or img_bgr.size == 0:
            return {
                "status": "REUPLOAD_REQUIRED",
                "blur_score": 0.0,
                "brightness_score": 0.0,
                "contrast_score": 0.0,
                "resolution": [0, 0],
                "document_corners_detected": False,
                "warnings": ["Image read nahi ho saki ya file empty hai."],
                "message": "Valid document upload karein."
            }

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        warnings = []

        if h < self.min_dim or w < self.min_dim:
            warnings.append(f"Image resolution [{w}x{h}] minimum ({self.min_dim}px) se kam hai.")

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < self.min_blur:
            warnings.append(f"Image blurry hai (Score: {blur_score:.1f} < {self.min_blur}).")

        mean_brightness = float(np.mean(gray))
        if mean_brightness < 35.0:
            warnings.append("Image bahut dark hai (underexposed).")
        elif mean_brightness > 230.0:
            warnings.append("Flash glare ya overexposure detect hua hai.")

        contrast_score = float(np.std(gray))
        if contrast_score < 20.0:
            warnings.append("Document me contrast bahut kam hai.")

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        corners_found = False
        if contours:
            largest = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.04 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > (0.20 * h * w):
                corners_found = True

        status = "PASS"
        if blur_score < self.min_blur or mean_brightness < 35.0 or mean_brightness > 230.0:
            status = "REUPLOAD_REQUIRED"
        elif len(warnings) > 0:
            status = "WARNING"

        return {
            "status": status,
            "blur_score": round(blur_score, 2),
            "brightness_score": round(mean_brightness, 2),
            "contrast_score": round(contrast_score, 2),
            "resolution": [int(w), int(h)],
            "document_corners_detected": corners_found,
            "warnings": warnings,
            "message": "Image quality sufficient hai." if status == "PASS" else "Quality issues detect hue hain."
        }

    def detect_screen_recapture(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Module 6E: Moiré pattern & Frequency domain screen recapture check.
        
        Thresholds widen when lamination glare is detected, since glare on
        plastic cards genuinely injects high-frequency FFT energy that is
        otherwise indistinguishable from screen-pixel-grid moiré.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        has_glare = self._detect_glare(gray)
        high_threshold = self.base_moire_high
        medium_threshold = self.base_moire_medium
        if has_glare:
            high_threshold += self.glare_tolerance_offset
            medium_threshold += self.glare_tolerance_offset

        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-5)

        h, w = gray.shape
        cx, cy = w // 2, h // 2
        r = min(h, w) // 4
        mask = np.ones((h, w), np.uint8)
        cv2.circle(mask, (cx, cy), r, 0, -1)

        high_freq = float(np.mean(magnitude[mask == 1]))
        risk = "HIGH" if high_freq > high_threshold else ("MEDIUM" if high_freq > medium_threshold else "LOW")

        evidence = "Natural physical capture"
        if risk != "LOW":
            evidence = "Suspicious periodic moire/display-grid pattern"
            if has_glare:
                evidence += " (note: lamination glare present — tolerance was widened, still exceeded)"

        return {
            "screen_recapture_risk": risk,
            "frequency_score": round(high_freq, 2),
            "glare_detected": has_glare,
            "threshold_used": round(high_threshold, 1),
            "evidence": evidence
        }

    def detect_copy_move(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Module 6C: Copy-Move Feature Clustering (Duplicated patches).
        
        Filters out matches consistent with Aadhaar's repeating guilloche/
        watermark security background — those produce genuine repeated
        keypoints that aren't evidence of tampering.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        kp, des = self.orb.detectAndCompute(gray, None)

        if des is None or len(kp) < 20:
            return {"status": "PASS", "risk": "LOW", "duplicate_clusters": 0}

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des, des, k=2)

        cloned_points = []
        displacement_vectors = []

        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.72 * n.distance and m.queryIdx != m.trainIdx:
                    pt1 = kp[m.queryIdx].pt
                    pt2 = kp[m.trainIdx].pt
                    dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                    if dist > 35:
                        cloned_points.append((pt1, pt2))
                        displacement_vectors.append((pt2[0] - pt1[0], pt2[1] - pt1[1]))

        # Real copy-move forgery = one displacement vector repeated many times
        # (a whole patch moved from A to B). A repeating security pattern
        # instead produces MANY DIFFERENT displacement vectors scattered
        # across the card (each repeat unit pairs with its neighbors).
        # So: cluster displacement vectors and require a dominant cluster.
        genuine_forgery_clusters = 0
        if displacement_vectors:
            vectors = np.array(displacement_vectors)
            # Round to nearest 10px bucket to cluster near-identical displacements
            buckets = {}
            for dx, dy in vectors:
                key = (round(dx / 10) * 10, round(dy / 10) * 10)
                buckets[key] = buckets.get(key, 0) + 1
            largest_cluster = max(buckets.values()) if buckets else 0
            genuine_forgery_clusters = largest_cluster

        # A real copy-paste forgery shows a strong single dominant displacement
        # cluster. Scattered small clusters (typical of security-pattern
        # texture) should NOT trigger high risk even if total match count is high.
        risk = "LOW"
        if genuine_forgery_clusters > 10:
            risk = "HIGH"
        elif genuine_forgery_clusters > 4:
            risk = "MEDIUM"

        return {
            "status": "SUSPICIOUS" if risk != "LOW" else "PASS",
            "risk": risk,
            "duplicate_clusters": len(cloned_points),
            "dominant_displacement_cluster_size": genuine_forgery_clusters,
            "message": (f"{genuine_forgery_clusters} keypoints share the same displacement vector "
                        f"— consistent with a moved/duplicated patch.") if risk != "LOW"
                       else f"{len(cloned_points)} scattered matches found, consistent with repeating "
                            f"security-pattern texture rather than tampering."
        }

    def validate_semantics(self, fields: Dict[str, Any], qr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Module 4B: Semantic consistency & fuzzy QR-to-OCR cross-validation."""
        issues = []
        dob_str = fields.get("dob", "")
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                dob_dt = datetime.strptime(dob_str, fmt)
                if dob_dt > datetime.now():
                    issues.append("Date of birth future me nahi ho sakti.")
                if (datetime.now().year - dob_dt.year) > 115:
                    issues.append("Date of birth unrealistic age (>115) indicate kar rahi hai.")
                break
            except ValueError:
                continue

        if qr_data and qr_data.get("qr_found"):
            raw_qr = qr_data.get("raw_data", "").upper().replace(" ", "")
            clean_name = fields.get("name", "").upper().replace(" ", "")
            if clean_name and clean_name != "NOTDETECTED" and len(clean_name) > 3:
                if clean_name not in raw_qr and raw_qr not in clean_name:
                    ratio = SequenceMatcher(None, clean_name, raw_qr).ratio()
                    if ratio < 0.60:
                        issues.append("Identity Mismatch: Visible name digital QR payload se match nahi karta.")

        return {"status": "FAIL" if len(issues) > 0 else "PASS", "issues": issues}