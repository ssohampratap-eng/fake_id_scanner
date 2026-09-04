import cv2
import numpy as np
from typing import Dict, Any


class AadhaarTypographyValidator:
    def __init__(self):
        # Baseline thresholds for clean, well-lit captures
        # NOTE: These are now MAD-based (Median Absolute Deviation), not
        # std/mean-based. MAD values are typically smaller in scale than
        # std-based variance, so these numbers were re-derived — do not
        # assume they mean the same thing as the old std-based thresholds.
        self.base_height_variance = 0.35
        self.base_aspect_variance = 0.18
        # Multipliers applied when glare/tilt are detected
        self.glare_tolerance_multiplier = 1.6
        self.tilt_tolerance_multiplier = 1.3

    def _detect_glare(self, gray: np.ndarray) -> bool:
        """
        Detects lamination glare / blown-out highlights that distort
        local contrast and inflate character variance on genuine
        physical cards. Returns True if significant glare is present.
        """
        bright_pixels = np.sum(gray > 240)
        glare_ratio = bright_pixels / gray.size
        if glare_ratio > 0.03:  # >3% of image is near-pure white
            return True

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var > 1500:  # empirically high for glare-heavy captures
            return True

        return False

    def _detect_tilt(self, gray: np.ndarray) -> bool:
        """
        Rough tilt/perspective-skew detector using Hough line angles.
        Off-axis phone captures distort character aspect ratios even
        without any tampering.
        """
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
        if lines is None:
            return False

        angles = []
        for line in lines[:50]:  # sample first 50 detected lines
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            angles.append(angle_deg)

        if not angles:
            return False

        deviation = np.std([a % 90 for a in angles])
        return deviation > 8.0  # degrees

    def verify_aadhaar_typography(self, image_path: str) -> Dict[str, Any]:
        img = cv2.imread(image_path)
        if img is None:
            return {"font_matched": True, "reasons": []}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Determine dynamic thresholds based on capture conditions
        has_glare = self._detect_glare(gray)
        has_tilt = self._detect_tilt(gray)

        height_threshold = self.base_height_variance
        aspect_threshold = self.base_aspect_variance
        condition_notes = []

        if has_glare:
            height_threshold *= self.glare_tolerance_multiplier
            aspect_threshold *= self.glare_tolerance_multiplier
            condition_notes.append("lamination glare detected")

        if has_tilt:
            height_threshold *= self.tilt_tolerance_multiplier
            aspect_threshold *= self.tilt_tolerance_multiplier
            condition_notes.append("camera tilt/perspective skew detected")

        # Otsu thresholding handles illumination gradients better than hard adaptive thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        char_heights = []
        char_ratios = []
        h_img, w_img = gray.shape[:2]

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter text glyphs strictly by size first
            if 12 < h < (h_img * 0.10) and 6 < w < (w_img * 0.12):
                aspect = w / float(h)

                # Exclude near-perfect squares — these are almost never real
                # text glyphs. They are typically QR code modules, emblem/
                # seal fragments, or logo artifacts that happen to fall
                # inside the same size window as text. Mixing these into
                # the character population contaminates the variance
                # calculation with a completely different aspect-ratio
                # distribution (~1.0) than real glyphs (~0.3-0.9).
                if 0.88 <= aspect <= 1.12:
                    continue

                char_heights.append(h)
                char_ratios.append(aspect)

        # Agar characters properly isolate na ho sakein, false alarm mat do
        if len(char_heights) < 15:
            return {"font_matched": True, "reasons": []}

        heights_arr = np.array(char_heights, dtype=np.float32)
        ratios_arr = np.array(char_ratios, dtype=np.float32)

        # Median Absolute Deviation (MAD) instead of std/mean — far less
        # sensitive to a handful of outlier contours (dust specks, laminate
        # scratch reflections, missed QR/logo fragments) than a plain
        # standard deviation, which real-world phone captures reliably
        # produce even on fully genuine cards.
        height_median = float(np.median(heights_arr))
        height_mad = float(np.median(np.abs(heights_arr - height_median)))
        height_var = height_mad / (height_median + 1e-5)

        ratio_median = float(np.median(ratios_arr))
        aspect_var = float(np.median(np.abs(ratios_arr - ratio_median)))

        warnings = []
        if height_var > height_threshold or aspect_var > aspect_threshold:
            note = f" ({', '.join(condition_notes)})" if condition_notes else ""
            warnings.append(
                f"Mixed font scale or baseline deviation detected "
                f"(Height Var: {height_var:.2f}/{height_threshold:.2f}, "
                f"Aspect Var: {aspect_var:.2f}/{aspect_threshold:.2f}){note}."
            )
            return {"font_matched": False, "reasons": warnings}

        return {"font_matched": True, "reasons": []}


def verify_font_and_layout(image_path: str) -> Dict[str, Any]:
    validator = AadhaarTypographyValidator()
    res = validator.verify_aadhaar_typography(image_path)
    return {
        "font_consistent": res["font_matched"],
        "detected_layout": "AADHAAR_STANDARD",
        "warnings": res["reasons"]
    }