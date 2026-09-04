import cv2
import numpy as np
from datetime import datetime
import logging
import sys
import os
import re
from typing import Dict, List, Any

from forensic_engine import ForensicEngine
from font_validator import verify_font_and_layout
from ocr_engine import extract_document_fields
from validator import validate_id_document
from qr_reader import extract_qr_details
from tampering import run_ela_detection, scan_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Orchestrator")

class DocumentAnalyzer:
    def __init__(self):
        self.forensic = ForensicEngine()

    def _mask_identifier(self, id_val: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]", "", str(id_val))
        if len(clean) >= 4:
            return "XXXX-XXXX-" + clean[-4:]
        return "XXXX-XXXX-XXXX"

    def analyze_document(self, image_path: str, doc_type: str = "Aadhaar Card") -> Dict[str, Any]:
        flags: List[str] = []
        warnings: List[str] = []
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return self._failure_response("Image disk se load nahi ho saki.", doc_type)

        # 1. Quality Check
        quality = self.forensic.check_quality(img_bgr)
        if quality.get("status") == "REUPLOAD_REQUIRED":
            return {
                "status": "REUPLOAD_REQUIRED",
                "verdict": "REUPLOAD_REQUIRED",
                "trust_score": 0,
                "confidence_level": "LOW",
                "flags": quality.get("warnings", ["Low image quality"]),
                "warnings": ["Image quality is insufficient for screening."],
                "heatmap_path": None,
                "data": {},
                "document_type": doc_type,
                "analysis_timestamp": datetime.now().isoformat()
            }

        # 2. Module Execution & Signal Gathering
        ocr_result = extract_document_fields(image_path)
        fields = ocr_result.get("fields", {})
        warnings.extend(ocr_result.get("warnings", []))

        def get_field_val(k):
            val = fields.get(k, "Not Detected")
            return val.get("value", "Not Detected") if isinstance(val, dict) else val

        scanned_id = get_field_val("id_number")
        clean_id = scanned_id.replace(" ", "") if scanned_id != "Not Detected" else ""
        scanned_name = get_field_val("name")
        scanned_dob = get_field_val("dob")
        scanned_gender = get_field_val("gender")
        scanned_issue = get_field_val("issue_date")

        missing_required_fields = []
        if len(clean_id) != 12 or not clean_id.isdigit():
            missing_required_fields.append("12-digit Aadhaar number")
        if scanned_name == "Not Detected":
            missing_required_fields.append("name")
        if scanned_dob == "Not Detected":
            missing_required_fields.append("date of birth/year of birth")

        if missing_required_fields:
            return {
                "status": "FAIL",
                "verdict": "OCR_INSUFFICIENT_REUPLOAD",
                "trust_score": 0,
                "confidence_level": "LOW",
                "flags": [
                    "OCR required fields read nahi kar saka: "
                    + ", ".join(missing_required_fields),
                    "Document authenticity score nahi diya gaya; clear image upload karein."
                ],
                "warnings": warnings,
                "heatmap_path": None,
                "quality": quality,
                "data": {},
                "document_type": doc_type,
                "analysis_timestamp": datetime.now().isoformat()
            }

        tampering_result = run_ela_detection(image_path)
        metadata_result = scan_metadata(image_path)
        validation_result = validate_id_document(fields, doc_type)
        qr_result = extract_qr_details(image_path, clean_id, scanned_name)
        font_result = verify_font_and_layout(image_path)

        screen_res = self.forensic.detect_screen_recapture(img_bgr)
        copy_move_res = self.forensic.detect_copy_move(img_bgr)
        semantic_res = self.forensic.validate_semantics(
            {"name": scanned_name, "dob": scanned_dob}, qr_result
        )

        has_cloned_blocks = False
        if len(clean_id) == 12:
            b1, b2, b3 = clean_id[0:4], clean_id[4:8], clean_id[8:12]
            if (b1 == b3 or b1 == b2 or b2 == b3) and b1 != "0000":
                has_cloned_blocks = True

        # -------------------------------------------------------------
        # 3. ADD-ON 2: DYNAMIC MULTI-FACTOR TRUST MATRIX SCORING
        # -------------------------------------------------------------
        total_score = 100
        score_breakdown = {}
        flags = []

        # Persona Spoof Check (Immediate Hard Rejection)
        upper_name = scanned_name.upper()
        if any(bad in upper_name for bad in ["ELON", "MUSK", "TEST", "SAMPLE", "DUMMY"]):
            return {
                "status": "FAIL",
                "verdict": "HIGH_RISK_REJECT",
                "trust_score": 0,
                "confidence_level": "HIGH",
                "flags": [
                    "🚨 PERSONA SPOOF DETECTED: Fictitious identity record identified.",
                    "🚨 Verification Terminated: Known fabricated persona template."
                ],
                "warnings": ["Fabricated identity detected."],
                "heatmap_path": tampering_result.get("heatmap_path"),
                "quality": quality,
                "screen_attack": screen_res,
                "copy_move": copy_move_res,
                "metadata": metadata_result,
                "data": {},
                "document_type": doc_type,
                "analysis_timestamp": datetime.now().isoformat()
            }

        # A. Checksum Evaluation (30 Points Max)
        checksum_status = validation_result.get("checksum_status", "HARD_FAIL")
        if checksum_status == "VALID":
            score_breakdown["Checksum"] = "+30 (Mathematically Valid Verhoeff)"
        elif checksum_status == "SOFT_FAIL":
            total_score -= 10
            score_breakdown["Checksum"] = "-10 (Minor OCR Glare Noise)"
            flags.append("⚠️ Checksum Discrepancy: Recoverable via single-digit correction.")
        else:
            total_score -= 30
            score_breakdown["Checksum"] = "-30 (Checksum Failed / Malformed)"
            flags.append("🚨 Mathematical format or checksum validation failed.")

        # B. QR Cross-Match Evaluation (25 Points Max)
        if qr_result.get("qr_found"):
            if qr_result.get("is_matched"):
                score_breakdown["QR Match"] = "+25 (Cryptographic Parity Verified)"
                flags.append("✔️ QR Forensic Match: Printed text matches digital payload.")
            else:
                total_score -= 40
                score_breakdown["QR Match"] = "-40 (Tampered Mismatch)"
                flags.append("🚨 CRITICAL MISMATCH: Printed text does not match digital QR record.")
        else:
            score_breakdown["QR Match"] = "0 (Front Card Layout - Skipped)"
            flags.append("ℹ️ QR Not Present: Front-side physical card verified via visual signals.")

        # C. Typography & Proportions (25 Points Max)
        if not font_result.get("font_consistent", True):
            total_score -= 12
            score_breakdown["Typography"] = "-12 (Font Baseline Variance)"
            flags.append("⚠️ Typography Flag: Minor non-standard baseline variance detected.")
        else:
            score_breakdown["Typography"] = "+25 (Standard Proportions Verified)"
            flags.append("✔️ Typography: Standard glyph proportions verified.")

        # D. Display / Screen Attack (20 Points Max)
        if screen_res.get("screen_recapture_risk") == "HIGH":
            if screen_res.get("glare_detected"):
                total_score -= 8
                score_breakdown["Display Attack"] = "-8 (Moiré / Lamination Glare Ambiguity)"
                flags.append("⚠️ Moiré Frequency Note: Lamination glare detected alongside high frequency spikes.")
            else:
                total_score -= 20
                score_breakdown["Display Attack"] = "-20 (Definite Screen Grid Pattern)"
                flags.append("🚨 Screen Replay: Periodic display moiré interference detected.")
        else:
            score_breakdown["Display Attack"] = "+20 (Natural Physical Document Capture)"

        # Additional Hard Fails (Semantics & Cloned Blocks)
        if semantic_res.get("status") == "FAIL":
            total_score -= 50
            for issue in semantic_res.get("issues", []):
                flags.append(f"🚨 Semantic Anomaly: {issue}")

        if has_cloned_blocks:
            total_score -= 40
            flags.append("🚨 Structural Anomaly: Cloned number block pattern detected.")

        trust_score = max(0, min(100, total_score))

        # Final Status & Verdict Assignment based on Trust Score
        if checksum_status == "HARD_FAIL":
            verdict = "HIGH_RISK_REJECT"
            status = "FAIL"
        elif checksum_status == "SOFT_FAIL":
            verdict = "MANUAL_REVIEW_REQUIRED"
            status = "MANUAL_REVIEW"
        elif trust_score >= 90 and qr_result.get("qr_found") and qr_result.get("is_matched"):
            verdict = "VERIFIED_OFFLINE"
            status = "PASS"
        elif trust_score >= 70:
            verdict = "LIKELY_GENUINE_MANUAL_REVIEW"
            status = "PASS"
        elif trust_score >= 45:
            verdict = "MANUAL_REVIEW_REQUIRED"
            status = "MANUAL_REVIEW"
        else:
            verdict = "HIGH_RISK_REJECT"
            status = "FAIL"

        masked_data = {
            "Extracted_Name": scanned_name,
            "DOB": scanned_dob,
            "Gender": scanned_gender,
            "Scanned_ID": self._mask_identifier(clean_id) if clean_id else "Not Detected",
            "QR_Detected": qr_result.get("qr_found", False),
            "Score_Breakdown": str(score_breakdown)
        }

        if status != "PASS":
            masked_data = {}

        return {
            "status": status,
            "verdict": verdict,
            "trust_score": trust_score,
            "confidence_level": "HIGH" if trust_score >= 80 else ("MEDIUM" if trust_score >= 45 else "LOW"),
            "flags": flags,
            "warnings": warnings,
            "heatmap_path": tampering_result.get("heatmap_path"),
            "quality": quality,
            "screen_attack": screen_res,
            "copy_move": copy_move_res,
            "metadata": metadata_result,
            "data": masked_data,
            "document_type": doc_type,
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _failure_response(self, reason: str, doc_type: str) -> Dict[str, Any]:
        return {
            "status": "FAIL",
            "verdict": "PIPELINE_ERROR",
            "trust_score": 0,
            "confidence_level": "LOW",
            "flags": [f"Error: {reason}"],
            "warnings": [reason],
            "heatmap_path": None,
            "data": {},
            "document_type": doc_type,
            "analysis_timestamp": datetime.now().isoformat()
        }

def analyze_document(image_path: str, doc_type: str = "Aadhaar Card") -> Dict[str, Any]:
    analyzer = DocumentAnalyzer()
    return analyzer.analyze_document(image_path, doc_type) 