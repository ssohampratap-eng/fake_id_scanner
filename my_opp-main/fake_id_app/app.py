import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="AI Fake ID Screener", page_icon="🛡️", layout="wide")
st.title("🛡️ Multi-Signal Document Fraud Screening System")
st.caption("SIH26188 | Ministry of Home Affairs | Privacy-Compliant Decision Support")
st.divider()

doc_type = st.sidebar.selectbox("Document Type:", ["Aadhaar Card"])
scan_mode = st.sidebar.radio("Input Source:", ["📁 Upload Image", "📷 Live Camera"])

# Mandatory Privacy Consent
consent = st.sidebar.checkbox(
    "I consent to temporary local processing of document for fraud screening.",
    value=True
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**System Scope:**\n"
    "- Single AI prediction par authenticity decide nahi hoti.\n"
    "- Checksum format consistency check karta hai, issuance/ownership nahi.\n"
    "- ELA compression anomaly indicator hai, conclusive proof nahi."
)

col1, col2 = st.columns([1, 1])
uploaded_img = None
temp_path = "temp_doc.jpg"

with col1:
    st.subheader("1. Document Input")
    if scan_mode == "📁 Upload Image":
        uploaded_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    else:
        uploaded_img = st.camera_input("Capture Live Photo")

    if uploaded_img:
        img = Image.open(uploaded_img).convert('RGB')
        img.save(temp_path)
        st.image(img, caption="Loaded Document (Local)", use_container_width=True)

with col2:
    st.subheader("2. Forensic Inspection Dashboard")
    if uploaded_img is None:
        st.info("👈 Pehle document upload ya camera se capture karein.")
    elif not consent:
        st.warning("⚠️ Processing ke liye user consent mandatory hai.")
    else:
        if st.button("🚀 Run Multi-Signal Verification", type="primary", use_container_width=True):
            with st.spinner("Analyzing Forensics, Quality, Checksum, Moiré Frequency & Semantics..."):
                try:
                    from orchestrator import analyze_document
                    res = analyze_document(temp_path, doc_type)
                except Exception as e:
                    res = {
                        "status": "ERROR",
                        "verdict": "PIPELINE_ERROR",
                        "flags": [f"Execution error: {str(e)}"],
                        "data": {}
                    }

            verdict = res.get("verdict", "UNKNOWN")
            if verdict == "VERIFIED_OFFLINE":
                st.success(f"✅ Status: **{verdict}** (Cryptographically Signed & Verified)")
            elif verdict == "REUPLOAD_REQUIRED":
                st.warning(f"⚠️ Status: **{verdict}** (Poor Document Capture Quality)")
            elif verdict == "HIGH_RISK_REJECT":
                st.error(f"🚨 Status: **{verdict}** (Decisive Fraud/Tamper Indicators Detected)")
            else:
                st.info(f"🔍 Status: **{verdict}** (Integrity Validated; Issuer Review Recommended)")

            if "trust_score" in res:
                st.metric("Fraud-Resistant Integrity Score", f"{res['trust_score']}%")

            tab1, tab2, tab3, tab4 = st.tabs(["📋 Masked Data", "🔬 Forensic Signals", "🔍 ELA Heatmap", "📐 Quality & Geometry"])

            with tab1:
                st.markdown("#### Masked Identity Fields")
                if res.get("data"):
                    for k, v in res["data"].items():
                        st.write(f"**{k.replace('_', ' ').title()}:** `{v}`")
                else:
                    st.write("No fields extracted.")

                st.markdown("#### Decision Findings & Audit Trail")
                for f in res.get("flags", []):
                    st.markdown(f"- {f}")

            with tab2:
                st.markdown("#### Physical Presentation & Frequency Forensics")
                c1, c2 = st.columns(2)
                with c1:
                    sa = res.get("screen_attack", {})
                    st.metric("Screen Replay Risk", sa.get("screen_recapture_risk", "UNKNOWN"))
                    st.caption(f"Frequency Energy: {sa.get('frequency_score', 'N/A')}")
                with c2:
                    cm = res.get("copy_move", {})
                    st.metric("Copy-Move Anomaly", cm.get("risk", "UNKNOWN"))
                    st.caption(f"Duplicate Clusters: {cm.get('duplicate_clusters', 0)}")

                meta = res.get("metadata", {})
                st.caption(f"**Software Metadata:** `{meta.get('software', 'None')}`")

            with tab3:
                st.markdown("#### JPEG Compression Anomaly Heatmap (ELA)")
                if res.get("heatmap_path") and os.path.exists(res["heatmap_path"]):
                    st.image(res["heatmap_path"], caption="Luminous regions indicate potential compression splicing anomalies", use_container_width=True)
                else:
                    st.write("Heatmap not available.")
                st.caption("Notice: ELA measures JPEG re-compression differences, not legal authenticity.")

            with tab4:
                qc = res.get("quality", {})
                st.write(f"**Quality Status:** `{qc.get('status', 'N/A')}` (Blur Score: {qc.get('blur_score', 0)})")
                st.write(f"**Detected Corners:** `{qc.get('document_corners_detected', False)}`")

            if os.path.exists(temp_path):
                os.remove(temp_path)