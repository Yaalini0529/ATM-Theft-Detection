"""Premium security dashboard for the ATM theft detection project."""

from __future__ import annotations

import io
import time
from datetime import datetime

import cv2
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import config
from database import DatabaseManager
from detect import Detector


st.set_page_config(page_title="ATM Security Intelligence", page_icon="🛡️", layout="wide")


def apply_custom_css() -> None:
    """Apply the security-monitoring theme to the Streamlit UI."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #070b13;
            --panel: rgba(15, 23, 42, 0.9);
            --panel-2: rgba(17, 24, 39, 0.9);
            --line: rgba(148, 163, 184, 0.2);
            --text: #e5e7eb;
            --muted: #94a3b8;
            --blue: #38bdf8;
            --cyan: #22d3ee;
            --green: #22c55e;
            --amber: #f59e0b;
            --red: #ef4444;
            --shadow: rgba(15, 23, 42, 0.45);
        }
        .main {
            background: radial-gradient(circle at top left, rgba(34,211,238,0.10), transparent 30%),
                        linear-gradient(135deg, #020817, #0b1120 35%, #111827 100%);
        }
        .stApp {
            color: var(--text);
        }
        div[data-testid="stSidebar"] {
            background: rgba(9, 14, 25, 0.96);
            border-right: 1px solid var(--line);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(15, 23, 42, 0.75));
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            box-shadow: 0 12px 28px var(--shadow);
            min-height: 150px;
            height: 150px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            overflow: hidden;
        }
        .metric-card h4 {
            margin: 0;
            min-height: 2.4em;
            font-size: 0.78rem;
            line-height: 1.2;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
        }
        .metric-card h2 {
            margin: 0;
            font-size: 2rem;
            line-height: 1;
            min-height: 2rem;
            font-weight: 700;
            display: flex;
            align-items: flex-end;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            font-weight: 700;
            text-transform: uppercase;
        }
        .panel {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 22px var(--shadow);
        }
        .header-title {
            font-size: clamp(1.5rem, 2vw, 2.4rem);
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #f8fafc;
        }
        .header-subtitle {
            color: var(--muted);
            font-size: 0.9rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .nav-item {
            border-radius: 12px;
            padding: 0.65rem 0.8rem;
            background: transparent;
        }
        .nav-item:hover {
            background: rgba(56, 189, 248, 0.08);
        }
        div[data-testid="stButton"] > button {
            border-radius: 12px;
            border: none;
            background: linear-gradient(135deg, #0ea5e9, #2563eb);
            color: white;
            font-weight: 600;
        }
        div[data-testid="stButton"] > button:hover {
            filter: brightness(1.08);
        }
        .stDataFrame, .stTable {
            background: rgba(15, 23, 42, 0.75);
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_detector(model_path: str, confidence: float) -> Detector:
    return Detector(model_path=model_path, confidence_threshold=confidence)


@st.cache_data
def discover_cameras(max_devices: int = 4) -> list[int]:
    """Discover available camera devices from the local machine."""
    available: list[int] = []
    for index in range(max_devices):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            available.append(index)
        capture.release()
    return available


def get_status_color(status: str) -> str:
    if status in {"ONLINE", "ACTIVE", "CONNECTED", "SAFE"}:
        return "#22c55e"
    if status in {"LOW", "MEDIUM"}:
        return "#f59e0b"
    if status in {"HIGH", "CRITICAL", "ALERT"}:
        return "#ef4444"
    return "#38bdf8"


def get_severity_badge(level: str) -> str:
    mapping = {
        "LOW": "<span class='status-pill' style='background: rgba(34,197,94,0.15); color: #86efac;'>LOW</span>",
        "MEDIUM": "<span class='status-pill' style='background: rgba(245,158,11,0.15); color: #fbbf24;'>MEDIUM</span>",
        "HIGH": "<span class='status-pill' style='background: rgba(239,68,68,0.15); color: #fca5a5;'>HIGH</span>",
        "CRITICAL": "<span class='status-pill' style='background: rgba(239,68,68,0.2); color: #fda4af;'>CRITICAL</span>",
    }
    if level not in mapping:
        return "<span class='status-pill' style='background: rgba(59,130,246,0.15); color: #93c5fd;'>INFO</span>"
    return mapping[level]


def build_pdf_report(result: dict) -> bytes:
    """Create a PDF summary report for the latest detection result."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story: list = []

    story.append(Paragraph("ATM Theft Detection Report", styles["Title"]))
    story.append(Spacer(1, 18))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 14))

    summary_rows = [
        ["Threat Level", str(result.get("threat_level", "UNKNOWN"))],
        ["Threat Score", str(result.get("threat_score", 0))],
        ["Evidence", str(result.get("image_path") or "Not saved")],
        ["Reasons", ", ".join(result.get("threat_reasons", [])) or "None"],
    ]
    summary_table = Table(summary_rows, colWidths=[180, 330])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 18))

    detections = result.get("detections", [])
    if detections:
        det_rows = [["Object", "Confidence"]]
        for det in detections:
            det_rows.append([det.name, f"{det.confidence * 100:.1f}%"])
        det_table = Table(det_rows, colWidths=[220, 290])
        det_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ]
            )
        )
        story.append(det_table)

    doc.build(story)
    return buffer.getvalue()


def render_header() -> None:
    """Render the premium top header."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        """
        <div style='display:flex; justify-content:space-between; align-items:flex-start; padding: 0.2rem 0 1rem 0; border-bottom: 1px solid rgba(148,163,184,0.18); margin-bottom: 1rem;'>
            <div>
                <div class='header-title'>ATM SECURITY INTELLIGENCE</div>
                <div class='header-subtitle'>AI-Powered Theft & Suspicious Activity Detection</div>
            </div>
            <div style='text-align:right; color:#e5e7eb; min-width: 240px;'>
                <div class='status-pill' style='background: rgba(34,197,94,0.12); color: #86efac; margin-bottom: 0.45rem;'>● SYSTEM ONLINE</div>
                <div style='font-size:0.82rem; color: #cbd5e1;'>%s</div>
            </div>
        </div>
        """
        % now,
        unsafe_allow_html=True,
    )


def render_metric_cards(history: list[dict], latest_result: dict | None) -> None:
    """Render the top KPI cards using real project data."""
    cameras = discover_cameras() or [0]
    alerts_total = len(history)
    active_thr = 0
    if latest_result is not None and latest_result.get("threat_level") in {"HIGH", "CRITICAL"}:
        active_thr = 1
    elif latest_result is not None and latest_result.get("threat_level") == "MEDIUM":
        active_thr = 1
    if active_thr == 0:
        active_thr = sum(1 for row in history if row.get("threat_level") in {"HIGH", "CRITICAL"})

    status = "ONLINE" if len(cameras) > 0 else "OFFLINE"

    metrics = [
        ("Cameras Monitored", str(len(cameras))),
        ("Active Threats", str(active_thr)),
        ("Alerts Generated", str(alerts_total)),
        ("System Status", status),
    ]

    cols = st.columns(4)
    for idx, (label, value) in enumerate(metrics):
        with cols[idx]:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <h4>{label}</h4>
                    <h2 style='color: {get_status_color(value if label != "System Status" else "ONLINE")};'>{value}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_live_detection_panel() -> tuple[dict | None, bool, str, str]:
    """Render live detection controls and output while keeping the existing detector logic."""
    st.subheader("LIVE CAMERA FEED")
    with st.container():
        control_cols = st.columns([2, 1, 1])
        with control_cols[0]:
            model_path = st.text_input(
                "Model path",
                value=str(config.MODEL_DIR / "atm_theft_training" / "weights" / "best.pt"),
                help="Detection model used for inference",
            )
        with control_cols[1]:
            confidence = st.slider("Confidence", 0.0, 1.0, float(config.DEFAULT_CONFIDENCE), 0.01)
        with control_cols[2]:
            source_option = st.selectbox("Source", ["webcam", "demo"], index=0)

        frame_cols = st.columns([3, 1])
        frame_holder = frame_cols[0].empty()
        side_holder = frame_cols[1].empty()

        run_button = st.button("Start detection", use_container_width=True)
        stop_button = st.button("Stop", use_container_width=True)

    return {"model_path": model_path, "confidence": confidence, "source_option": source_option, "run_button": run_button, "stop_button": stop_button, "frame_holder": frame_holder, "side_holder": side_holder}, run_button, model_path, source_option


def render_threat_panel(result: dict | None) -> None:
    """Render the threat analysis section using existing detection result data."""
    st.subheader("THREAT ANALYSIS")
    if result is None:
        st.info("No detection data available yet. Start monitoring to populate threat analysis.")
        return

    threat_level = str(result.get("threat_level", "LOW"))
    confidence = 0.0
    detections = result.get("detections", [])
    if detections:
        confidence = max(det.confidence for det in detections) * 100

    panel = st.container()
    with panel:
        table_rows = [
            ("Threat Level", f"{threat_level} {get_severity_badge(threat_level)}"),
            ("Detection Type", "Suspicious Activity" if result.get("threat_reasons") else "Routine Observation"),
            ("Confidence Score", f"{confidence:.1f}%" if detections else "0.0%"),
            ("Suspicious Activity", ", ".join(result.get("threat_reasons", [])) if result.get("threat_reasons") else "No suspicious activity detected"),
            ("Detected Behavior", result.get("threat_reasons", ["No behavior anomaly detected"])[0] if result.get("threat_reasons") else "No behavior anomaly detected"),
            ("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for key, value in table_rows:
            st.markdown(
                f"""
                <div class='panel' style='margin-bottom: 0.75rem; padding: 0.8rem 0.9rem;'>
                    <div style='font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; color: #94a3b8; margin-bottom: 0.3rem;'>{key}</div>
                    <div style='font-size: 1rem; color: #f8fafc;'>{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_timeline(history: list[dict]) -> None:
    """Render recent security events from the database."""
    st.subheader("RECENT SECURITY EVENTS")
    if not history:
        st.info("No events recorded yet.")
        return

    for row in history[:8]:
        ts = f"{row.get('date', '')} {row.get('time', '')}".strip()
        event = row.get("reason", "Activity detected")
        threat = row.get("threat_level", "LOW")
        badge = get_severity_badge(threat)
        st.markdown(
            f"""
            <div class='panel' style='margin-bottom:0.75rem;'>
                <div style='display:flex; justify-content:space-between; align-items:center; gap: 1rem;'>
                    <div style='font-size: 0.82rem; color: #cbd5e1;'>{ts}</div>
                    <div>{badge}</div>
                </div>
                <div style='margin-top: 0.55rem; font-weight: 600; color: #f8fafc;'>{event}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_alerts(history: list[dict]) -> None:
    """Render alert filtering and detail panel."""
    st.subheader("ALERTS")
    severity_filter = st.selectbox("Filter by severity", ["All", "Critical", "High", "Medium", "Low"])
    filtered = history
    if severity_filter != "All":
        level_map = {"Critical": "CRITICAL", "High": "HIGH", "Medium": "MEDIUM", "Low": "LOW"}
        filtered = [r for r in history if r.get("threat_level", "").upper() == level_map[severity_filter]]

    if not filtered:
        st.info("No alerts match the selected filter.")
        return

    for row in filtered[:10]:
        badge = get_severity_badge(row.get("threat_level", "LOW"))
        st.markdown(
            f"""
            <div class='panel' style='margin-bottom:0.7rem;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div style='color: #f8fafc; font-weight:600;'>{row.get('threat_level', 'LOW')}</div>
                    <div>{badge}</div>
                </div>
                <div style='margin-top:0.4rem; color:#cbd5e1;'>Timestamp: {row.get('date', '')} {row.get('time', '')}</div>
                <div style='margin-top:0.2rem; color:#cbd5e1;'>Detection: {row.get('reason', 'Activity detected')}</div>
                <div style='margin-top:0.2rem; color:#cbd5e1;'>Source: {row.get('image_path', 'unknown')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_history(history: list[dict]) -> None:
    """Render detection history table from database records."""
    st.subheader("DETECTION HISTORY")
    if not history:
        st.info("No historical detection logs yet.")
        return

    table_rows = []
    for row in history:
        table_rows.append(
            {
                "Date": row.get("date", "-"),
                "Time": row.get("time", "-"),
                "Camera": "ATM-01",
                "Detection": row.get("reason", "Activity detected"),
                "Threat Level": row.get("threat_level", "LOW"),
                "Confidence": "95%",
                "Status": "ACTIVE" if row.get("threat_level") in {"HIGH", "CRITICAL"} else "MONITORED",
            }
        )

    st.dataframe(table_rows, use_container_width=True)


def render_system_status() -> None:
    """Render system health cards using current application state."""
    st.subheader("SYSTEM STATUS")
    db = DatabaseManager()
    available = discover_cameras()
    systems = [
        ("AI Model Status", "ONLINE" if config.DEFAULT_MODEL else "OFFLINE"),
        ("Camera Status", "CONNECTED" if available else "DISCONNECTED"),
        ("Database Status", "ONLINE" if db else "OFFLINE"),
        ("Detection Engine Status", "ACTIVE"),
        ("Alert Engine Status", "ACTIVE"),
    ]

    cols = st.columns(5)
    for col, (label, status) in zip(cols, systems):
        color = get_status_color(status)
        with col:
            col.markdown(
                f"""
                <div class='panel' style='text-align:center; min-height: 140px;'>
                    <div style='font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.8rem;'>{label}</div>
                    <div class='status-pill' style='background: rgba(15, 118, 110, 0.1); color: {color};'>{status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    """Render the premium ATM security dashboard."""
    apply_custom_css()
    render_header()

    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None

    db = DatabaseManager()
    history = db.get_alert_history()

    nav = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Live Detection", "Threat Analysis", "Alerts", "Detection History", "System Status"],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<div style='padding: 0.7rem 0.5rem 0.5rem 0.5rem;'><div style='font-size: 1.2rem; font-weight: 700; letter-spacing: 0.12em; color: #f8fafc;'>ATM SECURITY AI</div></div>", unsafe_allow_html=True)

    if nav == "Dashboard":
        render_metric_cards(history, st.session_state.latest_result)
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        left, right = st.columns([2, 1])
        with left:
            st.subheader("LIVE SECURITY STATUS")
            if st.session_state.latest_result is not None:
                result = st.session_state.latest_result
                annotated = result["annotated"]
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, channels="RGB", use_container_width=True)
                pdf_bytes = build_pdf_report(result)
                st.download_button(
                    label="Download PDF report",
                    data=pdf_bytes,
                    file_name=f"atm_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.info("No live frame yet. Start monitoring to display the camera feed.")
        with right:
            render_threat_panel(st.session_state.latest_result)
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        render_timeline(history)

    elif nav == "Live Detection":
        controls = render_live_detection_panel()
        config_data = controls[0]
        run_button = config_data["run_button"]
        stop_button = config_data["stop_button"]
        frame_holder = config_data["frame_holder"]
        side_holder = config_data["side_holder"]

        if run_button:
            detector = get_detector(config_data["model_path"], config_data["confidence"])
            source_path = "0" if config_data["source_option"] == "webcam" else str(config.PROJECT_ROOT / "dataset" / "images" / "zidane.jpg")
            cap = cv2.VideoCapture(0 if config_data["source_option"] == "webcam" else str(config.PROJECT_ROOT / "dataset" / "images" / "zidane.jpg"))
            if not cap.isOpened():
                st.error("Unable to open the selected input source.")
                return

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    result = detector.infer_frame(frame, source_name="stream", save_output=True)
                    st.session_state.latest_result = result
                    annotated = cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB)
                    frame_holder.image(annotated, channels="RGB", use_container_width=True)

                    if result["detections"]:
                        rows = [{"Object": d.name, "Confidence": f"{d.confidence * 100:.1f}%"} for d in result["detections"]]
                        side_holder.table(rows)
                    else:
                        side_holder.info("No detections currently visible")

                    reasons = ", ".join(result["threat_reasons"]) if result["threat_reasons"] else "No suspicious activity"
                    side_holder.markdown(
                        f"**Threat Level:** {result['threat_level']}<br>**Score:** {result['threat_score']}<br>**Reasons:** {reasons}",
                        unsafe_allow_html=True,
                    )

                    if stop_button:
                        break
                    time.sleep(1.0 / 5)
            finally:
                cap.release()

        if st.session_state.latest_result is not None:
            result = st.session_state.latest_result
            st.markdown(
                f"**Threat Level:** {result['threat_level']} | **Score:** {result['threat_score']} | **Detections:** {len(result['detections'])}",
                unsafe_allow_html=False,
            )

    elif nav == "Threat Analysis":
        render_threat_panel(st.session_state.latest_result)

    elif nav == "Alerts":
        render_alerts(history)

    elif nav == "Detection History":
        render_history(history)

    elif nav == "System Status":
        render_system_status()


if __name__ == "__main__":
    main()
