import streamlit as st
from PIL import Image
from utils.predict import predict

st.set_page_config(
    page_title="FaceRead AI",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: #0a0a0f;
    color: #e8e4dc;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 3rem 2rem 4rem;
    max-width: 680px;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 3rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.18em;
    color: #9d8f6e;
    border: 1px solid #2a2820;
    padding: 6px 16px;
    border-radius: 100px;
    margin-bottom: 1.5rem;
    text-transform: uppercase;
}
.hero h1 {
    font-size: clamp(2.2rem, 6vw, 3.4rem);
    font-weight: 800;
    color: #f0ebe0;
    line-height: 1.08;
    letter-spacing: -0.03em;
    margin: 0 0 1rem;
}
.hero h1 span {
    color: #c8a96e;
}
.hero p {
    font-size: 15px;
    color: #7a7568;
    font-weight: 400;
    margin: 0;
    line-height: 1.6;
}
.hero-line {
    width: 1px;
    height: 48px;
    background: linear-gradient(to bottom, #2a2820, transparent);
    margin: 2rem auto 0;
}

/* ── Mode toggle ── */
.mode-tabs {
    display: flex;
    background: #111118;
    border: 1px solid #1e1e28;
    border-radius: 14px;
    padding: 5px;
    margin-bottom: 1.8rem;
    gap: 4px;
}
.mode-tab {
    flex: 1;
    text-align: center;
    padding: 10px 16px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.01em;
}
.mode-tab.active {
    background: #1e1e28;
    color: #c8a96e;
    border: 1px solid #2e2e3a;
}
.mode-tab.inactive {
    color: #4a4848;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #0e0e18 !important;
    border: 1.5px dashed #252530 !important;
    border-radius: 16px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3a3840 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    padding: 2.5rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #4a4848 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.05em;
}

/* ── Camera ── */
[data-testid="stCameraInput"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1.5px solid #1e1e28;
}
[data-testid="stCameraInputButton"] {
    background: #c8a96e !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* ── Image display ── */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid #1e1e28;
}

/* ── Radio ── */
[data-testid="stRadio"] {
    display: none;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #c8a96e !important;
}

/* ── Result cards ── */
.result-panel {
    margin-top: 1.8rem;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
.result-card {
    background: #0e0e18;
    border: 1px solid #1e1e28;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #c8a96e40, transparent);
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    color: #4a4848;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-value {
    font-size: 2rem;
    font-weight: 800;
    color: #c8a96e;
    letter-spacing: -0.02em;
    line-height: 1;
}
.result-value.gender {
    font-size: 1.6rem;
}

/* ── Analyze button ── */
.stButton > button {
    width: 100%;
    background: #c8a96e !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 24px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    margin-top: 1.2rem;
    transition: opacity 0.15s !important;
    text-transform: uppercase;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    background: #c8a96e !important;
}

/* ── Divider ── */
.section-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2rem 0 1.6rem;
}
.section-divider span {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    color: #2e2e3a;
    text-transform: uppercase;
    white-space: nowrap;
}
.section-divider::before,
.section-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a1a24;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid #141420;
}
.app-footer p {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #2e2e3a;
    letter-spacing: 0.12em;
    margin: 0;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: #0e0e18 !important;
    border: 1px solid #2a2820 !important;
    border-radius: 12px !important;
    color: #7a7568 !important;
}

/* ── Success ── */
.stSuccess {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">◈ Neural Vision · v2.0</div>
    <h1>Face<span>Read</span> AI</h1>
    <p>Upload a portrait or capture live from your camera.<br>
    Our model predicts age &amp; gender in seconds.</p>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)


# ── Mode selector (custom styled, backed by st.radio) ─────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "Upload Image"

col1, col2 = st.columns(2)
with col1:
    if st.button("⬆  Upload Image", use_container_width=True,
                 type="primary" if st.session_state.mode == "Upload Image" else "secondary"):
        st.session_state.mode = "Upload Image"
        st.rerun()
with col2:
    if st.button("◎  Use Camera", use_container_width=True,
                 type="primary" if st.session_state.mode == "Use Camera" else "secondary"):
        st.session_state.mode = "Use Camera"
        st.rerun()

# Re-style the mode buttons dynamically
active_left = "border: 1px solid #c8a96e; background: #1a160c;" if st.session_state.mode == "Upload Image" else ""
active_right = "border: 1px solid #c8a96e; background: #1a160c;" if st.session_state.mode == "Use Camera" else ""

st.markdown(f"""
<style>
div[data-testid="column"]:nth-child(1) .stButton > button {{
    background: #0e0e18 !important;
    color: {"#c8a96e" if st.session_state.mode == "Upload Image" else "#4a4848"} !important;
    {active_left}
    text-transform: none !important;
    letter-spacing: 0.02em !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    margin-top: 0 !important;
}}
div[data-testid="column"]:nth-child(2) .stButton > button {{
    background: #0e0e18 !important;
    color: {"#c8a96e" if st.session_state.mode == "Use Camera" else "#4a4848"} !important;
    {active_right}
    text-transform: none !important;
    letter-spacing: 0.02em !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    margin-top: 0 !important;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

# ── Input section ─────────────────────────────────────────────────────────────
image = None

if st.session_state.mode == "Upload Image":
    st.markdown("""
    <div class="section-divider"><span>Upload portrait</span></div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

else:
    st.markdown("""
    <div class="section-divider"><span>Live camera</span></div>
    """, unsafe_allow_html=True)

    camera_image = st.camera_input("", label_visibility="collapsed")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")


# ── Preview + Predict ─────────────────────────────────────────────────────────
if image is not None:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.image(image, use_container_width=True)

    predict_btn = st.button("Analyze Face →", use_container_width=True)

    if predict_btn:
        with st.spinner("Reading facial features..."):
            age, gender = predict(image)

        st.markdown(f"""
        <div class="result-panel">
            <div class="result-card">
                <div class="result-label">Predicted Age</div>
                <div class="result-value">{age}</div>
            </div>
            <div class="result-card">
                <div class="result-label">Gender</div>
                <div class="result-value gender">{gender}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.caption("Results are AI estimates and may not reflect true demographics.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <p>FACEREAD AI · POWERED BY DEEP VISION · ALL PROCESSING LOCAL</p>
</div>
""", unsafe_allow_html=True)