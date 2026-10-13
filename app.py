import streamlit as st
import os
from dotenv import load_dotenv

from core.pdf_loader import extract_pdf_data
from core.rag_engine import build_rag_index, retrieve_relevant_context
from core.personas import PERSONA_PROMPTS, SPICY_LEVELS
from core.gemini_service import (
    generate_instant_audit_stream,
    generate_chat_turn_stream,
    generate_job_match_stream,
    get_available_models,
)

# Load file .env jika ada
load_dotenv()

@st.cache_data(show_spinner="Mengambil daftar model Gemini...", ttl=600)
def fetch_gemini_models_data(key: str) -> dict:
    models, is_dynamic, error_msg = get_available_models(key)
    return {
        "models": models,
        "is_dynamic": is_dynamic,
        "error_msg": error_msg,
    }

def stream_with_status_clear(gen, status_box):
    """
    Menampilkan placeholder status loading sampai chunk pertama diterima dari LLM,
    lalu secara mulus membersihkan placeholder agar st.write_stream langsung
    menampilkan teks secara realtime token-by-token.
    """
    has_cleared = False
    try:
        for chunk in gen:
            if not has_cleared:
                status_box.empty()
                has_cleared = True
            yield chunk
    finally:
        if not has_cleared:
            status_box.empty()

# ── Konfigurasi Halaman Streamlit ───────────────────────────────────────────
st.set_page_config(
    page_title="KarirRoast AI — CV Roaster & Career Wingman",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS untuk Tampilan Modern & Menarik ──────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Font Family & Global Smoothing */
    html, body, [class*="css"], .stMarkdown, .stText, [data-testid="stAppViewContainer"], .stButton, .stTextInput, .stSelectbox {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 1180px !important;
    }

    /* Hero Banner: Modern, Simple, Clean SaaS Card */
    .hero-banner {
        position: relative;
        background: linear-gradient(145deg, #131826 0%, #0d111a 100%);
        padding: 1.6rem 2rem;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        margin-bottom: 1.2rem;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -40px;
        right: -40px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(255, 90, 54, 0.16) 0%, transparent 70%);
        pointer-events: none;
        filter: blur(20px);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #ffffff 40%, #ff7a59 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #94a3b8;
        font-weight: 400;
        line-height: 1.6;
        max-width: 780px;
        margin-bottom: 1rem;
    }

    /* Badge Pills */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(6px);
    }
    .badge:hover {
        transform: translateY(-1px);
    }
    .badge-fire {
        background: rgba(255, 90, 54, 0.12);
        color: #ff8264;
        border: 1px solid rgba(255, 90, 54, 0.28);
    }
    .badge-rag {
        background: rgba(56, 189, 248, 0.1);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    .badge-persona {
        background: rgba(168, 85, 247, 0.1);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.25);
    }

    /* Onboarding Welcome Card */
    .onboarding-card {
        background: linear-gradient(145deg, rgba(20, 26, 42, 0.95) 0%, rgba(13, 17, 28, 0.98) 100%);
        border: 1px solid rgba(255, 90, 54, 0.25);
        border-radius: 18px;
        padding: 1.8rem 2.2rem;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 12px 36px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    .onboarding-badge {
        display: inline-block;
        background: rgba(255, 90, 54, 0.15);
        color: #ff7a59;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.25rem 0.8rem;
        border-radius: 9999px;
        border: 1px solid rgba(255, 90, 54, 0.32);
        margin-bottom: 0.6rem;
        letter-spacing: 0.02em;
    }
    .onboarding-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
    }
    .onboarding-desc {
        font-size: 0.93rem;
        color: #94a3b8;
        margin: 0 0 1.3rem 0;
        line-height: 1.5;
    }
    .steps-container {
        display: flex;
        flex-direction: column;
        gap: 0.85rem;
    }
    .step-item {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 13px;
        padding: 1rem 1.25rem;
        transition: all 0.2s ease;
    }
    .step-item:hover {
        background: rgba(255, 255, 255, 0.045);
        border-color: rgba(255, 122, 89, 0.35);
    }
    .step-number {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: linear-gradient(135deg, #ff5a36 0%, #ff7a59 100%);
        color: #ffffff;
        font-size: 0.85rem;
        font-weight: 800;
        flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(255, 90, 54, 0.35);
    }
    .step-content {
        flex: 1;
    }
    .step-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.25rem;
    }
    .step-text {
        font-size: 0.86rem;
        color: #94a3b8;
        line-height: 1.5;
    }
    .step-text a {
        color: #ff7a59;
        text-decoration: none;
        font-weight: 600;
    }
    .step-text a:hover {
        text-decoration: underline;
    }
    .persona-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.65rem;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.35rem 0.7rem;
        font-size: 0.8rem;
        color: #cbd5e1;
    }
    .chip b {
        color: #ffffff;
        margin-right: 4px;
    }
    .onboarding-footer {
        margin-top: 1.2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        display: flex;
        align-items: center;
        gap: 8px;
        color: #ff8264;
        font-size: 0.88rem;
        font-weight: 600;
    }

    /* Modern Segmented Interactive Tabs (Tampak Jelas Bisa Diklik) */
    div[data-baseweb="tab-list"] {
        gap: 0.55rem !important;
        background: linear-gradient(180deg, rgba(20, 26, 42, 0.95) 0%, rgba(13, 17, 28, 0.98) 100%) !important;
        padding: 0.45rem !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    }
    button[data-baseweb="tab"] {
        cursor: pointer !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.25rem !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.12) !important;
        border-color: rgba(255, 122, 89, 0.55) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(255, 90, 54, 0.22) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, rgba(255, 90, 54, 0.32) 0%, rgba(255, 122, 89, 0.2) 100%) !important;
        border: 1.5px solid #ff7a59 !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 18px rgba(255, 90, 54, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Buttons: Primary & Secondary Micro-interactions */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #ff5a36 0%, #ff7a59 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 11px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.25rem !important;
        box-shadow: 0 4px 14px rgba(255, 90, 54, 0.3) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 90, 54, 0.45) !important;
    }
    div.stButton > button[kind="primary"]:active,
    div.stButton > button[data-testid="baseButton-primary"]:active {
        transform: translateY(0) scale(0.98) !important;
    }
    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="baseButton-secondary"] {
        background: rgba(255, 255, 255, 0.03) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 11px !important;
        font-weight: 500 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="baseButton-secondary"]:hover {
        background: rgba(255, 255, 255, 0.07) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }

    /* Chat Message Bubbles */
    [data-testid="stChatMessage"] {
        background-color: rgba(19, 24, 38, 0.55) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.85rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stChatMessage"]:hover {
        border-color: rgba(255, 255, 255, 0.12) !important;
    }
    [data-testid="stChatInput"] {
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background-color: rgba(15, 19, 30, 0.95) !important;
        backdrop-filter: blur(12px) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(255, 90, 54, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(255, 90, 54, 0.15) !important;
    }

    /* Expanders & Cards */
    [data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 12px !important;
        background: rgba(19, 24, 38, 0.35) !important;
        overflow: hidden;
        margin-bottom: 0.85rem;
    }
    [data-testid="stExpander"] summary {
        font-weight: 500 !important;
        color: #e2e8f0 !important;
        padding: 0.8rem 1rem !important;
        transition: background-color 0.2s ease;
    }
    [data-testid="stExpander"] summary:hover {
        background-color: rgba(255, 255, 255, 0.03) !important;
    }

    /* Form Inputs */
    textarea, input[type="text"], input[type="password"] {
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: rgba(19, 24, 38, 0.5) !important;
        transition: all 0.2s ease !important;
    }
    textarea:focus, input[type="text"]:focus, input[type="password"]:focus {
        border-color: #ff7a59 !important;
        box-shadow: 0 0 0 2px rgba(255, 90, 54, 0.2) !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Custom Modern Slim Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 9999px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Inisialisasi Session State ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "cv_data" not in st.session_state:
    st.session_state.cv_data = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "audit_result" not in st.session_state:
    st.session_state.audit_result = None
if "active_file_name" not in st.session_state:
    st.session_state.active_file_name = None
if "job_match_result" not in st.session_state:
    st.session_state.job_match_result = None

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔥 **Pengaturan KarirRoast**")

    # 1. API Key Input & Pilihan Model Gemini
    st.markdown("#### 🔑 **Google Gemini API**")
    env_key = os.getenv("GOOGLE_API_KEY", "")
    raw_api_key = st.text_input(
        "Google AI Studio API Key",
        value=env_key,
        type="password",
        help="Dapatkan API Key gratis di Google AI Studio (aistudio.google.com)",
    )
    api_key = raw_api_key.strip().strip('"').strip("'") if raw_api_key else ""

    st.markdown(
        """<div style="margin-top: -6px; margin-bottom: 10px;">
            <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color: #ff7b54; font-size: 0.8rem; text-decoration: none; font-weight: 600;">
                🔑 Dapatkan Google API Key Gratis disini ↗
            </a>
        </div>""",
        unsafe_allow_html=True,
    )

    model_data = fetch_gemini_models_data(api_key)
    model_options = model_data["models"]
    is_dynamic = model_data["is_dynamic"]
    error_msg = model_data["error_msg"]

    default_idx = 0
    for cand in ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
        if cand in model_options:
            default_idx = model_options.index(cand)
            break

    selected_model = st.selectbox(
        "🤖 Model Gemini:",
        options=model_options,
        index=default_idx,
        help="Model Gemini yang digunakan untuk Roasting, Chat, dan Job Match. Terhubung dinamis dengan API Key Anda."
    )

    if not api_key:
        st.warning("⚠️ Masukkan Google API Key untuk mengaktifkan AI.", icon="🔑")
    elif is_dynamic:
        st.caption(f"✨ *{len(model_options)} model aktif terdeteksi.*")
    elif error_msg:
        st.caption("ℹ️ *Menggunakan preset model default.*")

    # Tombol Reset Chat & Data tepat di bawah pilihan model
    if st.button("🗑️ Reset Chat & Data", use_container_width=True):
        st.session_state.messages = []
        st.session_state.audit_result = None
        st.session_state.job_match_result = None
        st.rerun()

    st.divider()

    # 2. Pilihan Dokumen CV
    st.markdown("#### 📄 **Pilih Dokumen CV**")
    cv_source_mode = st.radio(
        "Sumber CV:",
        ["Gunakan Contoh CV", "Upload PDF Sendiri"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    pdf_file_to_process = None
    target_name = None

    if cv_source_mode == "Gunakan Contoh CV":
        sample_path = "sample_cv.pdf"
        if os.path.exists(sample_path):
            target_name = "sample_cv.pdf"
            pdf_file_to_process = sample_path
        else:
            st.error("File sample_cv.pdf tidak ditemukan di direktori proyek.")
    else:
        uploaded_file = st.file_uploader("Upload CV (Format PDF)", type=["pdf"], label_visibility="collapsed")
        if uploaded_file is not None:
            target_name = uploaded_file.name
            pdf_file_to_process = uploaded_file

    # Load / reload dokumen jika ada file baru
    if pdf_file_to_process is not None and (st.session_state.active_file_name != target_name):
        with st.spinner("Membaca dan memproses PDF CV..."):
            try:
                extracted = extract_pdf_data(pdf_file_to_process)
                st.session_state.cv_data = extracted
                st.session_state.active_file_name = target_name
                st.session_state.audit_result = None
                st.session_state.messages = []
                
                # Buat RAG Vector Store jika API key tersedia
                if api_key:
                    st.session_state.vectorstore = build_rag_index(extracted["full_text"], api_key)
                else:
                    st.session_state.vectorstore = None
                st.success(f"Berhasil memuat: {target_name} ({extracted['total_pages']} hal)")
            except Exception as e:
                st.error(f"Gagal memproses PDF: {e}")

    # Indikator CV aktif yang ringkas (1 baris)
    if st.session_state.cv_data:
        st.success(
            f"📄 **{st.session_state.active_file_name}** ({st.session_state.cv_data['total_pages']} hal · {st.session_state.cv_data['word_count']} kata)",
            icon="✅"
        )

    st.divider()

    # 3. Persona AI & Level Kepedasan
    st.markdown("#### 🎭 **Pilih Persona AI**")
    persona_options = {
        "hrd_savage": "😈 HRD Savage (Roaster & Debat)",
        "dukun_karir": "🔮 Dukun Primbon Karir (Mbah Broto)",
        "ats_pro": "👔 Konsultan ATS & Karir Pro",
        "mock_interviewer": "🥊 Interviewer Muka Tembok",
    }

    selected_persona_key = st.selectbox(
        "Karakter Persona:",
        options=list(persona_options.keys()),
        format_func=lambda k: persona_options[k],
        index=0,
        label_visibility="collapsed",
        help="Pilih persona AI yang akan mengevaluasi CV, berdebat di chat, dan menilai kecocokan lowongan.",
    )
    current_persona_info = PERSONA_PROMPTS[selected_persona_key]
    st.caption(f"💡 *{current_persona_info['tagline']}*")

    spicy_level = 2
    if selected_persona_key == "hrd_savage":
        spicy_level = st.slider(
            "Tingkat Kepedasan Roasting 🌶️",
            min_value=1,
            max_value=3,
            value=2,
            help="1: Menyindir Halus, 2: HRD Jaksel Julid, 3: Neraka Jahanam (Tanpa Ampun 🔥)",
        )
        st.caption(f"🔥 *{SPICY_LEVELS[spicy_level]}*")



# ── Header Banner ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🔥 KarirRoast AI</div>
        <div class="hero-subtitle">
            Platform AI Roaster, Simulator Debat HRD, & Career Wingman Berbasis RAG (LangChain + FAISS + Gemini) <br>Buat lucu lucuan aja yagesyakk :)
        </div>
        <div class="badge-container">
            <span class="badge badge-fire">🔥 Multimodal CV Roaster</span>
            <span class="badge badge-rag">🧠 LangChain & FAISS RAG</span>
            <span class="badge badge-persona">🎭 Dynamic Multi-Persona</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Cek prasyarat sebelum mulai
if not api_key:
    st.html(
        """<div class="onboarding-card">
<div class="onboarding-header">
<span class="onboarding-badge">🚀 Panduan Cepat Memulai</span>
<h3 class="onboarding-title">Selamat Datang di KarirRoast AI!</h3>
<p class="onboarding-desc">Lengkapi 3 langkah mudah pada sidebar di sebelah kiri untuk mengaktifkan AI dan memulai analisis:</p>
</div>
<div class="steps-container">
<div class="step-item">
<div class="step-number">1</div>
<div class="step-content">
<div class="step-title">Masukkan Google AI Studio API Key</div>
<div class="step-text">Dapatkan API Key gratis di <a href="https://aistudio.google.com/app/apikey" target="_blank">Google AI Studio ↗</a> lalu masukkan pada form API di sidebar kiri.</div>
</div>
</div>
<div class="step-item">
<div class="step-number">2</div>
<div class="step-content">
<div class="step-title">Pilih Dokumen CV</div>
<div class="step-text">Pilih <b>Gunakan Contoh CV</b> untuk coba instan, atau pilih <b>Upload PDF Sendiri</b> untuk menguji CV aslimu.</div>
</div>
</div>
<div class="step-item">
<div class="step-number">3</div>
<div class="step-content">
<div class="step-title">Pilih Persona AI Favorit</div>
<div class="step-text">Sesuaikan karakter evaluator & lawan bicara yang kamu inginkan:</div>
<div class="persona-chips">
<span class="chip">😈 <b>HRD Savage:</b> Roaster julid 3 level kepedasan</span>
<span class="chip">🔮 <b>Mbah Broto:</b> Dukun weton tech & ramal aura</span>
<span class="chip">👔 <b>ATS Pro:</b> Evaluasi objektif standar korporat</span>
<span class="chip">🥊 <b>Interviewer:</b> Mock interview teknis killer</span>
</div>
</div>
</div>
</div>
<div class="onboarding-footer">
👉 <span><b>Buka menu di sidebar sebelah kiri</b> untuk mengisi API Key dan mulai sesi!</span>
</div>
</div>"""
    )
    st.stop()

if not st.session_state.cv_data:
    st.warning(
        "📄 **Dokumen CV belum dipilih.** Silakan pilih opsi **Gunakan Contoh CV** atau unggah berkas PDF CV Anda pada menu sidebar sebelah kiri untuk memulai analisis.",
        icon="⚠️"
    )
    st.stop()

# Buat ulang RAG index jika belum ada tapi API Key sudah tersedia
if st.session_state.vectorstore is None and st.session_state.cv_data:
    with st.spinner("Mengindeks CV ke dalam Vector Database (FAISS)..."):
        st.session_state.vectorstore = build_rag_index(st.session_state.cv_data["full_text"], api_key)


tab_audit, tab_chat, tab_jd, tab_preview = st.tabs([
    "💥 Instant Roasting & Audit",
    "💬 Arena Baku Hantam (Chat & RAG)",
    "🎯 Job Matcher & Simulator",
    "📄 Preview Teks CV",
])

# ── TAB 1: INSTANT ROASTING & AUDIT ─────────────────────────────────────────
with tab_audit:
    col_desc, col_btn = st.columns([3, 1])
    with col_desc:
        current_persona_info = PERSONA_PROMPTS[selected_persona_key]
        st.markdown(f"**Persona Aktif:** `{current_persona_info['name']}` — *{current_persona_info['tagline']}*")
    with col_btn:
        btn_placeholder = st.empty()
        roast_button = btn_placeholder.button("🔥 Roast CV Sekarang!", use_container_width=True, type="primary")

    if roast_button:
        # Langsung ubah tombol jadi disabled agar tidak bisa diklik dua kali
        btn_placeholder.button("⏳ Sedang Meroasting...", disabled=True, use_container_width=True)
        st.markdown("### 📊 Hasil Evaluasi & Roasting:")
        status_box = st.empty()
        status_box.info(
            f"🔥 **{current_persona_info['name']} sedang menganalisis dokumen CV & meracik evaluasi...** *(Model: `{selected_model}`)*",
            icon="⏳"
        )
        stream_gen = generate_instant_audit_stream(
            cv_text=st.session_state.cv_data["full_text"],
            persona_key=selected_persona_key,
            spicy_level=spicy_level,
            api_key=api_key,
            model_name=selected_model,
        )
        st.session_state.audit_result = st.write_stream(stream_with_status_clear(stream_gen, status_box))
        btn_placeholder.button("🔄 Roast Ulang CV", use_container_width=True)
    elif st.session_state.audit_result:
        st.markdown("### 📊 Hasil Evaluasi & Roasting:")
        st.markdown(st.session_state.audit_result)
        
        st.divider()
        st.markdown("##### 🚀 Aksi Cepat ke Arena Chat:")
        c1, c2, c3 = st.columns(3)
        if c1.button("🥊 Debat: 'Proyek saya nyata dan berdampak!'"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Enak aja dibilang klise! Pengalaman dan proyek yang ada di CV saya itu nyata dan punya dampak terukur. Coba tunjukkan bagian mana yang menurut kamu paling lemah?"
            })
            st.rerun()
        if c2.button("✍️ Tough Love: 'Tolong benerin bullet point saya'"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Oke, saya akui masih ada kekurangan. Tolong berikan contoh perbaikan konkret (sebelum vs sesudah) untuk pengalaman kerja di CV saya menggunakan metode STAR."
            })
            st.rerun()
        if c3.button("🎯 Cek Kesiapan: 'Posisi apa yang paling cocok?'"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Berdasarkan keahlian riil di CV saya, posisi apa di industri yang paling realistis untuk saya lamar saat ini tanpa bikin recruiter tertawa?"
            })
            st.rerun()
    else:
        st.info("Klik tombol **'🔥 Roast CV Sekarang!'** di atas untuk mendapatkan evaluasi awal dan roasting tajam!")

# ── TAB 2: ARENA BAKU HANTAM (CHAT & RAG) ───────────────────────────────────
with tab_chat:
    persona_avatars = {
        "hrd_savage": "😈",
        "dukun_karir": "🔮",
        "ats_pro": "👔",
        "mock_interviewer": "🥊",
    }
    bot_avatar = persona_avatars.get(selected_persona_key, "🤖")

    st.markdown(
        f"**Mode Chat:** Bertanya, membela diri, atau simulasi interview dengan **{persona_options[selected_persona_key]}**.<br>"
        f"<small style='color: #888;'>Setiap jawaban bot didukung oleh <b>FAISS RAG Retriever</b> yang mencari bukti langsung dari dokumen CV kamu.</small>",
        unsafe_allow_html=True,
    )

    # Wadah percakapan agar selalu berada DI ATAS kotak input
    chat_container = st.container()

    # Kotak Input Chat berada di bawah wadah percakapan
    chat_prompt = st.chat_input("Ketik pembelaanmu, pertanyaan interview, atau minta saran perbaikan...")

    # Tampilkan percakapan yang tersimpan di dalam container
    with chat_container:
        for idx, msg in enumerate(st.session_state.messages):
            avatar = "🧑‍💻" if msg["role"] == "user" else bot_avatar
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                
                # Jika ada kutipan referensi RAG, tampilkan dalam expander
                if msg.get("rag_context"):
                    with st.expander("🔍 Bukti Dokumen yang Ditemukan Vektor RAG (FAISS):"):
                        st.caption(msg["rag_context"])

    if chat_prompt:
        # 1. Simpan pesan user
        st.session_state.messages.append({"role": "user", "content": chat_prompt})

        # 2. Retrieval RAG
        rag_context_text = ""
        if st.session_state.vectorstore:
            retrieval_res = retrieve_relevant_context(st.session_state.vectorstore, chat_prompt, k=3)
            rag_context_text = retrieval_res["context_text"]

        # 3. Tampilkan pesan user baru dan respons bot DI DALAM chat_container (di atas input)
        with chat_container:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(chat_prompt)

            # 4. Generate Response dari Gemini (Streaming Real-time)
            with st.chat_message("assistant", avatar=bot_avatar):
                status_box = st.empty()
                status_box.markdown(f"*{persona_options[selected_persona_key]} sedang menyusun balasan...* ✍️")
                stream_gen = generate_chat_turn_stream(
                    messages=st.session_state.messages,
                    rag_context=rag_context_text,
                    cv_full_text=st.session_state.cv_data["full_text"],
                    persona_key=selected_persona_key,
                    spicy_level=spicy_level,
                    api_key=api_key,
                    model_name=selected_model,
                )
                response_text = st.write_stream(stream_with_status_clear(stream_gen, status_box))
                
                if rag_context_text:
                    with st.expander("🔍 Bukti Dokumen yang Ditemukan Vektor RAG (FAISS):"):
                        st.caption(rag_context_text)

        # 5. Simpan pesan bot ke session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "rag_context": rag_context_text if rag_context_text else None,
        })

# ── TAB 3: JOB DESCRIPTION MATCHER ──────────────────────────────────────────
with tab_jd:
    st.markdown("### 🎯 **Pencocokan Lowongan Kerja (Job Description Matcher)**")
    st.caption("Paste deskripsi pekerjaan / kualifikasi lowongan yang ingin kamu lamar. AI akan mencocokkan kualifikasi dengan isi CV kamu secara realistis.")

    jd_input = st.text_area(
        "Kualifikasi / Job Description Lowongan Kerja:",
        placeholder="Contoh:\n- Minimal 2 tahun pengalaman dengan Laravel & Flutter\n- Paham arsitektur microservices dan Docker\n- Memiliki portofolio aplikasi mobile yang sudah rilis di PlayStore/AppStore...",
        height=180,
    )

    btn_jd_placeholder = st.empty()
    jd_button = btn_jd_placeholder.button("🚀 Uji Kecocokan CV dengan Lowongan", type="primary")

    if jd_button:
        if not jd_input.strip():
            st.warning("Silakan masukkan teks Job Description terlebih dahulu.")
        else:
            btn_jd_placeholder.button("⏳ Sedang Menganalisis Kecocokan...", disabled=True)
            st.markdown("### 📋 Hasil Analisis Kecocokan:")
            status_box = st.empty()
            status_box.info(
                f"🎯 **Sedang menganalisis kecocokan profil CV dengan kualifikasi lowongan...** *(Model: `{selected_model}`)*",
                icon="⏳"
            )
            stream_gen = generate_job_match_stream(
                cv_text=st.session_state.cv_data["full_text"],
                job_description=jd_input,
                persona_key=selected_persona_key,
                spicy_level=spicy_level,
                api_key=api_key,
                model_name=selected_model,
            )
            st.session_state.job_match_result = st.write_stream(stream_with_status_clear(stream_gen, status_box))
            btn_jd_placeholder.button("🔄 Uji Ulang Lowongan Lain", type="primary")
    elif st.session_state.job_match_result:
        st.markdown("### 📋 Hasil Analisis Kecocokan:")
        st.markdown(st.session_state.job_match_result)

# ── TAB 4: PREVIEW TEKS CV ──────────────────────────────────────────────────
with tab_preview:
    st.markdown("### 📄 **Ekstraksi Teks Lengkap dari Dokumen CV**")
    st.caption(f"File: **{st.session_state.active_file_name}** | Total Karakter: {st.session_state.cv_data['char_count']}")

    for idx, page_txt in enumerate(st.session_state.cv_data["pages"]):
        with st.expander(f"📑 Halaman {idx + 1}", expanded=(idx == 0)):
            st.text(page_txt if page_txt else "(Halaman kosong atau hanya berisi gambar tanpa teks)")
