import re
from typing import List, Dict, Any, Optional, Generator, Tuple
from google import genai
from google.genai import types
from core.personas import (
    PERSONA_PROMPTS,
    SPICY_LEVELS,
    INSTANT_AUDIT_PROMPT,
    RAG_CHAT_PROMPT_TEMPLATE,
    JOB_MATCH_PROMPT_TEMPLATE,
)

def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)

EXCLUDED_MODEL_KEYWORDS = [
    "robotics",       # Model robotika / fisik
    "vision",         # Model vision lama (contoh gemini-1.0-pro-vision, tidak support chat/system instruction)
    "embedding",      # Model embedding vektor
    "embed",
    "imagen",         # Image generation
    "veo",            # Video generation
    "medlm",          # Medical domain models
    "aqa",            # Attributed question answering
    "live",           # Model streaming audio 2-way real-time
    "audio",
    "tts",
    "custom",
    "computer-use",
    "deep-research",
]

def is_usable_chat_model(clean_name: str, actions: List[str]) -> bool:
    name_lower = clean_name.lower()
    # Harus produk Gemini
    if not name_lower.startswith("gemini"):
        return False
    # Saring kata kunci model non-chat / khusus / tidak kompatibel
    for kw in EXCLUDED_MODEL_KEYWORDS:
        if kw in name_lower:
            return False
    # Jika API menyediakan supported_actions, harus mendukung generateContent
    if actions and "generateContent" not in actions:
        return False
    # Harus model keluarga flash atau pro
    if not ("flash" in name_lower or "pro" in name_lower):
        return False
    return True

def get_available_models(api_key: str) -> Tuple[List[str], bool, Optional[str]]:
    """
    Mengambil daftar model Gemini yang mendukung 'generateContent' secara dinamis
    menggunakan API Key pengguna, menyaring hanya model teks/multimodal yang siap pakai.
    Mengembalikan tuple (daftar_model, is_dynamic_success, error_message).
    """
    fallback_models = [
        "gemini-3.5-flash",
        "gemini-2.5-flash",
    ]
    clean_key = api_key.strip().strip('"').strip("'") if api_key else ""
    if not clean_key:
        return fallback_models, False, None

    try:
        client = get_client(clean_key)
        fetched_models = []
        for m in client.models.list():
            name = m.name or ""
            clean_name = name[7:] if name.startswith("models/") else name
            actions = m.supported_actions or []

            # Saring model agar hanya model yang benar-benar bisa dipakai
            if is_usable_chat_model(clean_name, actions):
                fetched_models.append(clean_name)

        if fetched_models:
            unique_models = list(dict.fromkeys(fetched_models))

            # Mengutamakan varian Flash terlebih dahulu (paling cepat & efisien)
            def sort_key(item: str):
                match = re.search(r"gemini-(\d+(?:\.\d+)?)", item.lower())
                version = float(match.group(1)) if match else 0.0

                tier = 2
                if "flash" in item.lower() and "lite" not in item.lower():
                    tier = 0
                elif "pro" in item.lower():
                    tier = 1
                elif "lite" in item.lower():
                    tier = 2

                # -version membuat angka versi yang lebih besar (terbaru) berada di paling atas
                return (-version, tier, item)

            sorted_models = sorted(unique_models, key=sort_key)
            return sorted_models, True, None

        return fallback_models, False, "Tidak ditemukan model Gemini teks yang kompatibel."
    except Exception as e:
        err_raw = str(e)
        if "API_KEY_INVALID" in err_raw or "API key not valid" in err_raw:
            err_msg = "Google menolak API Key (status: API_KEY_INVALID). Pastikan key aktif dari Google AI Studio."
        else:
            err_msg = f"Gagal menghubungi server Google: {err_raw[:150]}"
        return fallback_models, False, err_msg

def get_system_instruction(persona_key: str, spicy_level: int = 2) -> str:
    persona_info = PERSONA_PROMPTS.get(persona_key, PERSONA_PROMPTS["hrd_savage"])
    spicy_desc = SPICY_LEVELS.get(spicy_level, SPICY_LEVELS[2])
    raw_prompt = persona_info["system_instruction"]
    if "{spicy_level_description}" in raw_prompt:
        return raw_prompt.format(spicy_level_description=spicy_desc)
    return raw_prompt

def generate_instant_audit_stream(
    cv_text: str,
    persona_key: str,
    spicy_level: int,
    api_key: str,
    model_name: str = "gemini-3.5-flash"
) -> Generator[str, None, None]:
    client = get_client(api_key)
    system_instruction = get_system_instruction(persona_key, spicy_level)
    persona_name = PERSONA_PROMPTS.get(persona_key, {}).get("name", "HRD Savage")

    prompt = INSTANT_AUDIT_PROMPT.format(
        cv_text=cv_text[:12000],  # batas aman konteks
        persona_name=persona_name
    )

    temp = 0.9 if spicy_level >= 2 else 0.7
    if persona_key == "ats_pro":
        temp = 0.3

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temp,
    )

    try:
        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=prompt,
            config=config
        )
        for chunk in response_stream:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"\n\n⚠️ Terjadi error saat audit: {e}"

def generate_chat_turn_stream(
    messages: List[Dict[str, str]],
    rag_context: str,
    cv_full_text: str,
    persona_key: str,
    spicy_level: int,
    api_key: str,
    model_name: str = "gemini-3.5-flash"
) -> Generator[str, None, None]:
    client = get_client(api_key)
    base_instruction = get_system_instruction(persona_key, spicy_level)
    persona_name = PERSONA_PROMPTS.get(persona_key, {}).get("name", "HRD Savage")

    rag_system_wrapper = RAG_CHAT_PROMPT_TEMPLATE.format(
        rag_context=rag_context,
        cv_summary=cv_full_text[:8000],
        persona_name=persona_name
    )

    full_system_instruction = f"{base_instruction}\n\n{rag_system_wrapper}"

    temp = 0.9 if spicy_level >= 2 else 0.7
    if persona_key == "ats_pro":
        temp = 0.3

    config = types.GenerateContentConfig(
        system_instruction=full_system_instruction,
        temperature=temp,
    )

    # Susun contents history
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    try:
        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=config
        )
        for chunk in response_stream:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"\n\n⚠️ Terjadi error saat chat: {e}"

def generate_job_match_stream(
    cv_text: str,
    job_description: str,
    persona_key: str,
    spicy_level: int,
    api_key: str,
    model_name: str = "gemini-3.5-flash"
) -> Generator[str, None, None]:
    client = get_client(api_key)
    base_instruction = get_system_instruction(persona_key, spicy_level)

    # Sesuaikan instruksi jika persona adalah mock_interviewer agar tidak mensimulasikan tanya-jawab interview
    if persona_key == "mock_interviewer":
        system_instruction = (
            "Kamu adalah 'VP of Engineering' yang sangat kritis dan sedang menyusun lembar audit teknis "
            "evaluasi berkas lamaran kandidat terhadap kualifikasi lowongan kerja.\n"
            "Tuliskan laporan evaluasi tertulis satu arah secara dingin, objektif, dan tajam.\n"
            "JANGAN melakukan simulasi interview tatap muka dan DILARANG mengajukan pertanyaan kepada kandidat."
        )
    else:
        system_instruction = (
            f"{base_instruction}\n\n"
            "PANDUAN LAPORAN EVALUASI JOB MATCHING:\n"
            "- Ini adalah dokumen laporan evaluasi screening tertulis satu arah (bukan sesi wawancara atau chat interaktif).\n"
            "- DILARANG mengajukan pertanyaan apapun kepada kandidat di akhir respons (karena tidak ada fitur chat/balas di halaman ini).\n"
            "- Berikan kesimpulan dan vonis akhir yang tuntas dan definitif."
        )

    prompt = JOB_MATCH_PROMPT_TEMPLATE.format(
        job_description=job_description[:4000],
        cv_text=cv_text[:8000],
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.8,
    )

    try:
        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=prompt,
            config=config
        )
        for chunk in response_stream:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"\n\n⚠️ Terjadi error saat perbandingan: {e}"

# Fallback functions jika ada yang memanggil versi synchronous
def generate_instant_audit(cv_text, persona_key, spicy_level, api_key, model_name="gemini-3.5-flash") -> str:
    return "".join(generate_instant_audit_stream(cv_text, persona_key, spicy_level, api_key, model_name))

def generate_chat_turn(messages, rag_context, cv_full_text, persona_key, spicy_level, api_key, model_name="gemini-3.5-flash") -> str:
    return "".join(generate_chat_turn_stream(messages, rag_context, cv_full_text, persona_key, spicy_level, api_key, model_name))

def generate_job_match(cv_text, job_description, persona_key, spicy_level, api_key, model_name="gemini-3.5-flash") -> str:
    return "".join(generate_job_match_stream(cv_text, job_description, persona_key, spicy_level, api_key, model_name))
