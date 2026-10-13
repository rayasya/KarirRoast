from typing import List, Dict, Any, Optional, Generator
from google import genai
from google.genai import types
from core.personas import PERSONA_PROMPTS, SPICY_LEVELS, INSTANT_AUDIT_PROMPT, RAG_CHAT_PROMPT_TEMPLATE

def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)

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
    model_name: str = "gemini-2.5-flash"
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
    model_name: str = "gemini-2.5-flash"
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
    model_name: str = "gemini-2.5-flash"
) -> Generator[str, None, None]:
    client = get_client(api_key)
    system_instruction = get_system_instruction(persona_key, spicy_level)

    prompt = f"""Tolong bandingkan profil CV kandidat ini dengan Lowongan Kerja (Job Description) berikut:

<JOB_DESCRIPTION>
{job_description[:4000]}
</JOB_DESCRIPTION>

<CV_KANDIDAT>
{cv_text[:8000]}
</CV_KANDIDAT>

Format Jawaban yang Diharapkan (Markdown):
1. 🎯 **Peluang Lolos Screening:** (Persentase % & Vonis Singkat)
2. 🚨 **Kelemahan & Skill Gap Terbesar:** (Apa yang diminta lowongan tapi tidak ada di CV)
3. 🔥 **Komentar Pedas:** (Roasting realistis tentang seberapa delusional atau cocoknya kandidat melamar posisi ini)
4. 🛠️ **Strategi Menambal Kekurangan:** (3 langkah taktis agar lamaran ini tidak langsung masuk folder Spam)
"""

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
def generate_instant_audit(cv_text, persona_key, spicy_level, api_key, model_name="gemini-2.5-flash") -> str:
    return "".join(generate_instant_audit_stream(cv_text, persona_key, spicy_level, api_key, model_name))

def generate_chat_turn(messages, rag_context, cv_full_text, persona_key, spicy_level, api_key, model_name="gemini-2.5-flash") -> str:
    return "".join(generate_chat_turn_stream(messages, rag_context, cv_full_text, persona_key, spicy_level, api_key, model_name))

def generate_job_match(cv_text, job_description, persona_key, spicy_level, api_key, model_name="gemini-2.5-flash") -> str:
    return "".join(generate_job_match_stream(cv_text, job_description, persona_key, spicy_level, api_key, model_name))
