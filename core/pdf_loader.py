import pymupdf
from typing import Dict, Any, List

def extract_pdf_data(file_source) -> Dict[str, Any]:
    """
    Ekstrak teks dan metadata dari file PDF (bisa path string atau io.BytesIO / UploadedFile).
    Mengembalikan dict berisi teks lengkap, teks per halaman, jumlah halaman, dan statistik kata.
    """
    if isinstance(file_source, str):
        doc = pymupdf.open(file_source)
    elif hasattr(file_source, "read"):
        file_bytes = file_source.read()
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    elif isinstance(file_source, (bytes, bytearray)):
        doc = pymupdf.open(stream=file_source, filetype="pdf")
    else:
        raise ValueError("Format file PDF tidak didukung.")

    pages_text: List[str] = []
    full_text_list = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        pages_text.append(text)
        if text:
            full_text_list.append(f"--- Halaman {page_num + 1} ---\n{text}")

    full_text = "\n\n".join(full_text_list)
    word_count = len(full_text.split())
    char_count = len(full_text)

    doc.close()

    return {
        "full_text": full_text,
        "pages": pages_text,
        "total_pages": len(pages_text),
        "word_count": word_count,
        "char_count": char_count,
    }
