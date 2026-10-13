from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging

logger = logging.getLogger(__name__)

class CVVectorStore:
    def __init__(self, vectorstore=None, raw_chunks: Optional[List[Document]] = None):
        self.vectorstore = vectorstore
        self.raw_chunks = raw_chunks or []

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self.vectorstore is not None:
            try:
                return self.vectorstore.similarity_search(query, k=k)
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to lexical search: {e}")

        # Fallback lexical search jika embedding API sedang limit/gagal
        return self._fallback_lexical_search(query, k=k)

    def _fallback_lexical_search(self, query: str, k: int = 4) -> List[Document]:
        words = set(query.lower().split())
        scored_chunks = []
        for doc in self.raw_chunks:
            content_lower = doc.page_content.lower()
            score = sum(1 for w in words if w in content_lower)
            scored_chunks.append((score, doc))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_chunks[:k]]


def build_rag_index(cv_text: str, api_key: str) -> CVVectorStore:
    """
    Memecah teks CV menjadi chunk dan membuat indeks vektor FAISS
    menggunakan GoogleGenerativeAIEmbeddings (gemini-embedding-2 / text-embedding-004).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=70,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_text(cv_text)
    documents = [
        Document(
            page_content=chunk,
            metadata={"chunk_id": idx, "source": "CV Kandidat"}
        )
        for idx, chunk in enumerate(chunks)
    ]

    vectorstore = None
    embedding_models = ["models/gemini-embedding-2", "models/text-embedding-004"]

    for model_name in embedding_models:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key
            )
            vectorstore = FAISS.from_documents(documents, embeddings)
            break
        except Exception as e:
            logger.warning(f"Gagal inisialisasi embedding dengan {model_name}: {e}")

    return CVVectorStore(vectorstore=vectorstore, raw_chunks=documents)


def retrieve_relevant_context(cv_vector_store: CVVectorStore, query: str, k: int = 4) -> Dict[str, Any]:
    """
    Mencari bagian CV yang paling relevan dengan query user.
    """
    docs = cv_vector_store.similarity_search(query, k=k)
    formatted_context = "\n---\n".join([f"[Bagian {i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])
    return {
        "context_text": formatted_context,
        "docs": docs
    }
