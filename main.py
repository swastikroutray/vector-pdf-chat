import hashlib
import os
from dataclasses import dataclass
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_dotenv()

CHAT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)
EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001"
)
MAX_CONTEXT_CHUNKS = 5
MIN_USEFUL_TEXT_CHARS = 20

@dataclass
class Chunk:
    text: str
    page: int
    source: str

def get_client() -> genai.Client | None:
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    st.write("API key exists:", api_key is not None)
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

from typing import Any, Iterable
def file_hash(files: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(file.getvalue())
    return digest.hexdigest()

def extract_pdf_chunks(uploaded_files: list[Any]) -> list[Chunk]:
    chunks: list[Chunk] = []

    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(uploaded_file)
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = clean_extracted_text(page.extract_text() or "")
                if not page_text:
                    continue

                chunks.extend(
                    Chunk(text=chunk, page=page_number, source=uploaded_file.name)
                    for chunk in split_text(page_text)
                )

        except DependencyError as exc:
            st.error(f"Could not read `{uploaded_file.name}` Install the required dependencies"
            " Run `pip install -r requirements.txt`, then restart the app.")
            st.caption(str(exc))

        except PdfReadError as exc:
            st.error(f"Could not read `{uploaded_file.name}` as a valid PDF.")
            st.caption(str(exc))

    return chunks

def clean_extracted_text(text: str) -> str:
    text = "".join(char if char.isprintable() else " " for char in text)
    text = " ".join(text.split())

    useful_chars = sum(char.isalnum() for char in text)
    if useful_chars < MIN_USEFUL_TEXT_CHARS:
        return ""
    
    return text

def split_text(text: str, chunk_size: int = 1400, overlap: int = 220) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start< len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, 0)
    return chunks

def embed_texts(client: genai.Client, texts: list[str]) -> np.ndarray:
    vectors =[]
    for text in texts:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
        vectors.append(response.embeddings[0].values)

    vectors = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims= True)
    return vectors/np.clip(norms, 1e-12, None)


def build_index(client: genai, uploaded_files: list[Any]) -> None:
    chunks = extract_pdf_chunks(uploaded_files)
    if not chunks:
        st.session_state.pdf_chunks = []
        st.session_state.pdf_embeddings = None
        return
    
    with st.spinner("Reading your PDFs..."):
        embeddings = embed_texts(client, [chunk.text for chunk in chunks])

    st.session_state.pdf_chunks = chunks
    st.session_state.pdf_embeddings = embeddings

def retrieve_context(client: genai.Client, question: str)-> list[tuple[Chunk, float]]:
    chunks:list[Chunk] = st.session_state.get("pdf_chunks", [])
    embeddings = st.session_state.get("pdf_embeddings")
    if not chunks or embeddings is None:
        return []
    
    query_embedding = embed_texts(client,[question])[0]
    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:MAX_CONTEXT_CHUNKS]
    return [(chunks[index], float(scores[index])) for index in top_indices]

def answer_question(
    client: genai.Client,
    question: str,
    matches: list[tuple[Chunk, float]],
) -> str:

    context = "\n\n".join(
        f"[{index}] Source: {chunk.source}, page {chunk.page}\n{chunk.text}"
        for index, (chunk, _score) in enumerate(matches, start=1)
    )

    prompt = f"""
You are a careful PDF question-answering assistant.
Answer using only the provided PDF context.Use any explicit facts, dates, labels, headings, fields, table values, and document metadata that appear in the context.When answering date or time questions, distinguish between document date, sent date, effective date, due date, purchase date, event date, delivery date, or signature date.If the requested information is not present in the context, clearly say so instead of guessing. You are Athena.
Answer ONLY using the provided document context.
If the answer is not explicitly contained in the context, respond exactly with:
"I couldn't find information about that in the uploaded document."
Do not guess.
Do not use outside knowledge.

PDF Context:
{context}

Question:
{question}
""".strip()

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )

    return response.text

def initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pdf_file_hash", None)
    st.session_state.setdefault("pdf_chunks", [])
    st.session_state.setdefault("pdf_embeddings", None)

def main() -> None:
    st.set_page_config(page_title="Athena", page_icon=":material/description:", layout="wide")
    load_css()    
    initialize_state()

    st.title("Athena")
    st.caption("How can Athena help you today?")

    with st.sidebar:

        uploaded_files = st.file_uploader(
            "📄 Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    client = get_client()
    if not client:
        st.error("GEMINI_API_KEY not found.")
        return

    if uploaded_files:
        current_hash = file_hash(uploaded_files)
        if current_hash != st.session_state.pdf_file_hash:
            build_index(client, uploaded_files)
            st.session_state.pdf_file_hash = current_hash
            st.session_state.messages = []

        chunk_count = len(st.session_state.pdf_chunks)
        if chunk_count:
            st.success(f"Indexed {chunk_count} text chunks from {len(uploaded_files)} PDF file(s).")
        else:
            st.warning("No extractable text was found. Scanned PDFs may need OCR first.")
    else:
        st.info("Upload at least one PDF to start chatting.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDFs")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDFs and writing an answer..."):
            matches = retrieve_context(client, question)
            if not matches:
                answer = "I couldn't find any relevant information in the uploaded PDFs."
            else:
                answer = answer_question(client, question, matches)
        st.markdown(answer)

        with st.expander("Retrieved context"):
            for i, (chunk, score) in enumerate(matches, start=1):
                st.markdown(f"### Chunk {i}")
                st.markdown(f"**Source:** {chunk.source}")
                st.markdown(f"**Page:** {chunk.page}")
                st.markdown(f"**Similarity:** `{score:.3f}`")
                st.write(chunk.text)
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()