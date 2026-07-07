import hashlib
import os
from dataclasses import dataclass
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError

load_dotenv()

CHAT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)
EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "text-embedding-004"
)
MAX_CONTEXT_CHUNKS = 5
MIN_USEFUL_TEXT_CHARS = 20

@dataclass
class Chunk:
    text: str
    page: int
    source: str

def get_client() -> genai.Client | None:
    api_key = os.getenv("GEMINI_API_KEY")
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
