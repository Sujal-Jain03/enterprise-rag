from typing import List
import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks of up to chunk_size characters, respecting
    natural boundaries (paragraphs -> lines -> sentences -> words),
    with chunk_overlap characters of overlap between consecutive chunks.
    """
    with logfire.span(" Text chunking", text_length=len(text)):
        if not text.strip():
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"Generated {len(valid_chunks)} chunks")
        return valid_chunks
