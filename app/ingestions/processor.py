import json
import logfire
import os
import sys
from typing import List, Optional

from app.ingestions.loaders.pdf import parse_pdf
from app.ingestions.loaders.html import parse_html
from app.ingestions.loaders.text import parse_text
from app.ingestions.loaders.office import parse_office
from app.ingestions.chunking.splitter import chunk_text

logfire.configure(service_name="enterprise-ingestion-service", send_to_logfire=False)


PROCESSED_DATA_DIR = "processed_data"


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(file_path: str, filename: str, source_type: str) -> Optional[str]:
    """Parse one file -> chunk -> save JSON locally. Returns JSON path or None."""
    with logfire.span("Processing file", file=filename, source=source_type):
        try:
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return None

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} - skipping.")
                return None

            chunks = chunk_text(full_text)
            if not chunks:
                logfire.warning(f"No chunks produced for {filename} - skipping.")
                return None

            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "num_chunks": len(chunks),
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved {len(chunks)} chunks to {local_path}")
            return local_path
        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")
            return None


def process_directory(dir_path: str, source_type: str) -> List[str]:
    """Process every file in a directory. Returns list of saved JSON paths."""
    with logfire.span("Scanning directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path)
                 if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        saved = []
        for filename in files:
            result = process_file(os.path.join(dir_path, filename), filename, source_type)
            if result:
                saved.append(result)
        return saved


def infer_source_type(name: str) -> str:
    """Map a folder/file name to a source type: true / noisy / general."""
    name = name.lower()
    if "true" in name:
        return "true"
    if "noisy" in name:
        return "noisy"
    return "general"


def run_ingestion(base_dir: str, explicit_source_type: Optional[str] = None) -> List[str]:
    """Ingest every file under base_dir. Sub-folders map to source types."""
    with logfire.span("Ingestion started", base_directory=base_dir):
        subdirs = [d for d in os.listdir(base_dir)
                   if os.path.isdir(os.path.join(base_dir, d))]
        saved = []
        if not subdirs:
            source_type = explicit_source_type or infer_source_type(
                os.path.basename(os.path.normpath(base_dir)))
            logfire.info(f"No sub-folders found - processing '{base_dir}' as '{source_type}'.")
            saved = process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = infer_source_type(subdir)
                saved += process_directory(os.path.join(base_dir, subdir), source_type)
        logfire.info(f"Ingestion done: {len(saved)} files processed.")
        return saved


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "DATA"
    run_ingestion(target)
