# ingestion/loader.py
import os
from datetime import datetime


def collect_markdown_files(root_dir: str) -> list:
    """
    Recursively collect all .md files inside root_dir.
    Returns absolute file paths.
    """
    markdown_files = []

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file in files:
            if file.lower().endswith(".md") and not file.startswith("."):
                full_path = os.path.join(root, file)
                markdown_files.append(os.path.abspath(full_path))

    return markdown_files


def read_markdown_file(file_path: str) -> str:
    """
    Safely read markdown file content.
    Tries utf-8 first, falls back to latin-1.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            content = f.read()

    # Basic normalization
    content = content.replace("\r\n", "\n").strip()

    return content


def extract_metadata(file_path: str, root_dir: str) -> dict:
    """
    Extract metadata for a given file.
    """
    last_modified_timestamp = os.path.getmtime(file_path)
    file_size = os.path.getsize(file_path)
    relative_path = os.path.relpath(file_path, root_dir)

    return {
        "source": relative_path.replace("\\", "/"),  
        "last_updated": last_modified_timestamp,
        "last_updated_readable": datetime.fromtimestamp(
            last_modified_timestamp
        ).strftime("%Y-%m-%d %H:%M:%S"),
        "file_size": file_size
    }


def load_documents(root_dir: str) -> list:
    """
    Full ingestion pipeline:
    - Traverse file system
    - Read markdown files
    - Extract metadata
    - Return structured document objects
    """

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    documents = []
    markdown_files = collect_markdown_files(root_dir)

    for file_path in markdown_files:
        content = read_markdown_file(file_path)

        # Skip empty files
        if not content.strip():
            continue

        metadata = extract_metadata(file_path, root_dir)

        documents.append({
            "content": content,
            **metadata
        })

    return documents


if __name__ == "__main__":
    ROOT_DIRECTORY = "docs"

    docs = load_documents(ROOT_DIRECTORY)

    print(f"\nTotal documents loaded: {len(docs)}\n")

    for doc in docs:
        print("Source:", doc["source"])
        print("Last Updated:", doc["last_updated_readable"])
        print("File Size:", doc["file_size"], "bytes")
        print("-" * 40)