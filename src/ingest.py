"""
Ingestion script.

Run this once, and again whenever you add or change transcripts:

    python src/ingest.py

It reads every .txt file in data/speeches/, cleans and chunks the text,
creates embeddings, and saves a FAISS index to vectorstore/.

The Streamlit app (app.py) does not recreate the database at startup.
It only loads the existing FAISS index.
"""

import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------------

# Allows this file to be run directly:
# python src/ingest.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    SPEECHES_PATH,
    VECTORSTORE_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ---------------------------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean transcript text while preserving meaningful content.
    """

    if not text:
        return ""

    # Normalize Windows line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters
    text = text.replace("\x00", "")

    # Remove excessive spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------

def extract_metadata(filename: str) -> dict:
    """
    Extract useful metadata from the filename.

    Examples:

        speech_2022.txt
        -> {"year": "2022"}

        2019-08-15_independence.txt
        -> {"date": "2019-08-15", "year": "2019"}
    """

    metadata = {}

    date_match = re.search(
        r"(\d{4}-\d{2}-\d{2})",
        filename,
    )

    if date_match:
        metadata["date"] = date_match.group(1)

    year_match = re.search(
        r"(19|20)\d{2}",
        filename,
    )

    if year_match:
        metadata["year"] = year_match.group(0)

    return metadata


# ---------------------------------------------------------------------------
# LOAD SINGLE FILE
# ---------------------------------------------------------------------------

def load_single_file(file_path: Path):
    """
    Load one transcript.

    Try UTF-8 first.
    If that fails, try UTF-8 with BOM.
    Then try UTF-16.
    """

    encodings_to_try = [
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
    ]

    last_error = None

    for encoding in encodings_to_try:

        try:

            loader = TextLoader(
                str(file_path),
                encoding=encoding,
            )

            documents = loader.load()

            if documents:
                return documents

        except (
            UnicodeDecodeError,
            UnicodeError,
        ) as error:

            last_error = error
            continue

        except Exception as error:

            print(
                f"  ERROR while reading {file_path.name}: "
                f"{error}"
            )

            return []

    print(
        f"  ERROR: Could not decode {file_path.name}."
    )

    if last_error:
        print(f"  Last decoding error: {last_error}")

    return []


# ---------------------------------------------------------------------------
# LOAD ALL TRANSCRIPTS
# ---------------------------------------------------------------------------

def load_transcripts() -> list:
    """
    Load and clean every .txt file in data/speeches/.
    """

    txt_files = sorted(
        SPEECHES_PATH.glob("*.txt")
    )

    if not txt_files:

        print(
            f"ERROR: No .txt files found in:\n"
            f"{SPEECHES_PATH}"
        )

        print(
            "\nPlease add your transcript files to:"
        )

        print(
            "data/speeches/"
        )

        sys.exit(1)

    print(
        f"Found {len(txt_files)} file(s).\n"
    )

    documents = []

    for file_path in txt_files:

        print(
            f"Reading: {file_path.name}"
        )

        # Check file size before loading
        file_size = file_path.stat().st_size

        print(
            f"  File size: {file_size:,} bytes"
        )

        if file_size == 0:

            print(
                "  WARNING: File is empty. Skipping."
            )

            continue

        # Load file
        loaded_docs = load_single_file(
            file_path
        )

        if not loaded_docs:

            print(
                "  WARNING: No document content loaded."
            )

            continue

        for doc in loaded_docs:

            original_length = len(
                doc.page_content
            )

            # Clean text
            cleaned_content = clean_text(
                doc.page_content
            )

            cleaned_length = len(
                cleaned_content
            )

            print(
                f"  Characters: "
                f"{original_length:,} -> "
                f"{cleaned_length:,}"
            )

            # Skip empty documents
            if not cleaned_content:

                print(
                    "  WARNING: No usable text after "
                    "cleaning. Skipping."
                )

                continue

            # Update document
            doc.page_content = cleaned_content

            # Add filename
            doc.metadata["source"] = (
                file_path.name
            )

            # Add additional metadata
            doc.metadata.update(
                extract_metadata(
                    file_path.name
                )
            )

            documents.append(doc)

        print()

    # Final validation
    if not documents:

        print(
            "ERROR: No usable transcript documents "
            "were loaded."
        )

        print(
            "\nPossible reasons:"
        )

        print(
            "1. Your TXT files are empty."
        )

        print(
            "2. The files contain an unsupported encoding."
        )

        print(
            "3. The files do not contain readable text."
        )

        sys.exit(1)

    print(
        f"Successfully loaded "
        f"{len(documents)} document(s).\n"
    )

    return documents


# ---------------------------------------------------------------------------
# SPLIT DOCUMENTS
# ---------------------------------------------------------------------------

def split_documents(documents: list) -> list:
    """
    Split documents into overlapping chunks.
    """

    if not documents:

        print(
            "ERROR: No documents available "
            "for splitting."
        )

        sys.exit(1)

    print(
        "Splitting documents..."
    )

    print(
        f"Chunk size: {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: {CHUNK_OVERLAP}"
    )

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    # Remove accidental empty chunks
    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not chunks:

        print(
            "\nERROR: Text splitting produced "
            "0 chunks."
        )

        print(
            "\nCheck your transcript files and "
            "CHUNK_SIZE in src/config.py."
        )

        sys.exit(1)

    print(
        f"Created {len(chunks)} chunks.\n"
    )

    return chunks


# ---------------------------------------------------------------------------
# BUILD VECTOR DATABASE
# ---------------------------------------------------------------------------

def build_vectorstore(chunks: list) -> None:
    """
    Create embeddings and save a FAISS vector database.
    """

    if not chunks:

        print(
            "ERROR: Cannot build FAISS database "
            "because there are no chunks."
        )

        sys.exit(1)

    print(
        "Creating embeddings..."
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    try:

        embeddings = (
            GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL
            )
        )

    except Exception as error:

        print(
            "\nERROR: Could not initialize "
            "the embedding model."
        )

        print(
            f"Details: {error}"
        )

        sys.exit(1)

    print(
        "Building FAISS vector database..."
    )

    try:

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings,
        )

    except Exception as error:

        print(
            "\nERROR: Failed to create "
            "FAISS vector database."
        )

        print(
            f"Details: {error}"
        )

        sys.exit(1)

    # Make sure directory exists
    VECTORSTORE_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save FAISS
    vectorstore.save_local(
        str(VECTORSTORE_PATH)
    )

    print(
        "\nFAISS database saved successfully."
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    print("=" * 60)
    print("CHAT WITH [CHARACTER_NAME] - KNOWLEDGE BASE")
    print("=" * 60)
    print()

    # Load .env
    load_dotenv()

    # -------------------------------------------------------
    # API KEY CHECK
    # -------------------------------------------------------

    if not os.getenv("GOOGLE_API_KEY"):

        print(
            "ERROR: GOOGLE_API_KEY is not set."
        )

        print(
            "\nCreate a .env file in the project root:"
        )

        print(
            "GOOGLE_API_KEY=your_api_key_here"
        )

        sys.exit(1)

    # -------------------------------------------------------
    # PATH INFORMATION
    # -------------------------------------------------------

    print(
        f"Transcript folder:"
    )

    print(
        f"  {SPEECHES_PATH}"
    )

    print()

    print(
        f"Vectorstore folder:"
    )

    print(
        f"  {VECTORSTORE_PATH}"
    )

    print()

    # -------------------------------------------------------
    # CHECK TRANSCRIPT DIRECTORY
    # -------------------------------------------------------

    if not SPEECHES_PATH.exists():

        print(
            "\nERROR: Transcript folder does not exist:"
        )

        print(
            SPEECHES_PATH
        )

        print(
            "\nCreate the folder:"
        )

        print(
            "data/speeches/"
        )

        sys.exit(1)

    # -------------------------------------------------------
    # CHECK EXISTING VECTORSTORE
    # -------------------------------------------------------

    index_file = (
        VECTORSTORE_PATH / "index.faiss"
    )

    replacing = index_file.exists()

    if replacing:

        print(
            "Existing FAISS database found."
        )

        print(
            "It will be replaced with the "
            "new transcript data.\n"
        )

    # -------------------------------------------------------
    # LOAD
    # -------------------------------------------------------

    print(
        "Loading transcripts..."
    )

    documents = load_transcripts()

    # -------------------------------------------------------
    # SPLIT
    # -------------------------------------------------------

    chunks = split_documents(
        documents
    )

    # -------------------------------------------------------
    # BUILD FAISS
    # -------------------------------------------------------

    build_vectorstore(
        chunks
    )

    # -------------------------------------------------------
    # SUCCESS
    # -------------------------------------------------------

    action = (
        "replaced"
        if replacing
        else "created"
    )

    print()
    print("=" * 60)

    print(
        f"Successfully {action} the "
        "vector database!"
    )

    print("=" * 60)

    print()

    print(
        "Location:"
    )

    print(
        f"  {VECTORSTORE_PATH}"
    )

    print()

    print(
        f"Documents loaded: {len(documents)}"
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    print()

    print(
        "You can now run:"
    )

    print(
        "  streamlit run app.py"
    )

    print()


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()