"""
Configuration for the Chat with [CHARACTER_NAME] application.

Keep model names as strings.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------

# Project root:
# C:\Users\User\Desktop\chat-with-character

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Transcript directory
SPEECHES_PATH = PROJECT_ROOT / "data" / "speeches"

# FAISS vector database directory
VECTORSTORE_PATH = PROJECT_ROOT / "vectorstore"

# Character image
ASSETS_PATH = PROJECT_ROOT / "assets"


# ---------------------------------------------------------------------------
# CHARACTER
# ---------------------------------------------------------------------------

CHARACTER_NAME = "Bhagwan Mogi Ji"


# ---------------------------------------------------------------------------
# GEMINI MODELS
# ---------------------------------------------------------------------------

# Current Gemini chat model.
#
# IMPORTANT:
# MODEL_NAME MUST be a string.
#
MODEL_NAME = "gemini-3.6-flash"


# Current Gemini embedding model.
#
# This must match the model used when creating the FAISS database.
#
EMBEDDING_MODEL = "gemini-embedding-2"


# ---------------------------------------------------------------------------
# GENERATION SETTINGS
# ---------------------------------------------------------------------------

# Kept because rag.py currently imports TEMPERATURE.
#
# If the Gemini API/library ignores this parameter for newer models,
# the application can still work.
#
TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# RAG SETTINGS
# ---------------------------------------------------------------------------

# Number of transcript chunks retrieved for every question.

TOP_K = 4

TOP_K_OPTIONS = [
    2,
    3,
    4,
    5,
    6,
    8,
]


# ---------------------------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------------------------

# Number of characters in each transcript chunk.

CHUNK_SIZE = 1000

# Number of overlapping characters between chunks.

CHUNK_OVERLAP = 150


# ---------------------------------------------------------------------------
# LANGUAGE OPTIONS
# ---------------------------------------------------------------------------

LANGUAGE_OPTIONS = [
    "Auto",
    "English",
    "Hindi",
    "Hinglish",
]

DEFAULT_LANGUAGE = "Auto"