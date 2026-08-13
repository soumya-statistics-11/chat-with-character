"""
Core RAG functionality.

Responsible for:
    - Loading FAISS
    - Building the retriever
    - Retrieving transcript chunks
    - Calling Gemini
    - Formatting Gemini responses
    - Returning generated answers and source chunks

app.py only handles the Streamlit UI.
"""

from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from src.config import (
    VECTORSTORE_PATH,
    EMBEDDING_MODEL,
    MODEL_NAME,
    TEMPERATURE,
    TOP_K,
)

from src.prompts import get_system_prompt


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class SourceChunk:
    """A single retrieved chunk displayed in the UI."""

    source: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Response returned from the RAG pipeline."""

    answer: str
    sources: list


# ---------------------------------------------------------------------------
# VECTORSTORE
# ---------------------------------------------------------------------------

def vectorstore_exists() -> bool:
    """Check whether the FAISS index exists."""

    return (
        VECTORSTORE_PATH / "index.faiss"
    ).exists()


def load_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Load the same embedding model used during ingestion.

    IMPORTANT:
    This must match the model used by src/ingest.py.
    """

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )


def load_vectorstore() -> FAISS:
    """Load the existing FAISS vector database."""

    embeddings = load_embeddings()

    return FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ---------------------------------------------------------------------------
# GEMINI
# ---------------------------------------------------------------------------

def load_llm() -> ChatGoogleGenerativeAI:
    """
    Load the Gemini chat model.

    For newer Gemini models, sampling parameters such as temperature
    are deprecated. Therefore we only pass the model name here.
    """

    return ChatGoogleGenerativeAI(
        model=str(MODEL_NAME),
    )


# ---------------------------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------------------------

def format_chat_history(
    messages: list,
    max_turns: int = 6,
) -> str:
    """
    Convert Streamlit's message list into a compact text block.

    Only the most recent max_turns exchanges are included.
    """

    if not messages:
        return "(no previous conversation)"

    recent = messages[
        -(max_turns * 2):
    ]

    lines = []

    for msg in recent:

        role = (
            "User"
            if msg["role"] == "user"
            else "Assistant"
        )

        content = msg.get(
            "content",
            "",
        )

        # Make sure content is a string.
        if not isinstance(content, str):
            content = str(content)

        lines.append(
            f"{role}: {content}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RETRIEVED CONTEXT
# ---------------------------------------------------------------------------

def format_retrieved_context(
    docs: list,
) -> str:
    """
    Convert retrieved LangChain documents into
    text that can be inserted into the prompt.
    """

    if not docs:
        return (
            "(no relevant context was found "
            "in the transcript collection)"
        )

    parts = []

    for i, doc in enumerate(
        docs,
        start=1,
    ):

        source = doc.metadata.get(
            "source",
            "unknown",
        )

        content = doc.page_content

        if not isinstance(content, str):
            content = str(content)

        parts.append(
            f"[Chunk {i} — source: {source}]\n"
            f"{content}"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# GEMINI RESPONSE CONVERSION
# ---------------------------------------------------------------------------

def extract_text_from_response(
    response,
) -> str:
    """
    Convert Gemini/LangChain response content into a plain string.

    Depending on the version of langchain-google-genai,
    response.content can be:

        1. A normal string
        2. A list of strings
        3. A list of dictionaries/content blocks

    This function handles all three cases.
    """

    content = getattr(
        response,
        "content",
        "",
    )

    # -------------------------------------------------------
    # CASE 1: NORMAL STRING
    # -------------------------------------------------------

    if isinstance(content, str):
        return content.strip()

    # -------------------------------------------------------
    # CASE 2: LIST
    # -------------------------------------------------------

    if isinstance(content, list):

        text_parts = []

        for item in content:

            # A plain string inside the list
            if isinstance(item, str):

                text_parts.append(item)

                continue

            # Dictionary content block
            if isinstance(item, dict):

                text = item.get(
                    "text"
                )

                if text:
                    text_parts.append(
                        str(text)
                    )

                continue

            # Object content block
            text = getattr(
                item,
                "text",
                None,
            )

            if text:
                text_parts.append(
                    str(text)
                )

        return "\n".join(
            text_parts
        ).strip()

    # -------------------------------------------------------
    # CASE 3: SOMETHING UNEXPECTED
    # -------------------------------------------------------

    return str(content).strip()


# ---------------------------------------------------------------------------
# MAIN RAG FUNCTION
# ---------------------------------------------------------------------------

def get_answer(
    question: str,
    chat_history: list,
    language: str = "Auto",
    top_k: int = TOP_K,
) -> RAGResponse:
    """
    Run one complete RAG turn:

        1. Load FAISS
        2. Retrieve relevant transcript chunks
        3. Build persona prompt
        4. Call Gemini
        5. Convert Gemini response to plain text
        6. Return answer + source chunks
    """

    # -------------------------------------------------------
    # VALIDATE QUESTION
    # -------------------------------------------------------

    if not isinstance(question, str):
        question = str(question)

    question = question.strip()

    if not question:
        return RAGResponse(
            answer="Please ask me something.",
            sources=[],
        )

    # -------------------------------------------------------
    # LOAD VECTORSTORE
    # -------------------------------------------------------

    vectorstore = load_vectorstore()

    # -------------------------------------------------------
    # RETRIEVER
    # -------------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": int(top_k),
        }
    )

    # -------------------------------------------------------
    # RETRIEVE RELEVANT DOCUMENTS
    # -------------------------------------------------------

    retrieved_docs = retriever.invoke(
        question
    )

    # -------------------------------------------------------
    # LOAD GEMINI
    # -------------------------------------------------------

    llm = load_llm()

    # -------------------------------------------------------
    # SYSTEM PROMPT
    # -------------------------------------------------------

    system_prompt = get_system_prompt(
        language=language
    )

    # -------------------------------------------------------
    # FORMAT PROMPT
    # -------------------------------------------------------

    filled_prompt = system_prompt.format(
        context=format_retrieved_context(
            retrieved_docs
        ),

        chat_history=format_chat_history(
            chat_history
        ),

        question=question,
    )

    # -------------------------------------------------------
    # CALL GEMINI
    # -------------------------------------------------------

    response = llm.invoke(
        filled_prompt
    )

    # -------------------------------------------------------
    # EXTRACT TEXT
    # -------------------------------------------------------

    answer_text = extract_text_from_response(
        response
    )

    # -------------------------------------------------------
    # FALLBACK
    # -------------------------------------------------------

    if not answer_text:

        answer_text = (
            "I received an empty response. "
            "Please try asking the question again."
        )

    # -------------------------------------------------------
    # SOURCES
    # -------------------------------------------------------

    sources = []

    for doc in retrieved_docs:

        sources.append(
            SourceChunk(
                source=doc.metadata.get(
                    "source",
                    "unknown",
                ),

                content=doc.page_content,

                metadata=doc.metadata,
            )
        )

    # -------------------------------------------------------
    # RETURN
    # -------------------------------------------------------

    return RAGResponse(
        answer=answer_text,
        sources=sources,
    )