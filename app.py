"""
Streamlit UI for "Chat with [CHARACTER_NAME]".

Flow:
    Landing page -> Disclaimer -> Accept -> Chat interface
    User question -> RAG retrieval -> Gemini -> Answer + Sources

Additional features:
    - Suggested political/development questions
    - Quick-question buttons above chat input
    - Bhagwan Mogi Ji style catchphrase
    - Special catchphrase questions
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------------

sys.path.append(
    str(Path(__file__).resolve().parent)
)


# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

from src.config import (
    CHARACTER_NAME,
    SPEECHES_PATH,
    LANGUAGE_OPTIONS,
    DEFAULT_LANGUAGE,
    TOP_K,
    TOP_K_OPTIONS,
)

from src import rag


# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=f"Chat with {CHARACTER_NAME}",
    page_icon="🤖",
    layout="centered",
)


# ---------------------------------------------------------------------------
# CHARACTER PHOTO / LOGO
# ---------------------------------------------------------------------------

LOGO_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "mogi_ji.png"
)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Fictional recurring catchphrase.
CATCHPHRASE = (
    "Angen gatram nayanena vaktram "
    "nayaena rajyam lavanena bhojyam"
)


# ---------------------------------------------------------------------------
# SUGGESTED QUESTIONS
# ---------------------------------------------------------------------------

SUGGESTED_QUESTIONS = [
    {
        "label": "🇮🇳 Manipur Issue",
        "question": "What do you say about the Manipur issue?",
    },
    {
        "label": "🎓 Student Protest",
        "question": "What do you say about student protests?",
    },
    {
        "label": "🏗️ Development",
        "question": "What does development really mean?",
    },
    {
        "label": "⚖️ Rape Cases",
        "question": (
            "What should be done about rape cases and women's safety?"
        ),
    },
    {
        "label": "💼 Unemployment",
        "question": (
            "What about unemployment among young people?"
        ),
    },
    {
        "label": "📚 Education",
        "question": (
            "What should be done to improve education?"
        ),
    },
    {
        "label": "🌾 Farmers",
        "question": "What should be done for farmers?",
    },
    {
        "label": "👨‍🎓 Youth",
        "question": "What is your message for the youth?",
    },
]


# ---------------------------------------------------------------------------
# SPECIAL CATCHPHRASE QUESTIONS
# ---------------------------------------------------------------------------

# These are the ONLY questions that automatically receive the
# fictional catchphrase instead of a Gemini-generated answer.

SPECIAL_CATCHPHRASE_QUESTIONS = {
    item["question"].strip().lower()
    for item in SUGGESTED_QUESTIONS
}


def is_special_catchphrase_question(
    question: str,
) -> bool:
    """
    Check whether the question is one of the predefined
    suggested questions.

    Matching is case-insensitive and ignores extra spaces.
    """

    if not isinstance(question, str):
        return False

    normalized_question = " ".join(
        question.strip().lower().split()
    )

    return (
        normalized_question
        in SPECIAL_CATCHPHRASE_QUESTIONS
    )


# ---------------------------------------------------------------------------
# SESSION STATE DEFAULTS
# ---------------------------------------------------------------------------

if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False


if "messages" not in st.session_state:
    st.session_state.messages = []


if "language" not in st.session_state:
    st.session_state.language = DEFAULT_LANGUAGE


if "top_k" not in st.session_state:
    st.session_state.top_k = TOP_K


# ---------------------------------------------------------------------------
# CONFIGURATION CHECKS
# ---------------------------------------------------------------------------

def render_config_warnings() -> bool:
    """
    Show blocking warnings for missing setup.

    Returns:
        True if everything required is available.
    """

    ok = True

    # -------------------------------------------------------
    # API KEY
    # -------------------------------------------------------

    if not os.getenv("GOOGLE_API_KEY"):

        st.warning(
            "⚠️ Please configure your GOOGLE_API_KEY "
            "in the .env file."
        )

        ok = False

    # -------------------------------------------------------
    # TRANSCRIPTS
    # -------------------------------------------------------

    txt_files = (
        list(SPEECHES_PATH.glob("*.txt"))
        if SPEECHES_PATH.exists()
        else []
    )

    if not txt_files:

        st.warning(
            "⚠️ No transcript files found.\n\n"
            "Please add `.txt` files to:\n\n"
            "`data/speeches/`"
        )

        ok = False

    # -------------------------------------------------------
    # VECTORSTORE
    # -------------------------------------------------------

    if ok and not rag.vectorstore_exists():

        st.warning(
            "⚠️ Knowledge base not found.\n\n"
            "Please run:\n\n"
            "```bash\n"
            "python src/ingest.py\n"
            "```"
        )

        ok = False

    return ok


# ---------------------------------------------------------------------------
# LANDING PAGE + DISCLAIMER
# ---------------------------------------------------------------------------

def render_landing_page():
    """Render the landing page and disclaimer."""

    if LOGO_PATH.exists():
        st.image(
            str(LOGO_PATH),
            width=150,
        )

    st.title(
        f"🤖 Chat with {CHARACTER_NAME}"
    )

    st.caption(
        f'"Ask anything. {CHARACTER_NAME} has something to say."'
    )

    st.markdown("### ⚠️ Important Disclaimer")

    st.markdown(
        """
        **Chat with Bhagwan Mogi Ji** is a fictional AI character created
        solely for **entertainment, parody, and humorous purposes**.

        This chatbot is **not any real person, including the Prime Minister
        of any country**, and does not represent, speak for, or impersonate
        any real person. The responses generated by this chatbot are
        **AI-generated and fictional** and should not be interpreted as the
        actual statements, beliefs, opinions, intentions, or views of any
        real individual.

        The chatbot may generate humorous, exaggerated, satirical, or
        fictional responses that do not reflect reality.

        **Religious & Cultural Disclaimer:**

        Some responses may contain references to religions, religious
        beliefs, traditions, communities, cultures, or sensitive topics.
        These responses are generated by AI for entertainment and **are not
        intended to insult, disrespect, mock, or hurt anyone's religious
        beliefs or sentiments**. If any response hurts your religious or
        cultural sentiments, please understand that it is fictional
        AI-generated content and was not intended to target or offend any
        individual, religion, or community.

        By continuing, you acknowledge that you understand and accept these
        terms.
        """
    )

    accepted = st.checkbox(
        "☑️ I Understand & Accept"
    )

    if accepted:

        st.session_state.disclaimer_accepted = True

        st.rerun()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render sidebar controls."""

    with st.sidebar:

        if LOGO_PATH.exists():
            st.image(
                str(LOGO_PATH),
                width=100,
            )

        st.header(
            f"🤖 {CHARACTER_NAME}"
        )

        # ---------------------------------------------------
        # LANGUAGE
        # ---------------------------------------------------

        st.subheader(
            "Response Language"
        )

        st.session_state.language = (
            st.selectbox(
                "Response Language",
                options=LANGUAGE_OPTIONS,
                index=LANGUAGE_OPTIONS.index(
                    st.session_state.language
                ),
                label_visibility="collapsed",
            )
        )

        # ---------------------------------------------------
        # RAG SETTINGS
        # ---------------------------------------------------

        st.subheader(
            "RAG Settings"
        )

        st.text(
            "Knowledge Base: FAISS"
        )

        st.text(
            "Retriever: Enabled"
        )

        st.session_state.top_k = (
            st.selectbox(
                "Number of sources",
                options=TOP_K_OPTIONS,
                index=(
                    TOP_K_OPTIONS.index(
                        st.session_state.top_k
                    )
                    if st.session_state.top_k
                    in TOP_K_OPTIONS
                    else TOP_K_OPTIONS.index(
                        TOP_K
                    )
                ),
            )
        )

        st.divider()

        # ---------------------------------------------------
        # CLEAR CHAT
        # ---------------------------------------------------

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True,
        ):

            st.session_state.messages = []

            st.rerun()


# ---------------------------------------------------------------------------
# SOURCE DISPLAY
# ---------------------------------------------------------------------------

def render_sources(sources: list):
    """Display retrieved RAG sources."""

    if not sources:
        return

    with st.expander(
        "📚 View Retrieved Context"
    ):

        last_source_name = None

        for chunk in sources:

            if (
                chunk.source
                != last_source_name
            ):

                st.markdown(
                    f"**Source: {chunk.source}**"
                )

                last_source_name = (
                    chunk.source
                )

            st.text(
                chunk.content
            )

            st.markdown("---")


# ---------------------------------------------------------------------------
# SUGGESTED QUESTIONS
# ---------------------------------------------------------------------------

def render_suggested_questions():
    """
    Display suggested questions above the chat input.

    Returns:
        The selected question or None.
    """

    st.markdown(
        "### 💡 Try asking"
    )

    selected_question = None

    # -------------------------------------------------------
    # FIRST ROW
    # -------------------------------------------------------

    row1 = st.columns(4)

    for column, suggestion in zip(
        row1,
        SUGGESTED_QUESTIONS[:4],
    ):

        with column:

            if st.button(
                suggestion["label"],
                use_container_width=True,
            ):

                selected_question = (
                    suggestion["question"]
                )

    # -------------------------------------------------------
    # SECOND ROW
    # -------------------------------------------------------

    row2 = st.columns(4)

    for column, suggestion in zip(
        row2,
        SUGGESTED_QUESTIONS[4:8],
    ):

        with column:

            if st.button(
                suggestion["label"],
                use_container_width=True,
            ):

                selected_question = (
                    suggestion["question"]
                )

    return selected_question


# ---------------------------------------------------------------------------
# SPECIAL CATCHPHRASE RESPONSE
# ---------------------------------------------------------------------------

def generate_special_catchphrase_response():
    """
    Return the predefined fictional catchphrase.

    IMPORTANT:
    This function does NOT call Gemini.
    """

    return (
        f"**{CATCHPHRASE}**"
    )


# ---------------------------------------------------------------------------
# PROCESS USER QUESTION
# ---------------------------------------------------------------------------

def process_question(question: str):
    """
    Process a user question.

    There are two possible paths:

    1. Special predefined question
       -> Return the catchphrase immediately.

    2. Normal question
       -> RAG retrieval -> Gemini -> Answer + Sources.
    """

    # -------------------------------------------------------
    # CLEAN QUESTION
    # -------------------------------------------------------

    question = question.strip()

    if not question:
        return

    # -------------------------------------------------------
    # SAVE USER MESSAGE
    # -------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # -------------------------------------------------------
    # DISPLAY USER MESSAGE
    # -------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # -------------------------------------------------------
    # SPECIAL CATCHPHRASE QUESTIONS
    # -------------------------------------------------------

    if is_special_catchphrase_question(
        question
    ):

        with st.chat_message(
            "assistant",
            avatar=(
                str(LOGO_PATH)
                if LOGO_PATH.exists()
                else "🤖"
            ),
        ):

            special_answer = (
                generate_special_catchphrase_response()
            )

            st.markdown(
                special_answer
            )

        # Save the catchphrase in chat history.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": special_answer,
                "sources": [],
            }
        )

        # VERY IMPORTANT:
        # Do not call Gemini or RAG.
        return

    # -------------------------------------------------------
    # NORMAL GEMINI + RAG RESPONSE
    # -------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar=(
            str(LOGO_PATH)
            if LOGO_PATH.exists()
            else "🤖"
        ),
    ):

        with st.spinner(
            f"🤔 {CHARACTER_NAME} is thinking..."
        ):

            try:

                response = rag.get_answer(
                    question=question,

                    # Don't include the current question
                    # because it was just added above.
                    chat_history=(
                        st.session_state.messages[
                            :-1
                        ]
                    ),

                    language=(
                        st.session_state.language
                    ),

                    top_k=(
                        st.session_state.top_k
                    ),
                )

                # ---------------------------------------------------
                # DISPLAY ANSWER
                # ---------------------------------------------------

                st.markdown(
                    response.answer
                )

                # ---------------------------------------------------
                # DISPLAY SOURCES
                # ---------------------------------------------------

                render_sources(
                    response.sources
                )

                # ---------------------------------------------------
                # SAVE ASSISTANT RESPONSE
                # ---------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                    }
                )

            except Exception as e:

                error_message = (
                    "⚠️ Something went wrong while "
                    f"talking to Gemini: {e}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )


# ---------------------------------------------------------------------------
# CHAT INTERFACE
# ---------------------------------------------------------------------------

def render_chat_interface():

    st.title(
        f"🤖 Chat with {CHARACTER_NAME}"
    )

    st.caption(
        "Ask anything — serious, political, random, "
        "or completely unnecessary. 😌"
    )

    # -------------------------------------------------------
    # REPLAY CONVERSATION HISTORY
    # -------------------------------------------------------

    for msg in st.session_state.messages:

        if msg["role"] == "assistant":
            avatar = (
                str(LOGO_PATH)
                if LOGO_PATH.exists()
                else "🤖"
            )
        else:
            avatar = "👤"

        with st.chat_message(
            msg["role"],
            avatar=avatar,
        ):

            st.markdown(
                msg["content"]
            )

            if (
                msg["role"] == "assistant"
                and msg.get("sources")
            ):

                render_sources(
                    msg["sources"]
                )

    # -------------------------------------------------------
    # SUGGESTED QUESTIONS
    # -------------------------------------------------------

    selected_question = (
        render_suggested_questions()
    )

    # -------------------------------------------------------
    # NORMAL CHAT INPUT
    # -------------------------------------------------------

    typed_question = st.chat_input(
        f"Ask {CHARACTER_NAME} something..."
    )

    # -------------------------------------------------------
    # DETERMINE QUESTION
    # -------------------------------------------------------

    question = None

    # Suggested button takes priority.
    if selected_question:

        question = selected_question

    elif typed_question:

        question = typed_question

    # -------------------------------------------------------
    # PROCESS QUESTION
    # -------------------------------------------------------

    if question:

        process_question(
            question
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    # -------------------------------------------------------
    # DISCLAIMER
    # -------------------------------------------------------

    if not st.session_state.disclaimer_accepted:

        render_landing_page()

        return

    # -------------------------------------------------------
    # CONFIGURATION CHECK
    # -------------------------------------------------------

    if not render_config_warnings():

        st.stop()

    # -------------------------------------------------------
    # SIDEBAR
    # -------------------------------------------------------

    render_sidebar()

    # -------------------------------------------------------
    # CHAT
    # -------------------------------------------------------

    render_chat_interface()


# ---------------------------------------------------------------------------
# RUN APPLICATION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()