# 🤖 Chat with [CHARACTER_NAME]

A fun, fictional AI chatbot project. It uses **RAG (Retrieval-Augmented
Generation)** over a collection of speech transcripts to answer questions
"in character" as a fictional persona called `[CHARACTER_NAME]`.

> ⚠️ **Disclaimer:** This is a fictional AI chatbot created for entertainment
> purposes. The responses are AI-generated and may not represent the views
> or statements of any real person.

## Features

- 🔍 Retrieval-Augmented Generation over your own `.txt` transcripts
- 🗂️ FAISS vector database (built once via a separate ingestion script)
- 🔗 LangChain for orchestration
- ✨ Google Gemini for response generation
- 💬 Streamlit chat UI with conversation memory
- 🇮🇳 English / Hindi / Hinglish response modes
- 📚 Source chunks shown alongside every answer
- 🎭 Fictional, customizable persona

## Project Structure

```text
chat-with-character/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── data/
│   └── speeches/
│       └── Put_your_txt_files_here.txt
│
├── vectorstore/
│
└── src/
    ├── __init__.py
    ├── ingest.py
    ├── rag.py
    ├── prompts.py
    └── config.py
```

## 1. Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Configure Gemini

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
GOOGLE_API_KEY=YOUR_KEY
```

You can get a Gemini API key from Google AI Studio.

## 3. Add Your Transcript Files

Drop your `.txt` speech transcripts into:

```text
data/speeches/
```

Any number of files is fine — the app auto-discovers all `.txt` files in
that folder. If a filename contains a year or a date like `2022` or
`2019-08-15`, that gets captured as metadata automatically.

## 4. Build the FAISS Knowledge Base

Run the ingestion script whenever you add or change transcripts:

```bash
python src/ingest.py
```

This loads the transcripts, cleans and splits them into chunks, embeds
them, and saves a FAISS index into `vectorstore/`. The Streamlit app does
**not** rebuild this at startup — it just loads what's already there.

## 5. Run the App

```bash
streamlit run app.py
```

Accept the disclaimer, and start chatting.

## How the RAG Pipeline Works

```text
TXT files -> Document Loader -> Text Cleaning -> Text Splitting
          -> Embeddings -> FAISS Vector Database -> Retriever
          -> Relevant Chunks -> Prompt -> Gemini -> Response
```

1. `src/ingest.py` (offline, run manually): loads every `.txt` file,
   cleans whitespace, splits into overlapping chunks, embeds them, and
   saves a FAISS index to `vectorstore/`.
2. `src/rag.py` (used live by the app): loads that FAISS index, embeds the
   user's question, retrieves the top-K most relevant chunks, and builds a
   prompt combining those chunks + recent chat history + the question.
3. Gemini generates a response **in character**, grounded in the retrieved
   material where relevant. If nothing relevant was retrieved, the persona
   says so rather than making something up.
4. `app.py` displays the answer and, in an expander, the exact source
   chunks that were used — so it's obvious the chatbot is actually doing
   RAG and not just freely hallucinating.

## Customizing `[CHARACTER_NAME]`

Open `src/config.py` and change:

```python
CHARACTER_NAME = "[CHARACTER_NAME]"
```

to whatever name you want. It's used everywhere — titles, chat labels,
placeholders, and the persona prompt in `src/prompts.py` — so this one
change is enough. If you want to go further, you can also edit the
personality bullet points directly in `src/prompts.py`.

## Modifying the Disclaimer Later

The disclaimer text and checkbox live in `app.py`, inside the
`render_landing_page()` function:

```python
st.info(
    "⚠️ **Disclaimer:** ..."
)
accepted = st.checkbox("☑️ I understand")
```

Edit the string passed to `st.info(...)` to change the wording — nothing
else needs to change.

## Configuration Reference (`src/config.py`)

| Setting             | Purpose                                             |
|----------------------|------------------------------------------------------|
| `CHARACTER_NAME`     | The persona's display name                          |
| `MODEL_NAME`         | Gemini chat model                                    |
| `EMBEDDING_MODEL`    | Embedding model (must match between ingest & app)    |
| `CHUNK_SIZE`         | Characters per chunk                                 |
| `CHUNK_OVERLAP`      | Overlap between chunks                               |
| `TOP_K`              | Default number of retrieved chunks                   |
| `SPEECHES_PATH`      | Where `.txt` transcripts live                        |
| `VECTORSTORE_PATH`   | Where the FAISS index is saved                       |

## Notes

- Conversations are kept only in Streamlit session state (in memory) —
  nothing is permanently stored, and refreshing the browser resets it.
- If you change `EMBEDDING_MODEL`, delete `vectorstore/` and re-run
  `python src/ingest.py`, since old embeddings won't be compatible.
