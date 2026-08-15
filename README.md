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



- Conversations are kept only in Streamlit session state (in memory) —
  nothing is permanently stored, and refreshing the browser resets it.
- If you change `EMBEDDING_MODEL`, delete `vectorstore/` and re-run
  `python src/ingest.py`, since old embeddings won't be compatible.
