# 💼 Resume Chatbot

A lightweight chatbot that answers questions about your professional experience using RAG (Retrieval-Augmented Generation).

## Features

- 🔍 Semantic search over your resume and experience documents
- 💬 Conversational interface with chat history
- 🌐 Web UI powered by Streamlit
- ⚡ Fast local vector store with FAISS
- 🧠 GPT-4o-mini for natural language responses

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key

Copy the example environment file and add your OpenAI API key:

```bash
copy .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-your-actual-api-key
```

### 3. Add Your Documents

Place your resume and experience files in the `data/` folder. Supported formats:
- `.txt` - Plain text
- `.md` - Markdown

### 4. Run the Chatbot

**Web Interface (recommended):**
```bash
streamlit run app.py
```

**Command Line:**
```bash
python chatbot.py
```

## Project Structure

```
code-resume-chatbot/
├── data/              # Your resume files go here
├── vectorstore/       # Generated vector index (cached)
├── app.py             # Streamlit web interface
├── chatbot.py         # Core chatbot logic
├── requirements.txt   # Python dependencies
├── .env.example       # API key template
└── .gitignore
```

## How It Works

1. **Document Loading** - Reads all `.txt` and `.md` files from `data/`
2. **Text Chunking** - Splits documents into ~1000 character chunks with overlap
3. **Embedding** - Converts chunks to vectors using OpenAI embeddings
4. **Indexing** - Stores vectors in a local FAISS index
5. **Retrieval** - Finds the 3 most relevant chunks for each question
6. **Generation** - GPT-4o-mini generates an answer using the retrieved context

## Rebuilding the Index

If you update your documents, delete the `vectorstore/` folder and restart:

```bash
rmdir /s /q vectorstore
streamlit run app.py
```

## Customization

Edit `chatbot.py` to customize:

| Setting | Location | Default |
|---------|----------|---------|
| Chunk size | `create_vectorstore()` | 1000 |
| Chunk overlap | `create_vectorstore()` | 200 |
| Retrieved chunks | `create_chatbot()` | 3 |
| LLM model | `create_chatbot()` | gpt-4o-mini |
| Temperature | `create_chatbot()` | 0.7 |

## License

MIT
