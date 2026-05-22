# Document Q&A Chatbot
> Ask any question about any PDF and get grounded, cited answers — built with RAG (Retrieval-Augmented Generation).

🔗 **[Live Demo](https://omsingh-document-chatbot.streamlit.app/)** · 👤 **[Om Sanjaysingh Bais](https://www.linkedin.com/in/omsingh-bais-77312b22a/)** · 📧 omthakur0718@gmail.com

---

## The Problem

Professionals waste hours searching through long PDFs — contracts, HR policies, product manuals — hunting for a single answer. Ctrl+F fails when the wording doesn't match exactly. It doesn't understand meaning.

This chatbot reads your document, understands it semantically, and answers questions in plain English — with the exact source chunks shown alongside every answer so users can verify.

---

## Architecture

```
PDF Upload
    ↓
pdfplumber
(text extraction + table detection)
    ↓
Table blocks → markdown  |  Text blocks → RecursiveCharacterTextSplitter
(never split)             |  (500 chars, 50 overlap)
    ↓
merge_split_tables()
(stitches tables that span two pages)
    ↓
sentence-transformers → all-MiniLM-L6-v2
(384-dimensional embeddings)
    ↓
ChromaDB — persisted vector store
(cosine similarity search)
    ↓
User Question → Embed → Top N chunks retrieved
    ↓
LLM — Ollama locally / Groq API on cloud
(grounded answer using only document context)
    ↓
Streamlit UI — answer + source chunks displayed
```

---

## Key Features

**Table extraction that actually works**
Most PDF libraries read left-to-right and destroy table structure — columns merge into unreadable blobs. This project uses pdfplumber to detect table regions, extract structured cell data, and convert it to clean markdown. Tables are stored as atomic chunks and never split by the chunker.

**Cross-page table merging**
pdfplumber is page-aware, not document-aware. A table split across two pages arrives as two broken fragments — the second with no header row. A custom `merge_split_tables()` function detects consecutive fragments with matching column counts, takes the header from the first, combines all data rows, and stitches them into one correct table.

**Source transparency**
Every answer shows the exact chunks it was built from — page number, source file, and raw text. Users can verify every claim. Clients love this because it builds trust instantly.

**Input guard during generation**
The chat input disables while the LLM is generating, preventing race conditions and out-of-order answers. Built using Streamlit's session state with a pending question pattern.

**Switchable LLM backend**
One environment variable controls which LLM runs. `LLM_PROVIDER=ollama` for local development (free, no internet). `LLM_PROVIDER=groq` for cloud deployment. Zero code changes required to switch.

---

## Tech Stack

| Layer | Tool |
|---|---|
| PDF Extraction | pdfplumber |
| Text Splitting | langchain-text-splitters — RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers — all-MiniLM-L6-v2 |
| Vector Store | ChromaDB (persisted to disk) |
| LLM — Local | Ollama + gemma4 |
| LLM — Cloud | Groq API + llama-3.1-8b-instant |
| UI | Streamlit |
| Language | Python 3.12 |

---

## Project Structure

```
week-01-document-chatbot/
├── data/                          # Drop PDF files here
├── notebooks/                     # Embedding experiments
├── src/
│   ├── ingest.py                  # PDF → chunks → ChromaDB
│   ├── retriever.py               # Question → top N chunks
│   ├── chatbot.py                 # LLM wrapper (Ollama + Groq)
│   └── app.py                     # Streamlit UI
├── tests/
│   ├── test_args.py
|   ├──test_personas.py
|   ├──test_tetrieval.py
|   ├──test_temperature.py
├── vectorstore/                   # ChromaDB persisted data
├── .env                           # API keys — never commit
├── requirements.txt
└── README.md
```

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/week-01-document-chatbot.git
cd week-01-document-chatbot
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
# Fill in your values
```

`.env` file:
```env
LLM_PROVIDER=ollama          # or "groq" for cloud
GROQ_API_KEY=your_key_here   # only needed if using groq
```

**5. Start Ollama (local mode only)**
```bash
ollama run gemma4
```

**6. Ingest a PDF**
```bash
python src/ingest.py --pdf data/yourfile.pdf
```

**7. Run the app**
```bash
streamlit run src/app.py
```

---

## Challenges Solved

| Challenge | Root Cause | Fix |
|---|---|---|
| Tables extracted as garbled text | pypdf reads left-to-right, destroying column structure | Replaced with pdfplumber — detects and extracts table cells as structured grids |
| Table split across two PDF pages | pdfplumber is page-aware, not document-aware | `merge_split_tables()` — detects fragments by matching column count, stitches them |
| Sources expander in wrong position | Rendered after all messages, not attached to each answer | Chunks stored inside each message dict, `render_sources()` called inside history loop |
| Sources expander opened blank | Loop iterated over wrong session state variable | Fixed loop to iterate over the `chunks` parameter, not the stale session variable |
| User could submit Q2 before Q1 finished | `st.chat_input` always enabled | Pending question pattern — `is_thinking` flag disables input during generation |

---

## What I Learned

**Why keyword search fails** — searching "explain ML" won't find a chunk that says "what is machine learning" even though the meaning is identical. Embeddings convert both to similar vectors so semantic search finds the match.

**Why chunking strategy matters** — chunk_overlap=50 means the second chunk backs up 50 characters from where the first ended. Without overlap, sentences cut at boundaries lose all context and the LLM can't answer correctly.

**Streamlit's rerun cycle** — Streamlit reruns the entire script top-to-bottom on every interaction. Widget state only reflects new values on the next rerun, not mid-script. The pending question pattern is the only reliable way to disable input before processing starts.

**pdfplumber vs pypdf** — pypdf reads characters left-to-right across the full page width. A table with 3 columns becomes one unreadable row of merged text. pdfplumber understands that tables are regions and extracts them as structured cell grids.

---

## Client Use Cases

- **Legal teams** — search contracts, NDAs, and case documents without reading 80 pages
- **HR departments** — answer employee policy questions at scale without manual lookup
- **Product teams** — build support bots from technical manuals and documentation
- **Recruitment agencies** — search through job descriptions and compliance documents

**Revenue potential: ₹25,000 – ₹80,000 per project**

---

## Non-Negotiable Rules (Personal Sprint Rules)

- Never commit `.env` — API keys go in `.env`, `.env` goes in `.gitignore`
- Always build paths from `__file__` — never relative paths that break on different machines
- Always use `get_or_create_collection` — prevents duplicate collection crashes on re-ingestion
- Embedding model must match between `ingest.py` and `retriever.py` — different model = wrong vector space = garbage results

---

*Built by Om Sanjaysingh Bais — omthakur0718@gmail.com*
