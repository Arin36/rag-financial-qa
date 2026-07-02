# RAG Financial Q&A System

> Ask natural language questions over financial documents — powered by LangChain, FAISS, Groq (Llama 3.1), FastAPI, MLflow, DeepEval, and Docker.

![CI](https://github.com/Arin36/rag-financial-qa/actions/workflows/ci.yml/badge.svg)

---

## What This Does

Upload financial PDFs (annual reports, SEC filings, earnings calls) and ask questions like:
- *"What was the net revenue in Q4?"*
- *"What are the main risk factors?"*
- *"How did operating income change year-over-year?"*

The system retrieves the most relevant passages from your documents and generates a grounded answer — no hallucination, sources included.

---

## Architecture

```
PDFs → Chunk (512 tokens) → Embed (MiniLM-L6-v2) → FAISS Index
                                                          ↓
Query → Embed → Retrieve top-5 chunks → Groq LLaMA 3.1 → Answer
                                                          ↓
                                              MLflow logs run metrics
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq (Llama 3.1 8B Instant) — free |
| Judge LLM | Groq (Llama 3.3 70B Versatile) — used for evaluation |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 — local, free |
| Vector DB | FAISS — local, free |
| Orchestration | LangChain (LCEL) |
| API | FastAPI + Uvicorn |
| Experiment Tracking | MLflow (SQLite backend) |
| Evaluation | DeepEval (Faithfulness + Answer Relevancy) |
| Containerisation | Docker |
| CI/CD | GitHub Actions |

---

## Evaluation Results (DeepEval)

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | 1.00 | 0.70 | ✅ Pass |
| Answer Relevancy | 0.92 | 0.70 | ✅ Pass |

Evaluated on 3 financial questions across NVIDIA and Tesla documents.
Judge model: Groq Llama 3.3 70B Versatile (LLM-as-a-judge pattern).

---

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/Arin36/rag-financial-qa.git
cd rag-financial-qa

# 2. Set up environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Add secrets
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 4. Add financial PDFs to data/raw/
# Then build the index:
python -m src.ingestion

# 5. Run the API
uvicorn src.api:app --reload
```

API available at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

### Run with Docker

```bash
docker build -t rag-financial-qa .
docker run -p 8000:8000 --env-file .env rag-financial-qa
```

---

## Project Structure

```
rag-financial-qa/
├── .github/workflows/ci.yml   # GitHub Actions CI — lint + docker build
├── data/
│   ├── raw/                   # Your PDFs (gitignored)
│   └── faiss_index/           # Built FAISS vector index
├── src/
│   ├── ingestion.py           # Load PDFs, chunk, embed, save FAISS index
│   ├── retriever.py           # Load FAISS index, similarity search
│   ├── rag_pipeline.py        # LCEL chain: retriever → prompt → Groq LLM
│   ├── api.py                 # FastAPI endpoints: GET /health, POST /ask
│   ├── tracker.py             # MLflow experiment tracking
│   └── evaluate.py            # DeepEval evaluation pipeline
├── Dockerfile                 # Container definition (Python 3.11-slim)
├── .dockerignore              # Excludes venv, .env, raw PDFs from image
├── .env.example               # Environment variable template
└── requirements.txt           # Python dependencies
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check — returns `{"status": "ok"}` |
| POST | `/ask` | Ask a question, get an answer from your PDFs |

Example request:
```json
POST /ask
{
  "question": "What was NVIDIA's total revenue?"
}
```

Example response:
```json
{
  "question": "What was NVIDIA's total revenue?",
  "answer": "$215,938 million for the year ended Jan 25, 2026.",
  "latency_seconds": 0.411,
  "sources_used": 5
}
```

---

## Running Evaluation

```bash
python -m src.evaluate
```

---

## Author

Arindita Ghosh · [LinkedIn](https://linkedin.com/in/arindita-ghosh) · [GitHub](https://github.com/Arin36)
