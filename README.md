# RAG Financial Q&A System

> Ask natural language questions over financial documents — powered by LangChain, FAISS, Groq (Llama 3.1), FastAPI, MLflow, and Docker.

![CI](https://github.com/YOUR_USERNAME/rag-financial-qa/actions/workflows/ci.yml/badge.svg)

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
| LLM | Groq (Llama 3.1 8B) — free |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 — local, free |
| Vector DB | FAISS — local, free |
| Orchestration | LangChain |
| API | FastAPI + Uvicorn |
| Experiment Tracking | MLflow |
| Evaluation | RAGAS |
| Containerisation | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Cloud | AWS (ECR + EC2/SageMaker) |

---

## Evaluation Results (RAGAS)

| Metric | Score |
|---|---|
| Faithfulness | TBD |
| Answer Relevancy | TBD |
| Context Precision | TBD |

---

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/YOUR_USERNAME/rag-financial-qa.git
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
python src/ingestion.py

# 5. Run everything with Docker
docker compose up
```

API available at: `http://localhost:8000`
MLflow UI at: `http://localhost:5000`
API docs at: `http://localhost:8000/docs`

---

## Project Structure

```
rag-financial-qa/
├── .github/workflows/ci.yml   # GitHub Actions CI
├── data/
│   └── raw/                   # Your PDFs (gitignored)
├── src/
│   ├── ingestion.py           # Load, chunk, embed, index
│   ├── retriever.py           # FAISS retrieval
│   ├── rag_pipeline.py        # Full RAG chain
│   ├── api.py                 # FastAPI app
│   ├── tracker.py             # MLflow logging
│   └── evaluate.py            # RAGAS evaluation
├── tests/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Author

Arindita Ghosh · [LinkedIn](https://linkedin.com/in/arindita-ghosh) · [GitHub](https://github.com/Arin36)
