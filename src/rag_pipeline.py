"""
rag_pipeline.py — Phase 2, Step 3

What this file does:
Wires the retriever and the LLM together into a complete RAG chain.

The flow for every query:
  1. User asks a question
  2. Retriever finds the top-K relevant chunks from FAISS
  3. Chunks + question are inserted into a prompt template
  4. Groq LLM (Llama 3.1) generates a grounded answer
  5. Answer + source documents + latency are returned

Why a prompt template?
Without explicit instructions, the LLM might answer from its general
training knowledge and ignore your documents entirely. The template
forces it to only use the retrieved context, which prevents hallucination.
"""

import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.retriever import get_retriever

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TOP_K = int(os.getenv("TOP_K", 5))

# --- Prompt Template ---
# {context} = the retrieved chunks, injected automatically by LangChain
# {question} = the user's question, injected automatically
PROMPT_TEMPLATE = """You are a financial analyst assistant. Your job is to answer questions
accurately using ONLY the information provided in the context below.

Rules:
- If the answer is clearly in the context, answer it directly and concisely.
- If the context does not contain enough information, say: "I don't have enough information in the provided documents to answer this."
- Do not use any knowledge outside of the provided context.
- Always be factual — do not guess or estimate.

Context:
{context}

Question: {question}

Answer:"""


def build_rag_chain(top_k: int = TOP_K):
    """
    Build the full RAG chain.

    ChatGroq connects to Groq's API using your GROQ_API_KEY from .env.
    'llama-3.1-8b-instant' is fast and free. You can swap to
    'llama-3.1-70b-versatile' for better quality (still free, just slower).

    RetrievalQA is a LangChain chain that:
    - Takes a question
    - Runs the retriever to get context
    - Fills in the prompt template
    - Sends it to the LLM
    - Returns the answer

    chain_type="stuff" means all retrieved chunks are "stuffed" into
    one prompt. Fine for 5 chunks. For 20+ chunks, use "map_reduce".

    return_source_documents=True means the chain also returns which
    chunks it used — great for transparency and debugging.
    """
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.1,        # low temperature = more factual, less creative
        max_tokens=512,
        api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = get_retriever(top_k=top_k)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    print(f"✅ RAG chain ready (model={GROQ_MODEL}, top_k={top_k})")
    return chain


def query_rag(question: str, chain=None) -> dict:
    """
    Run a question through the RAG pipeline.

    Returns a dict with:
    - question: the original question
    - answer: the LLM's grounded answer
    - source_documents: list of chunk previews that were used
    - latency_seconds: total time taken (retrieval + LLM)

    We return all of this so the FastAPI layer (Phase 3) can
    format it for the API response, and MLflow (Phase 4) can
    log the latency as a metric.
    """
    if chain is None:
        chain = build_rag_chain()

    start_time = time.time()
    result = chain.invoke({"query": question})
    latency = round(time.time() - start_time, 3)

    # Extract source text previews (first 300 chars of each chunk)
    sources = [
        {
            "content_preview": doc.page_content[:300],
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?")
        }
        for doc in result.get("source_documents", [])
    ]

    return {
        "question": question,
        "answer": result["result"],
        "source_documents": sources,
        "latency_seconds": latency
    }


# --- Test the full pipeline when this file is run directly ---
if __name__ == "__main__":
    print("=== Testing RAG Pipeline ===\n")
    chain = build_rag_chain()

    test_questions = [
        "What was the total revenue?",
        "What are the main risk factors?",
        "How did operating income change?"
    ]

    for question in test_questions:
        print(f"\n❓ Question: {question}")
        result = query_rag(question, chain)
        print(f"💬 Answer: {result['answer']}")
        print(f"⏱️  Latency: {result['latency_seconds']}s")
        print(f"📄 Sources used: {len(result['source_documents'])}")
        print("-" * 60)
