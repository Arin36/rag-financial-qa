"""
retriever.py — Phase 2, Step 2

What this file does:
Given a user's question, finds the most relevant chunks from the FAISS index.

How it works:
1. The question gets converted to a vector using the same embedding model
2. FAISS searches for the top-K most similar chunk vectors
3. Returns those chunks as context for the LLM

Why the same embedding model?
The question and the document chunks must live in the same vector space.
If you embed documents with Model A and questions with Model B, the
similarity scores mean nothing. Always use the same model for both.
"""

import os
from dotenv import load_dotenv
from src.ingestion import load_faiss_index

load_dotenv()

TOP_K = int(os.getenv("TOP_K", 5))


def get_retriever(top_k: int = TOP_K):
    """
    Build and return a LangChain retriever backed by the FAISS index.

    as_retriever() wraps the FAISS vectorstore in a standard LangChain
    interface — this means it plugs directly into any LangChain chain
    without any extra glue code.

    search_kwargs={"k": top_k} tells it to return the top K most
    similar chunks for every query.
    """
    vectorstore = load_faiss_index()
    retriever = vectorstore.as_retriever(
        search_type="similarity",      # cosine similarity search
        search_kwargs={"k": top_k}
    )
    print(f"✅ Retriever ready (top_k={top_k})")
    return retriever


def retrieve_chunks(question: str, top_k: int = TOP_K) -> list:
    """
    Retrieve the most relevant chunks for a given question.

    Useful for testing retrieval quality independently of the LLM.
    Call this to verify the right passages are being found before
    wiring in the LLM.
    """
    retriever = get_retriever(top_k=top_k)
    docs = retriever.invoke(question)

    print(f"\n📄 Top {len(docs)} chunks for: '{question}'")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"\n  [{i+1}] Source: {source} | Page: {page}")
        print(f"       {doc.page_content[:200]}...")

    return docs


# --- Test retrieval when this file is run directly ---
if __name__ == "__main__":
    question = "What was the total revenue?"
    retrieve_chunks(question)
