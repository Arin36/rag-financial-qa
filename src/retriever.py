import os
from dotenv import load_dotenv
from src.ingestion import load_faiss_index
load_dotenv()
TOP_K=int(os.getenv("TOP_K", 5))

def get_retriever(top_k: int=TOP_K):
    vectorstore=load_faiss_index()
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k":top_k}
    )
    print(f"retriever ready (top_k={top_k})")
    return retriever

def retrieve_chunks(question: str, top_k:int =TOP_K):
    retriever = get_retriever (top_k=top_k)
    docs= retriever.invoke(question)
    print(f"retrieved {len(docs)} chunks for : {question}")
    return docs

if __name__ == "__main__":
    print("=== Testing Retriever ===")
    test_question = "What was the total revenue?"
    chunks = retrieve_chunks(test_question)
    for i, doc in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(doc.page_content[:200])
        print(f"Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', '?')}")