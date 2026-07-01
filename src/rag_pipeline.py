import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from src.retriever import get_retriever
load_dotenv()

GROQ_MODEL=os.getenv("GROQ_MODEL","llama-3.1-8b-instant")
TOP_K=int (os.getenv("TOP_K", 5))

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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(top_k: int = TOP_K):
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=512,
        api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = get_retriever(top_k=top_k)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        .assign(answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt
            | llm
            | StrOutputParser()
        ))
    )

    print(f"RAG chain ready (model={GROQ_MODEL}, top_k={top_k})")
    return chain

def query_rag(question: str, chain=None) -> dict:
    if chain is None:
        chain = build_rag_chain()

    start_time = time.time()
    result = chain.invoke(question)
    latency = round(time.time() - start_time, 3)

    sources = [
        {
            "content_preview": doc.page_content[:300],
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?")
        }
        for doc in result.get("context", [])
    ]

    return {
        "question": question,
        "answer": result["answer"],
        "source_documents": sources,
        "latency_seconds": latency
    }

if __name__ == "__main__":
    print("=== Testing RAG Pipeline ===\n")
    chain = build_rag_chain()

    test_questions = [
        "What was the total revenue?",
        "What are the main risk factors?",
        "How did operating income change?"
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = query_rag(question, chain)
        print(f"Answer: {result['answer']}")
        print(f"Latency: {result['latency_seconds']}s")
        print(f"Sources used: {len(result['source_documents'])}")
        print("-" * 60)
