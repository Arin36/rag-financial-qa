import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_pipeline import build_rag_chain, query_rag

load_dotenv()

app =FastAPI(
    title="RAG financial Q&A",
    description="Ask questions about financial documents",
    version="1.0.0"
)
chain= build_rag_chain()

class QuestionRequest(BaseModel):
    question: str
class AnswerResponse(BaseModel):
    question: str
    answer: str
    latency_seconds: float
    sources_used: int


@app.get("/health")
def health_check():
    return {"status":"ok"}


@app.post("/ask", response_model= AnswerResponse)
def ask_question(request: QuestionRequest):
    result= query_rag(request.question, chain)
    return AnswerResponse(
        question=result["question"],
        answer=result["answer"],
        latency_seconds=result["latency_seconds"],
        sources_used=len(result["source_documents"])
    )