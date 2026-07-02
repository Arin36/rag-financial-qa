import os
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from langchain_groq import ChatGroq
from src.rag_pipeline import build_rag_chain, query_rag
from deepeval.models import DeepEvalBaseLLM

load_dotenv()



class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "groq-llama-3.1-8b"
    
def evaluate_rag(test_questions: list = None):
    if test_questions is None:
        test_questions = [
            "What was the total revenue?",
            "What are the main risk factors?",
            "How did operating income change?"
        ]

    chain = build_rag_chain()
    test_cases = []

    for question in test_questions:
        result = query_rag(question, chain)
        chunks = [doc["content_preview"] for doc in result["source_documents"]]

        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            retrieval_context=chunks
        )
        test_cases.append(test_case)

    groq_judge = GroqJudge()
    metrics = [
        FaithfulnessMetric(threshold=0.7, model=groq_judge, async_mode=False),
        AnswerRelevancyMetric(threshold=0.7, model=groq_judge, async_mode=False)
    ]

    results = evaluate(test_cases, metrics)
    return results

if __name__ == "__main__":
    print("=== Running RAG Evaluation ===\n")
    results = evaluate_rag()
    print("\n=== Evaluation Complete ===")