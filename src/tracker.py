import os
import mlflow
from dotenv import load_dotenv

load_dotenv()


EXPERIMENT_NAME= os.getenv("EXPERIMENT_NAME","rag-financial-qa")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

def log_rag_run(question: str, answer: str, latency: float, sources_used: int):
    with mlflow.start_run():
        mlflow.log_param("question", question)
        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_metric("sources_used", sources_used)
        mlflow.log_param("answer", answer)
    print(f"MLflow run logged: latency={latency}s, sources={sources_used}")

if __name__ == "__main__":
    print("=== Testing MLflow Tracker ===")
    log_rag_run(
        question="What was the total revenue?",
        answer="$215,938 million",
        latency=0.411,
        sources_used=5
    )
    print("Check mlruns/ folder — MLflow saved the run there.")