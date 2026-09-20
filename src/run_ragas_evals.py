from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
import json

from golden_qa import GOLDEN_QA

# LLM for judging (faithfulness, answer_relevancy reasoning)
ragas_llm = LangchainLLMWrapper(ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",
    model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
))

# Embeddings for similarity-based metrics (answer_relevancy, context_precision/recall)
ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
)


def build_ragas_dataset(eval_results_path: str) -> Dataset:
    with open(eval_results_path) as f:
        results = json.load(f)

    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for gqa in GOLDEN_QA:
        match = next((r for r in results if r["query"] == gqa["question"]), None)
        if match is None:
            continue
        data["question"].append(gqa["question"])
        data["answer"].append(match["answer"])
        data["contexts"].append(match["retrieved_chunks"])
        data["ground_truth"].append(gqa["ground_truth"])

    return Dataset.from_dict(data)


def run_ragas(eval_results_path: str, label: str):
    print(f"\n{'='*60}\nRAGAS eval: {label}\n{'='*60}")
    ds = build_ragas_dataset(eval_results_path)
    print(f"Matched {len(ds)} / {len(GOLDEN_QA)} golden questions")

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    print(result)
    return result


if __name__ == "__main__":
    naive_result = run_ragas("eval_results_naive.json", "Naive (dense-only)")
    hybrid_result = run_ragas("eval_results_hybrid.json", "Hybrid + Rerank")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Naive:  {naive_result}")
    print(f"Hybrid: {hybrid_result}")