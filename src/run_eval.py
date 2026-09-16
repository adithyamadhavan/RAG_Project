from generate import generate_answer
from retrieve import load_index_data
import json
from pathlib import Path

TEST_QUESTIONS = [
    "What is quality diversity optimization in the context of NAS?",
    "What are the three main components of the CoLLM-NAS framework?",
    "What search space does RegNet use for its architecture?",
    "How much does CoLLM-NAS reduce search costs compared to existing methods?",
    "What datasets were used to evaluate RZ-NAS?",
    "What is the Reflective Zero-Cost strategy in RZ-NAS?",
    "How many neural architectures were generated and evaluated in the Duvvuri et al. few-shot prompting study?",
    "What is DiffusionNAG and what role do diffusion models play in it?",
    "What datasets/benchmarks does CoLLM-NAS achieve state-of-the-art results on?",
    "What is NAS-PEFT and how does it relate to parameter-efficient fine-tuning?",
    "How does CoLLM-NAS's two-LLM approach differ from single-LLM NAS methods?",
    "How does the Navigator LLM's role differ from the Generator LLM's role in CoLLM-NAS?",
    "What are the main limitations of traditional (non-LLM-guided) NAS methods, according to these papers?",
    "How does RegNet's parameter efficiency compare to ResNet?",
    "What role does evolutionary search play in combination with diffusion models for architecture generation?",
    "How do these papers use LLMs differently for guiding the architecture search process?",
    "What is RegNet's design space?",
    "Does LLMatic use MAP-Elites, and if so, how?",
    "What is the exact GPU/hardware setup used to train RZ-NAS's models?",
    "How would CoLLM-NAS's Coordinator module apply to a completely different domain outside NAS?",
]


# def run_full_eval(output_path: str = "eval_results_naive.json"):
#     results = []
#     for i, question in enumerate(TEST_QUESTIONS, 1):
#         print(f"\n[{i}/{len(TEST_QUESTIONS)}] {question}")
#         result = generate_answer(question)
#         results.append(result)
#         print(f"→ {result['answer'][:150]}...")
#         print(f"  Sources: {result['retrieved_sources']}")

#     with open(output_path, "w") as f:
#         json.dump(results, f, indent=2)

#     print(f"\n\nSaved {len(results)} results to {output_path}")
#     return results

from retrieve import load_index_data
from generate import generate_with_retry
import json

def run_full_eval(output_path="eval_results_naive.json", start_from=0):
    index, chunks, sources, model = load_index_data()  # loaded once, not per-question
    results = []
    for i, question in enumerate(TEST_QUESTIONS[start_from:], start_from + 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {question}")
        result = generate_with_retry(question, index, chunks, sources, model)
        results.append(result)
        print(f"→ {result['answer'][:150]}...")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nSaved {len(results)} results to {output_path}")
    return results

if __name__ == "__main__":
    run_full_eval()