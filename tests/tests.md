# RAG Evaluation Test Set — Week 1

20 questions spanning the 10-paper NAS corpus, mixed difficulty.
Used for: Day 5-6 A/B comparison (naive vs hybrid+rerank), Week 2 RAGAS eval.

For each question: expected source paper(s) noted for ground-truth reference.
Fill in "Notes" after running each question through the pipeline.

---

## Factoid (should have a clear, retrievable answer)

1. **What is quality diversity optimization in the context of NAS?**
   Expected source: RZ_NAS.pdf (cites LLMatic)

2. **What are the three main components of the CoLLM-NAS framework?**
   Expected source: CoLLM-NAS paper (Navigator LLM, Generator LLM, Coordinator)

3. **What search space does RegNet use for its architecture?**
   Expected source: RegNet paper
   Note: known weak retrieval case from Day 3/4 testing — track improvement here specifically

4. **How much does CoLLM-NAS reduce search costs compared to existing methods?**
   Expected source: CoLLM-NAS paper (4-10x reduction)

5. **What datasets were used to evaluate RZ-NAS?**
   Expected source: RZ_NAS.pdf

6. **What is the Reflective Zero-Cost strategy in RZ-NAS?**
   Expected source: RZ_NAS.pdf

7. **How many neural architectures were generated and evaluated in the Duvvuri et al. few-shot prompting study?**
   Expected source: Duvvuri paper (1,900 architectures)

8. **What is DiffusionNAG and what role do diffusion models play in it?**
   Expected source: DiffusionNAG paper

9. **What datasets/benchmarks does CoLLM-NAS achieve state-of-the-art results on?**
   Expected source: CoLLM-NAS paper (ImageNet, NAS-Bench-201)

10. **What is NAS-PEFT and how does it relate to parameter-efficient fine-tuning?**
    Expected source: NAS_PEFT.pdf

---

## Analytical (requires synthesis or comparison across context)

11. **How does CoLLM-NAS's two-LLM approach differ from single-LLM NAS methods?**
    Expected source: CoLLM-NAS paper

12. **How does the Navigator LLM's role differ from the Generator LLM's role in CoLLM-NAS?**
    Expected source: CoLLM-NAS paper

13. **What are the main limitations of traditional (non-LLM-guided) NAS methods, according to these papers?**
    Expected source: multiple (CoLLM-NAS, RZ-NAS likely both discuss this)

14. **How does RegNet's parameter efficiency compare to ResNet?**
    Expected source: RegNet paper

15. **What role does evolutionary search play in combination with diffusion models for architecture generation?**
    Expected source: "Evolution Meets Diffusion" paper

16. **How do these papers use LLMs differently for guiding the architecture search process?**
    Expected source: multiple — cross-paper synthesis (CoLLM-NAS, RZ-NAS, Duvvuri)

---

## Hard / deliberately ambiguous (tests grounding + "I don't know" behavior)

17. **What is RegNet's design space?** (kept from Day 4 — known failure case, re-test after Week 1 improvements)
    Expected: system should either answer precisely or explicitly say context is insufficient

18. **Does LLMatic use MAP-Elites, and if so, how?**
    Expected source: likely only cited/referenced, not deeply explained in this corpus — good test of whether the model over-answers from a shallow citation

19. **What is the exact GPU/hardware setup used to train RZ-NAS's models?**
    Expected: likely NOT in the corpus — good test of "insufficient information" honesty

20. **How would CoLLM-NAS's Coordinator module apply to a completely different domain outside NAS?**
    Expected: speculative/out-of-scope — tests whether model hallucinates an answer vs. correctly flags this isn't covered

---

## Usage notes
- Run all 20 through `generate.py` for the current (naive dense retrieval) pipeline — record answers + sources used
- Re-run same 20 after Day 5-6 hybrid+rerank changes — compare directly
- Questions 17-20 are the most diagnostic for corrective RAG work in Week 2