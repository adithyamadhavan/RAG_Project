# # # src/run_agent_evals.py
# # # Runs every question in evals/agent_eval_set.json through plain RAG, the
# # # supervisor and the swarm, and prints one comparison table.
# # # Run from the project root:  python src/run_agent_evals.py
# # #
# # # Checks (all deterministic, no LLM judge):
# # #   qa   : answer contains >= min_hits of the expected keywords,
# # #          and the expected source paper appears in the retrieved chunks
# # #   ooc  : out-of-corpus question; the answer should say the info is not
# # #          available (heuristic phrase match, so treat as indicative)
# # #   plus : supervisor HITL test (approve -> approved, reject -> blocked)
# # import os
# # import csv
# # import json
# # import time
# # import itertools
# # import statistics
# # from collections import defaultdict

# # from langgraph.types import Command

# # from bench_utils import CALLS, counted_chat
# # from prompts import build_prompt
# # import rag_plain
# # import supervisor_agent as sup
# # import swarm_agents as sw
# # import supervisor_llm as sl

# # EVAL_PATH = os.environ.get("EVAL_SET", "src/agent_evals.json")
# # RESULTS_PATH = "evals/agent_eval_results.csv"

# # REFUSAL_HINTS = (
# #     "not_in_corpus",
# #     "not mentioned", "no information", "not provided", "does not provide",
# #     "doesn't provide", "does not mention", "cannot", "can't", "unable",
# #     "not available", "no relevant", "not found", "no mention",
# # )

# # _ids = itertools.count()
# # _sup_app = None
# # _sw_app = None


# # def _cfg():
# #     return {"configurable": {"thread_id": f"eval-{next(_ids)}"}}


# # def run_rag(q):
# #     from retrieve import search
# #     index, chunks, sources, model = rag_plain._get_index()
# #     results = search(q, index, chunks, sources, model, 3)
# #     findings = "\n\n".join(f"[{r['source']}] {r['chunk'][:300]}" for r in results)
# #     prompt = build_prompt(q, findings)
# #     resp = counted_chat(
# #         rag_plain.client, model=rag_plain.MODEL_NAME,
# #         messages=[{"role": "user", "content": prompt}],
# #         temperature=0.1, max_tokens=300,
# #     )
# #     return resp.choices[0].message.content, findings


# # def run_supervisor(q):
# #     global _sup_app
# #     if _sup_app is None:
# #         _sup_app = sup.build_supervisor_graph()
# #     cfg = _cfg()
# #     r = _sup_app.invoke(sup._initial_state(q), config=cfg)
# #     if "__interrupt__" in r:
# #         r = _sup_app.invoke(Command(resume="approve"), config=cfg)
# #     return r["synthesis_output"], r["retriever_output"]


# # def run_swarm(q):
# #     global _sw_app
# #     if _sw_app is None:
# #         _sw_app = sw.build_swarm_graph()
# #     r = _sw_app.invoke(sw._initial_state(q), config=_cfg())
# #     return r["answer"], r["findings"]


# # _sl_app = None


# # def run_supervisor_llm(q):
# #     global _sl_app
# #     if _sl_app is None:
# #         _sl_app = sl.build_llm_supervisor_graph()
# #     cfg = _cfg()
# #     cfg["recursion_limit"] = 12
# #     r = _sl_app.invoke(sup._initial_state(q), config=cfg)
# #     if "__interrupt__" in r:
# #         r = _sl_app.invoke(Command(resume="approve"), config=cfg)
# #     return r["synthesis_output"], r["retriever_output"]


# # MODES = {"plain_rag": run_rag, "supervisor": run_supervisor,
# #          "supervisor_llm": run_supervisor_llm, "swarm": run_swarm}


# # def score(item, answer, findings):
# #     """Returns (task_ok, source_ok); None means not scored."""
# #     a = answer.lower()
# #     if item["type"] == "qa":
# #         kws = item.get("keywords") or []
# #         if not kws:
# #             return None, None
# #         hits = sum(k.lower() in a for k in kws)
# #         ok = hits >= item.get("min_hits", max(1, len(kws) // 2))
# #         src = item.get("expected_source", "")
# #         src_ok = (src.lower() in findings.lower()) if src else None
# #         return ok, src_ok
# #     if item["type"] == "ooc":
# #         return any(h in a for h in REFUSAL_HINTS), None
# #     return None, None


# # def hitl_test():
# #     """Approve must return the answer; reject must block it."""
# #     app = sup.build_supervisor_graph()
# #     q = "What does the Navigator LLM do in CoLLM-NAS?"
# #     out = {}
# #     for decision in ("approve", "reject"):
# #         cfg = _cfg()
# #         r = app.invoke(sup._initial_state(q), config=cfg)
# #         if "__interrupt__" not in r:
# #             out[decision] = False
# #             continue
# #         final = app.invoke(Command(resume=decision), config=cfg)
# #         out[decision] = (final.get("approved") is True) if decision == "approve" \
# #             else (final.get("approved") is False)
# #     return out


# # def main():
# #     items = json.load(open(EVAL_PATH))
# #     # Load the index and warm up the LLM once so the first timed item is fair.
# #     # Each module keeps its own cached index, so warm all three before timing.
# #     rag_plain._get_index()
# #     sup._get_index()
# #     sw._get_index()
# #     run_rag("warm-up")

# #     rows, skipped = [], 0
# #     for mode, fn in MODES.items():
# #         for item in items:
# #             if item["type"] == "qa" and not item.get("keywords"):
# #                 skipped += 1
# #                 continue
# #             CALLS.clear()
# #             t0 = time.perf_counter()
# #             try:
# #                 answer, findings = fn(item["question"])
# #                 err = ""
# #             except Exception as e:  # keep going; record the failure
# #                 answer, findings, err = "", "", f"{type(e).__name__}: {e}"[:120]
# #             latency = time.perf_counter() - t0
# #             ok, src_ok = score(item, answer, findings)
# #             rows.append({
# #                 "mode": mode, "id": item["id"], "type": item["type"],
# #                 "task_ok": ok, "source_ok": src_ok,
# #                 "latency_s": round(latency, 2), "llm_calls": CALLS["llm"],
# #                 "retries": CALLS["retries"], "error": err,
# #                 "answer": answer.replace("\n", " ")[:200],
# #             })
# #             print(f"[{mode}] {item['id']}: task_ok={ok} source_ok={src_ok} "
# #                   f"{latency:.1f}s {err}")

# #     os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
# #     with open(RESULTS_PATH, "w", newline="") as f:
# #         w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
# #         w.writeheader()
# #         w.writerows(rows)

# #     print("\n=== RESULTS ===")
# #     print(f"{'mode':<15}{'scored':>7}{'task%':>8}{'source%':>9}"
# #           f"{'med_s':>8}{'calls':>7}{'retried':>9}{'errors':>8}")
# #     for mode in MODES:
# #         r = [x for x in rows if x["mode"] == mode]
# #         scored = [x for x in r if x["task_ok"] is not None]
# #         src = [x for x in r if x["source_ok"] is not None]
# #         pct = lambda xs, k: (f"{100 * sum(1 for x in xs if x[k]) / len(xs):.0f}"
# #                              if xs else "n/a")
# #         print(f"{mode:<15}{len(scored):>7}{pct(scored, 'task_ok'):>8}"
# #               f"{pct(src, 'source_ok'):>9}"
# #               f"{statistics.median(x['latency_s'] for x in r):>8.2f}"
# #               f"{statistics.mean(x['llm_calls'] for x in r):>7.1f}"
# #               f"{sum(1 for x in r if x['retries']):>9}"
# #               f"{sum(1 for x in r if x['error']):>8}")
# #     sl.print_route_stats()
# #     print(f"\nSkipped (no keywords yet): {skipped // len(MODES)} questions")

# #     h = hitl_test()
# #     print(f"HITL (supervisor): approve passes answer = {h['approve']}, "
# #           f"reject blocks answer = {h['reject']}")
# #     print(f"Details written to {RESULTS_PATH}")


# # if __name__ == "__main__":
# #     main()


# # src/run_agent_evals.py
# # Runs every question in evals/agent_eval_set.json through plain RAG, the
# # supervisor and the swarm, and prints one comparison table.
# # Run from the project root:  python src/run_agent_evals.py
# #
# # Checks (all deterministic, no LLM judge):
# #   qa   : answer contains >= min_hits of the expected keywords,
# #          and the expected source paper appears in the retrieved chunks
# #   ooc  : out-of-corpus question; the answer should say the info is not
# #          available (heuristic phrase match, so treat as indicative)
# #   plus : supervisor HITL test (approve -> approved, reject -> blocked)
# import os
# import csv
# import json
# import time
# import itertools
# import statistics
# from collections import defaultdict

# from langgraph.types import Command

# from bench_utils import CALLS, counted_chat
# from prompts import build_prompt
# import rag_plain
# import supervisor_agent as sup
# import swarm_agents as sw
# import supervisor_llm as sl
# import agentic_rag as ag

# EVAL_PATH = os.environ.get("EVAL_SET", "src/agent_evals.json")
# RESULTS_PATH = "evals/agent_eval_results.csv"

# REFUSAL_HINTS = (
#     "not_in_corpus",
#     "not mentioned", "no information", "not provided", "does not provide",
#     "doesn't provide", "does not mention", "cannot", "can't", "unable",
#     "not available", "no relevant", "not found", "no mention",
# )

# _ids = itertools.count()
# _sup_app = None
# _sw_app = None


# def _cfg():
#     return {"configurable": {"thread_id": f"eval-{next(_ids)}"}}


# def run_rag(q):
#     from retrieve import search
#     index, chunks, sources, model = rag_plain._get_index()
#     results = search(q, index, chunks, sources, model, 3)
#     findings = "\n\n".join(f"[{r['source']}] {r['chunk'][:300]}" for r in results)
#     prompt = build_prompt(q, findings)
#     resp = counted_chat(
#         rag_plain.client, model=rag_plain.MODEL_NAME,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.1, max_tokens=300,
#     )
#     return resp.choices[0].message.content, findings


# def run_supervisor(q):
#     global _sup_app
#     if _sup_app is None:
#         _sup_app = sup.build_supervisor_graph()
#     cfg = _cfg()
#     r = _sup_app.invoke(sup._initial_state(q), config=cfg)
#     if "__interrupt__" in r:
#         r = _sup_app.invoke(Command(resume="approve"), config=cfg)
#     return r["synthesis_output"], r["retriever_output"]


# def run_swarm(q):
#     global _sw_app
#     if _sw_app is None:
#         _sw_app = sw.build_swarm_graph()
#     r = _sw_app.invoke(sw._initial_state(q), config=_cfg())
#     return r["answer"], r["findings"]


# _sl_app = None


# def run_supervisor_llm(q):
#     global _sl_app
#     if _sl_app is None:
#         _sl_app = sl.build_llm_supervisor_graph()
#     cfg = _cfg()
#     cfg["recursion_limit"] = 12
#     r = _sl_app.invoke(sup._initial_state(q), config=cfg)
#     if "__interrupt__" in r:
#         r = _sl_app.invoke(Command(resume="approve"), config=cfg)
#     return r["synthesis_output"], r["retriever_output"]


# _ag_app = None


# def run_agentic(q):
#     global _ag_app
#     if _ag_app is None:
#         _ag_app = ag.build_agentic_graph()
#     cfg = _cfg()
#     r = _ag_app.invoke(ag._initial_state(q), config=cfg)
#     if "__interrupt__" in r:
#         r = _ag_app.invoke(Command(resume="approve"), config=cfg)
#     return r["answer"], r["findings"]


# MODES = {"plain_rag": run_rag, "supervisor": run_supervisor,
#          "supervisor_llm": run_supervisor_llm, "swarm": run_swarm,
#          "agentic_rag": run_agentic}


# def score(item, answer, findings):
#     """Returns (task_ok, source_ok); None means not scored."""
#     a = answer.lower()
#     if item["type"] == "qa":
#         kws = item.get("keywords") or []
#         if not kws:
#             return None, None
#         hits = sum(k.lower() in a for k in kws)
#         ok = hits >= item.get("min_hits", max(1, len(kws) // 2))
#         src = item.get("expected_source", "")
#         src_ok = (src.lower() in findings.lower()) if src else None
#         return ok, src_ok
#     if item["type"] == "ooc":
#         return any(h in a for h in REFUSAL_HINTS), None
#     return None, None


# def hitl_test():
#     """Approve must return the answer; reject must block it."""
#     app = sup.build_supervisor_graph()
#     q = "What does the Navigator LLM do in CoLLM-NAS?"
#     out = {}
#     for decision in ("approve", "reject"):
#         cfg = _cfg()
#         r = app.invoke(sup._initial_state(q), config=cfg)
#         if "__interrupt__" not in r:
#             out[decision] = False
#             continue
#         final = app.invoke(Command(resume=decision), config=cfg)
#         out[decision] = (final.get("approved") is True) if decision == "approve" \
#             else (final.get("approved") is False)
#     return out


# def main():
#     items = json.load(open(EVAL_PATH))
#     # Load the index and warm up the LLM once so the first timed item is fair.
#     # Each module keeps its own cached index, so warm all three before timing.
#     rag_plain._get_index()
#     sup._get_index()
#     sw._get_index()
#     ag._get_index()
#     run_rag("warm-up")

#     rows, skipped = [], 0
#     for mode, fn in MODES.items():
#         for item in items:
#             if item["type"] == "qa" and not item.get("keywords"):
#                 skipped += 1
#                 continue
#             CALLS.clear()
#             t0 = time.perf_counter()
#             try:
#                 answer, findings = fn(item["question"])
#                 err = ""
#             except Exception as e:  # keep going; record the failure
#                 answer, findings, err = "", "", f"{type(e).__name__}: {e}"[:120]
#             latency = time.perf_counter() - t0
#             ok, src_ok = score(item, answer, findings)
#             rows.append({
#                 "mode": mode, "id": item["id"], "type": item["type"],
#                 "task_ok": ok, "source_ok": src_ok,
#                 "latency_s": round(latency, 2), "llm_calls": CALLS["llm"],
#                 "retries": CALLS["retries"], "error": err,
#                 "answer": answer.replace("\n", " ")[:200],
#             })
#             print(f"[{mode}] {item['id']}: task_ok={ok} source_ok={src_ok} "
#                   f"{latency:.1f}s {err}")

#     os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
#     with open(RESULTS_PATH, "w", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#         w.writeheader()
#         w.writerows(rows)

#     print("\n=== RESULTS ===")
#     print(f"{'mode':<15}{'scored':>7}{'task%':>8}{'source%':>9}"
#           f"{'med_s':>8}{'calls':>7}{'retried':>9}{'errors':>8}")
#     for mode in MODES:
#         r = [x for x in rows if x["mode"] == mode]
#         scored = [x for x in r if x["task_ok"] is not None]
#         src = [x for x in r if x["source_ok"] is not None]
#         pct = lambda xs, k: (f"{100 * sum(1 for x in xs if x[k]) / len(xs):.0f}"
#                              if xs else "n/a")
#         print(f"{mode:<15}{len(scored):>7}{pct(scored, 'task_ok'):>8}"
#               f"{pct(src, 'source_ok'):>9}"
#               f"{statistics.median(x['latency_s'] for x in r):>8.2f}"
#               f"{statistics.mean(x['llm_calls'] for x in r):>7.1f}"
#               f"{sum(1 for x in r if x['retries']):>9}"
#               f"{sum(1 for x in r if x['error']):>8}")
#     sl.print_route_stats()
#     print(f"\nSkipped (no keywords yet): {skipped // len(MODES)} questions")

#     h = hitl_test()
#     print(f"HITL (supervisor): approve passes answer = {h['approve']}, "
#           f"reject blocks answer = {h['reject']}")
#     print(f"Details written to {RESULTS_PATH}")


# if __name__ == "__main__":
#     main()


# src/run_agent_evals.py
# Runs every question in evals/agent_eval_set.json through plain RAG, the
# supervisor and the swarm, and prints one comparison table.
# Run from the project root:  python src/run_agent_evals.py
#
# Checks (all deterministic, no LLM judge):
#   qa   : answer contains >= min_hits of the expected keywords,
#          and the expected source paper appears in the retrieved chunks
#   ooc  : out-of-corpus question; the answer should say the info is not
#          available (heuristic phrase match, so treat as indicative)
#   plus : supervisor HITL test (approve -> approved, reject -> blocked)
# import os
# import csv
# import json
# import time
# import itertools
# import statistics
# from collections import defaultdict

# from langgraph.types import Command

# from bench_utils import CALLS, counted_chat
# from prompts import build_prompt
# import rag_plain
# import supervisor_agent as sup
# import swarm_agents as sw
# import supervisor_llm as sl
# import agentic_rag as ag

# os.environ["EVAL_RUN"] = "1"   # auto-approved answers must not enter long-term memory
# EVAL_PATH = os.environ.get("EVAL_SET", "evals/agent_eval_set.json")
# RESULTS_PATH = "evals/agent_eval_results.csv"

# REFUSAL_HINTS = (
#     "not_in_corpus",
#     "not mentioned", "no information", "not provided", "does not provide",
#     "doesn't provide", "does not mention", "cannot", "can't", "unable",
#     "not available", "no relevant", "not found", "no mention",
# )

# _ids = itertools.count()
# _sup_app = None
# _sw_app = None


# def _cfg():
#     return {"configurable": {"thread_id": f"eval-{next(_ids)}"}}


# def run_rag(q):
#     from retrieve import search
#     index, chunks, sources, model = rag_plain._get_index()
#     results = search(q, index, chunks, sources, model, 3)
#     findings = "\n\n".join(f"[{r['source']}] {r['chunk'][:300]}" for r in results)
#     prompt = build_prompt(q, findings)
#     resp = counted_chat(
#         rag_plain.client, model=rag_plain.MODEL_NAME,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.1, max_tokens=300,
#     )
#     return resp.choices[0].message.content, findings


# def run_supervisor(q):
#     global _sup_app
#     if _sup_app is None:
#         _sup_app = sup.build_supervisor_graph()
#     cfg = _cfg()
#     r = _sup_app.invoke(sup._initial_state(q), config=cfg)
#     if "__interrupt__" in r:
#         r = _sup_app.invoke(Command(resume="approve"), config=cfg)
#     return r["synthesis_output"], r["retriever_output"]


# def run_swarm(q):
#     global _sw_app
#     if _sw_app is None:
#         _sw_app = sw.build_swarm_graph()
#     r = _sw_app.invoke(sw._initial_state(q), config=_cfg())
#     return r["answer"], r["findings"]


# _sl_app = None


# def run_supervisor_llm(q):
#     global _sl_app
#     if _sl_app is None:
#         _sl_app = sl.build_llm_supervisor_graph()
#     cfg = _cfg()
#     cfg["recursion_limit"] = 12
#     r = _sl_app.invoke(sup._initial_state(q), config=cfg)
#     if "__interrupt__" in r:
#         r = _sl_app.invoke(Command(resume="approve"), config=cfg)
#     return r["synthesis_output"], r["retriever_output"]


# _ag_app = None


# def run_agentic(q):
#     global _ag_app
#     if _ag_app is None:
#         _ag_app = ag.build_agentic_graph()
#     cfg = _cfg()
#     r = _ag_app.invoke(ag._initial_state(q), config=cfg)
#     if "__interrupt__" in r:
#         r = _ag_app.invoke(Command(resume="approve"), config=cfg)
#     return r["answer"], r["findings"]


# MODES = {"plain_rag": run_rag, "supervisor": run_supervisor,
#          "supervisor_llm": run_supervisor_llm, "swarm": run_swarm,
#          "agentic_rag": run_agentic}


# def score(item, answer, findings):
#     """Returns (task_ok, source_ok); None means not scored."""
#     a = answer.lower()
#     if item["type"] == "qa":
#         kws = item.get("keywords") or []
#         if not kws:
#             return None, None
#         hits = sum(k.lower() in a for k in kws)
#         ok = hits >= item.get("min_hits", max(1, len(kws) // 2))
#         src = item.get("expected_source", "")
#         src_ok = (src.lower() in findings.lower()) if src else None
#         return ok, src_ok
#     if item["type"] == "ooc":
#         return any(h in a for h in REFUSAL_HINTS), None
#     return None, None


# def hitl_test():
#     """Approve must return the answer; reject must block it."""
#     app = sup.build_supervisor_graph()
#     q = "What does the Navigator LLM do in CoLLM-NAS?"
#     out = {}
#     for decision in ("approve", "reject"):
#         cfg = _cfg()
#         r = app.invoke(sup._initial_state(q), config=cfg)
#         if "__interrupt__" not in r:
#             out[decision] = False
#             continue
#         final = app.invoke(Command(resume=decision), config=cfg)
#         out[decision] = (final.get("approved") is True) if decision == "approve" \
#             else (final.get("approved") is False)
#     return out


# def main():
#     items = json.load(open(EVAL_PATH))
#     # Load the index and warm up the LLM once so the first timed item is fair.
#     # Each module keeps its own cached index, so warm all three before timing.
#     rag_plain._get_index()
#     sup._get_index()
#     sw._get_index()
#     ag._get_index()
#     run_rag("warm-up")

#     rows, skipped = [], 0
#     for mode, fn in MODES.items():
#         for item in items:
#             if item["type"] == "qa" and not item.get("keywords"):
#                 skipped += 1
#                 continue
#             CALLS.clear()
#             t0 = time.perf_counter()
#             try:
#                 answer, findings = fn(item["question"])
#                 err = ""
#             except Exception as e:  # keep going; record the failure
#                 answer, findings, err = "", "", f"{type(e).__name__}: {e}"[:120]
#             latency = time.perf_counter() - t0
#             ok, src_ok = score(item, answer, findings)
#             rows.append({
#                 "mode": mode, "id": item["id"], "type": item["type"],
#                 "task_ok": ok, "source_ok": src_ok,
#                 "latency_s": round(latency, 2), "llm_calls": CALLS["llm"],
#                 "retries": CALLS["retries"], "error": err,
#                 "answer": answer.replace("\n", " ")[:200],
#             })
#             print(f"[{mode}] {item['id']}: task_ok={ok} source_ok={src_ok} "
#                   f"{latency:.1f}s {err}")

#     os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
#     with open(RESULTS_PATH, "w", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#         w.writeheader()
#         w.writerows(rows)

#     print("\n=== RESULTS ===")
#     print(f"{'mode':<15}{'scored':>7}{'task%':>8}{'source%':>9}"
#           f"{'med_s':>8}{'calls':>7}{'retried':>9}{'errors':>8}")
#     for mode in MODES:
#         r = [x for x in rows if x["mode"] == mode]
#         scored = [x for x in r if x["task_ok"] is not None]
#         src = [x for x in r if x["source_ok"] is not None]
#         pct = lambda xs, k: (f"{100 * sum(1 for x in xs if x[k]) / len(xs):.0f}"
#                              if xs else "n/a")
#         print(f"{mode:<15}{len(scored):>7}{pct(scored, 'task_ok'):>8}"
#               f"{pct(src, 'source_ok'):>9}"
#               f"{statistics.median(x['latency_s'] for x in r):>8.2f}"
#               f"{statistics.mean(x['llm_calls'] for x in r):>7.1f}"
#               f"{sum(1 for x in r if x['retries']):>9}"
#               f"{sum(1 for x in r if x['error']):>8}")
#     sl.print_route_stats()
#     print(f"\nSkipped (no keywords yet): {skipped // len(MODES)} questions")

#     h = hitl_test()
#     print(f"HITL (supervisor): approve passes answer = {h['approve']}, "
#           f"reject blocks answer = {h['reject']}")
#     print(f"Details written to {RESULTS_PATH}")


# if __name__ == "__main__":
#     main()

# src/run_agent_evals.py
# Runs every question in evals/agent_eval_set.json through plain RAG, the
# supervisor and the swarm, and prints one comparison table.
# Run from the project root:  python src/run_agent_evals.py
#
# Checks (all deterministic, no LLM judge):
#   qa   : answer contains >= min_hits of the expected keywords,
#          and the expected source paper appears in the retrieved chunks
#   ooc  : out-of-corpus question; the answer should say the info is not
#          available (heuristic phrase match, so treat as indicative)
#   plus : supervisor HITL test (approve -> approved, reject -> blocked)
# src/run_agent_evals.py
# Runs every question in evals/agent_eval_set.json through plain RAG, the
# supervisor and the swarm, and prints one comparison table.
# Run from the project root:  python src/run_agent_evals.py
#
# Checks (all deterministic, no LLM judge):
#   qa   : answer contains >= min_hits of the expected keywords,
#          and the expected source paper appears in the retrieved chunks
#   ooc  : out-of-corpus question; the answer should say the info is not
#          available (heuristic phrase match, so treat as indicative)
#   plus : supervisor HITL test (approve -> approved, reject -> blocked)
import os
import csv
import json
import time
import itertools
import statistics
from collections import defaultdict

from langgraph.types import Command

from bench_utils import CALLS, counted_chat
from prompts import build_prompt
import rag_plain
import supervisor_agent as sup
import swarm_agents as sw
import supervisor_llm as sl
import agentic_rag as ag
import workflows as wf

os.environ["EVAL_RUN"] = "1"   # auto-approved answers must not enter long-term memory
EVAL_PATH = os.environ.get("EVAL_SET", "src/agent_evals.json")
RESULTS_PATH = "evals/agent_eval_results.csv"

REFUSAL_HINTS = (
    "not_in_corpus",
    "not mentioned", "no information", "not provided", "does not provide",
    "doesn't provide", "does not mention", "cannot", "can't", "unable",
    "not available", "no relevant", "not found", "no mention",
)

_ids = itertools.count()
_sup_app = None
_sw_app = None


def _cfg():
    return {"configurable": {"thread_id": f"eval-{next(_ids)}"}}


def run_rag(q):
    from retrieve import search
    index, chunks, sources, model = rag_plain._get_index()
    results = search(q, index, chunks, sources, model, 3)
    findings = "\n\n".join(f"[{r['source']}] {r['chunk'][:300]}" for r in results)
    prompt = build_prompt(q, findings)
    resp = counted_chat(
        rag_plain.client, model=rag_plain.MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=300,
    )
    return resp.choices[0].message.content, findings


def run_supervisor(q):
    global _sup_app
    if _sup_app is None:
        _sup_app = sup.build_supervisor_graph()
    cfg = _cfg()
    r = _sup_app.invoke(sup._initial_state(q), config=cfg)
    if "__interrupt__" in r:
        r = _sup_app.invoke(Command(resume="approve"), config=cfg)
    return r["synthesis_output"], r["retriever_output"]


def run_swarm(q):
    global _sw_app
    if _sw_app is None:
        _sw_app = sw.build_swarm_graph()
    r = _sw_app.invoke(sw._initial_state(q), config=_cfg())
    return r["answer"], r["findings"]


_sl_app = None


def run_supervisor_llm(q):
    global _sl_app
    if _sl_app is None:
        _sl_app = sl.build_llm_supervisor_graph()
    cfg = _cfg()
    cfg["recursion_limit"] = 12
    r = _sl_app.invoke(sup._initial_state(q), config=cfg)
    if "__interrupt__" in r:
        r = _sl_app.invoke(Command(resume="approve"), config=cfg)
    return r["synthesis_output"], r["retriever_output"]


_ag_app = None


def run_agentic(q):
    global _ag_app
    if _ag_app is None:
        _ag_app = ag.build_agentic_graph()
    cfg = _cfg()
    r = _ag_app.invoke(ag._initial_state(q), config=cfg)
    if "__interrupt__" in r:
        r = _ag_app.invoke(Command(resume="approve"), config=cfg)
    return r["answer"], r["findings"]


_wf_app = None


def run_workflows(q):
    global _wf_app
    if _wf_app is None:
        _wf_app = wf.build_workflow_graph()
    r = _wf_app.invoke(wf.initial(q))
    return r["answer"], r["findings"]


MODES = {"plain_rag": run_rag, "supervisor": run_supervisor,
         "supervisor_llm": run_supervisor_llm, "swarm": run_swarm,
         "agentic_rag": run_agentic, "workflows": run_workflows}


if os.environ.get("ONLY"):          # e.g. ONLY=workflows or ONLY=agentic_rag,workflows
    MODES = {k: v for k, v in MODES.items() if k in os.environ["ONLY"].split(",")}


def score(item, answer, findings):
    """Returns (task_ok, source_ok); None means not scored."""
    a = answer.lower()
    if item["type"] == "qa":
        kws = item.get("keywords") or []
        if not kws:
            return None, None
        hits = sum(k.lower() in a for k in kws)
        ok = hits >= item.get("min_hits", max(1, len(kws) // 2))
        src = item.get("expected_source", "")
        src_ok = (src.lower() in findings.lower()) if src else None
        return ok, src_ok
    if item["type"] == "ooc":
        return any(h in a for h in REFUSAL_HINTS), None
    return None, None


def hitl_test():
    """Approve must return the answer; reject must block it."""
    app = sup.build_supervisor_graph()
    q = "What does the Navigator LLM do in CoLLM-NAS?"
    out = {}
    for decision in ("approve", "reject"):
        cfg = _cfg()
        r = app.invoke(sup._initial_state(q), config=cfg)
        if "__interrupt__" not in r:
            out[decision] = False
            continue
        final = app.invoke(Command(resume=decision), config=cfg)
        out[decision] = (final.get("approved") is True) if decision == "approve" \
            else (final.get("approved") is False)
    return out


def main():
    items = json.load(open(EVAL_PATH))
    # Load the index and warm up the LLM once so the first timed item is fair.
    # Each module keeps its own cached index, so warm all three before timing.
    rag_plain._get_index()
    sup._get_index()
    sw._get_index()
    ag._get_index()
    run_rag("warm-up")

    rows, skipped = [], 0
    for mode, fn in MODES.items():
        for item in items:
            if item["type"] == "qa" and not item.get("keywords"):
                skipped += 1
                continue
            CALLS.clear()
            t0 = time.perf_counter()
            try:
                answer, findings = fn(item["question"])
                err = ""
            except Exception as e:  # keep going; record the failure
                answer, findings, err = "", "", f"{type(e).__name__}: {e}"[:120]
            latency = time.perf_counter() - t0
            ok, src_ok = score(item, answer, findings)
            rows.append({
                "mode": mode, "id": item["id"], "type": item["type"],
                "task_ok": ok, "source_ok": src_ok,
                "latency_s": round(latency, 2), "llm_calls": CALLS["llm"],
                "retries": CALLS["retries"], "error": err,
                "answer": answer.replace("\n", " ")[:200],
            })
            print(f"[{mode}] {item['id']}: task_ok={ok} source_ok={src_ok} "
                  f"{latency:.1f}s {err}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== RESULTS ===")
    print(f"{'mode':<15}{'scored':>7}{'task%':>8}{'source%':>9}"
          f"{'med_s':>8}{'calls':>7}{'retried':>9}{'errors':>8}")
    for mode in MODES:
        r = [x for x in rows if x["mode"] == mode]
        scored = [x for x in r if x["task_ok"] is not None]
        src = [x for x in r if x["source_ok"] is not None]
        pct = lambda xs, k: (f"{100 * sum(1 for x in xs if x[k]) / len(xs):.0f}"
                             if xs else "n/a")
        print(f"{mode:<15}{len(scored):>7}{pct(scored, 'task_ok'):>8}"
              f"{pct(src, 'source_ok'):>9}"
              f"{statistics.median(x['latency_s'] for x in r):>8.2f}"
              f"{statistics.mean(x['llm_calls'] for x in r):>7.1f}"
              f"{sum(1 for x in r if x['retries']):>9}"
              f"{sum(1 for x in r if x['error']):>8}")
    sl.print_route_stats()
    print(f"\nSkipped (no keywords yet): {skipped // len(MODES)} questions")

    h = hitl_test()
    print(f"HITL (supervisor): approve passes answer = {h['approve']}, "
          f"reject blocks answer = {h['reject']}")
    print(f"Details written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()