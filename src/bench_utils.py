# # src/bench_utils.py
# # Shared by supervisor_agent.py, swarm_agents.py and app.py so every system
# # is counted and timed in exactly the same way.
# import time
# import statistics
# from collections import Counter

# import openai

# CALLS = Counter()


# def counted_chat(client, retries=4, **kwargs):
#     """Route every LLM call through here. Counts one logical call, and retries
#     if the local vllm-mlx server answers 503 'text_generation_busy' (it serves
#     one generation at a time). Retries are tallied in CALLS['retries'] so a
#     benchmark run that had to wait can be spotted."""
#     CALLS["llm"] += 1
#     for attempt in range(retries + 1):
#         try:
#             return client.chat.completions.create(**kwargs)
#         except openai.InternalServerError as e:
#             if "text_generation_busy" not in str(e) or attempt == retries:
#                 raise
#             CALLS["retries"] += 1
#             time.sleep(3 * (attempt + 1))


# def benchmark(run_fn, n_timed=3):
#     """One warm-up run (discarded), then n_timed timed runs.
#     run_fn() must perform one full run and must not wait for human input.
#     Runs with retries are flagged: their latency includes waiting for the
#     server, so rerun them with nothing else hitting vllm-mlx."""
#     CALLS.clear()
#     run_fn()  # warm-up
#     results = []
#     for _ in range(n_timed):
#         CALLS.clear()
#         t0 = time.perf_counter()
#         run_fn()
#         results.append((time.perf_counter() - t0, CALLS["llm"], CALLS["retries"]))
#     for i, (secs, calls, retried) in enumerate(results, 1):
#         flag = f"  <-- {retried} retries, rerun this" if retried else ""
#         print(f"run {i}: {secs:.2f}s | LLM calls: {calls}{flag}")
#     print(f"MEDIAN: {statistics.median(r[0] for r in results):.2f}s | "
#           f"LLM calls: {results[0][1]}")
#     return results



# src/bench_utils.py
# Shared by supervisor_agent.py, swarm_agents.py and app.py so every system
# is counted and timed in exactly the same way.
import time
import statistics
from collections import Counter

import openai

CALLS = Counter()


def counted_chat(client, retries=4, **kwargs):
    """Route every LLM call through here. Counts one logical call, and retries
    if the local vllm-mlx server answers 503 'text_generation_busy' (it serves
    one generation at a time). Retries are tallied in CALLS['retries'] so a
    benchmark run that had to wait can be spotted."""
    CALLS["llm"] += 1
    for attempt in range(retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except openai.InternalServerError as e:
            if "text_generation_busy" not in str(e) or attempt == retries:
                raise
            CALLS["retries"] += 1
            time.sleep(3 * (attempt + 1))


def benchmark(run_fn, n_timed=3):
    """One warm-up run (discarded), then n_timed timed runs.
    run_fn() must perform one full run and must not wait for human input.
    Runs with retries are flagged: their latency includes waiting for the
    server, so rerun them with nothing else hitting vllm-mlx."""
    CALLS.clear()
    run_fn()  # warm-up
    results = []
    for _ in range(n_timed):
        CALLS.clear()
        t0 = time.perf_counter()
        run_fn()
        results.append((time.perf_counter() - t0, CALLS["llm"], CALLS["retries"]))
    for i, (secs, calls, retried) in enumerate(results, 1):
        flag = f"  <-- {retried} retries, rerun this" if retried else ""
        print(f"run {i}: {secs:.2f}s | LLM calls: {calls}{flag}")
    print(f"MEDIAN: {statistics.median(r[0] for r in results):.2f}s | "
          f"LLM calls: {results[0][1]}")
    return results