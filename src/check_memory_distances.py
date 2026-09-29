from agent_memory import LongTermMemory

mem = LongTermMemory()

mem.add(
    "User asked about the Navigator LLM in CoLLM-NAS. It provides search "
    "strategy through comprehension of the search space and exploration strategy.",
    metadata={"topic": "collm_nas"},
)
mem.add(
    "User asked what 15 * 23 + 7 equals. The calculator tool returned 352.",
    metadata={"topic": "calculation"},
)
mem.add(
    "User asked about RZ-NAS's Reflective Zero-Cost strategy, which combines "
    "LLM reflection with zero-cost proxy evaluation.",
    metadata={"topic": "rz_nas"},
)

print(f"total memories: {len(mem.memories)}\n")

print("--- relevant query (CoLLM-NAS) ---")
for r in mem.retrieve_with_scores("what does the navigator LLM do in CoLLM-NAS", k=3):
    print(f"{r['distance']:.3f}  {r['memory']['text'][:80]}")

print("\n--- unrelated query (arithmetic) ---")
for r in mem.retrieve_with_scores("what is 15 times 23", k=3):
    print(f"{r['distance']:.3f}  {r['memory']['text'][:80]}")
