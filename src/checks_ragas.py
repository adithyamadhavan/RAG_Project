# quick check — src/check_ragas_fields.py
import json

with open("eval_results_naive.json") as f:
    data = json.load(f)

print(data[0].keys())
print(data[0])