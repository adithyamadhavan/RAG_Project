# src/prompts.py
# One shared prompt so plain RAG, the supervisor, the swarm and the app are
# compared on identical instructions.
NOT_FOUND = "NOT_IN_CORPUS"


def build_prompt(question, findings):
    return f"""Question: {question}

Retrieved information:
{findings}

Answer using ONLY the retrieved information above. Do not use outside knowledge.
If the retrieved information does not contain the answer, reply with exactly: {NOT_FOUND}
Otherwise write a concise, grounded answer."""