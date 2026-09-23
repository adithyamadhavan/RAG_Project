import re
from typing import List, Tuple
from openai import OpenAI
import os

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed", timeout=60.0)


# --- PII Detection (regex-based) ---

PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone_uk": r'\b(?:\+44\s?|0)(?:\d{4}\s?\d{6}|\d{3}\s?\d{3}\s?\d{4})\b',
    "phone_us": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    "ssn_us": r'\b\d{3}-\d{2}-\d{4}\b',
    "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
}


def detect_pii(text: str) -> dict:
    """
    Regex-based PII detection. Returns a dict of {pii_type: [matches]}.
    Note: regex-based detection is a scoped, lightweight approach — a
    production system would use a dedicated NER-based tool (e.g. Microsoft
    Presidio) for more accurate, context-aware detection (e.g. distinguishing
    a real name from a common word). Documented here as a deliberate
    complexity/dependency tradeoff, same rationale as the RAGAS decision.
    """
    found = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found[pii_type] = matches
    return found


def redact_pii(text: str) -> str:
    """Replaces detected PII with a redaction placeholder."""
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", text)
    return text


# --- Content moderation (LLM-based) ---

def check_content_appropriateness(text: str) -> Tuple[bool, str]:
    """
    Uses the LLM to classify whether a query/response contains adult-rated
    or otherwise inappropriate content. Returns (is_appropriate, reason).
    """
    prompt = f"""Classify the following text for content appropriateness in a professional/enterprise context.

Text: {text}

Does this text contain adult-rated, sexually explicit, violent, or otherwise
inappropriate content for a professional setting?

Answer with exactly one line in this format: VERDICT: YES or VERDICT: NO
If YES, add a brief reason on the next line."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=50,
    )
    raw = response.choices[0].message.content.strip()

    if "VERDICT: YES" in raw.upper():
        reason_lines = raw.split("\n")[1:]
        reason = " ".join(reason_lines).strip() or "flagged as inappropriate"
        return False, reason
    return True, ""


if __name__ == "__main__":
    # PII tests
    test_text = "Contact John at john.smith@example.com or 020 7946 0958. His card is 4532-1234-5678-9010."
    print("PII detected:", detect_pii(test_text))
    print("Redacted:", redact_pii(test_text))

    # Content moderation tests
    test_prompts = [
        "What is quality diversity in neural architecture search?",
        "Explain how RegNet's architecture works",
    ]
    for prompt in test_prompts:
        appropriate, reason = check_content_appropriateness(prompt)
        print(f"\nPrompt: {prompt[:50]!r}")
        print(f"Appropriate: {appropriate} {f'({reason})' if reason else ''}")