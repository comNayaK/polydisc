# =============================================================
# api/ai_assistant.py — Groq LLaMA Chat Integration
# =============================================================

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are PolyLab AI Assistant — a helpful expert in sustainable polymer discovery.

You help users configure discovery runs for the Polymer Discovery AI platform, which uses:
- A character-level GRU VAE for molecular generation
- XGBoost + Random Forest + GNN ensemble for property prediction
- 7 normalised properties: Biodegradability, Synthesizability, Toxicity, Recyclability, Strength, Tg (glass transition), Solubility
- A weighted Green Score combining all 7 properties

You can help users by:
1. Explaining polymer properties and their significance
2. Suggesting weight configurations for specific applications (packaging, medical, construction, etc.)
3. Recommending generation parameters (temperature, N candidates)
4. Interpreting discovery results
5. Providing a JSON config block that can be auto-applied to the discovery form

When suggesting a configuration, always include a JSON block in this format:
```json
{
  "w_bio": 0.22, "w_sa": 0.13, "w_tox": 0.18,
  "w_rec": 0.13, "w_str": 0.10, "w_tg": 0.12, "w_sol": 0.12,
  "temperature": 1.0, "n_candidates": 500,
  "mode": "Generate new materials"
}
```

Be concise, helpful, and scientifically accurate. Use emoji sparingly for readability."""


def chat_with_groq(messages: list[dict]) -> dict:
    """
    Send conversation to Groq API and return the assistant reply.
    Returns dict with 'reply' (str) and optionally 'config' (dict).
    """
    if not GROQ_API_KEY:
        return {
            "reply": "⚠️ Groq API key not configured. Please set GROQ_API_KEY in your .env file.",
            "config": None
        }

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": api_messages,
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]

        # Try to extract config JSON from the reply
        config = _extract_config(reply)

        return {"reply": reply, "config": config}

    except requests.exceptions.Timeout:
        return {"reply": "⏳ Request timed out. Please try again.", "config": None}
    except requests.exceptions.RequestException as e:
        return {"reply": f"❌ API error: {str(e)}", "config": None}
    except Exception as e:
        return {"reply": f"❌ Unexpected error: {str(e)}", "config": None}


def _extract_config(text: str) -> dict | None:
    """Try to extract a JSON config block from the assistant's reply."""
    try:
        # Look for ```json ... ``` blocks
        import re
        match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
        if match:
            config = json.loads(match.group(1))
            # Validate it has expected keys
            expected = {"w_bio", "w_sa", "w_tox", "w_rec", "w_str", "w_tg", "w_sol"}
            if expected.issubset(set(config.keys())):
                return config
        return None
    except (json.JSONDecodeError, AttributeError):
        return None
