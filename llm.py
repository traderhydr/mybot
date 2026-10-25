"""Claude client shared by the v1 loop (bot.py) and the v2 engine (execution.py).

Every call returns the parsed JSON object the model produced; transport problems raise, they are never swallowed.
"""
import json
import os

from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")
AI_TIMEOUT = int(os.environ.get("AI_TIMEOUT", "180"))


def extract_json(text):
    """Parse the outermost {...} in a model reply (tolerates code fences / chatter around it)."""
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


_extract_json = extract_json  # backwards-compatible name used by bot.py / backtest.py


def ask_claude(prompt, system=None, max_tokens=800):
    import anthropic
    kwargs = {"system": system} if system else {}
    msg = anthropic.Anthropic(timeout=AI_TIMEOUT, max_retries=3).messages.create(
        model=CLAUDE_MODEL, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}], **kwargs)
    return extract_json("".join(b.text for b in msg.content if b.type == "text"))


def enabled_ais():
    """{name: callable(prompt, system=None, max_tokens=...)} for every configured model."""
    return {"claude": ask_claude} if os.environ.get("ANTHROPIC_API_KEY") else {}
