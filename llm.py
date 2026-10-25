"""LLM clients (Gemini, Claude) with pacing, retries and a rate-limit circuit breaker.

Shared by the v1 loop (bot.py) and the v2 engine (execution.py). Every call returns the parsed JSON object the
model produced; transport problems raise, they are never swallowed.
"""
import json
import os
import threading
import time

import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")
AI_TIMEOUT = int(os.environ.get("AI_TIMEOUT", "180"))
GEMINI_RPM = float(os.environ.get("GEMINI_RPM", "8"))  # max Gemini requests per minute
GEMINI_COOLDOWN = int(os.environ.get("GEMINI_COOLDOWN_MINUTES", "30")) * 60  # pause after repeated 429s


def extract_json(text):
    """Parse the outermost {...} in a model reply (tolerates code fences / chatter around it)."""
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


_extract_json = extract_json  # backwards-compatible name used by bot.py / backtest.py

_gemini_lock = threading.Lock()
_gemini_last = [0.0]
_gemini_blocked_until = [0.0]


def _gemini_pace():
    """Keep Gemini calls at most GEMINI_RPM per minute (free tiers allow only a few)."""
    with _gemini_lock:
        gap = 60.0 / GEMINI_RPM - (time.time() - _gemini_last[0])
        if gap > 0:
            time.sleep(gap)
        _gemini_last[0] = time.time()


def gemini_blocked():
    return time.time() < _gemini_blocked_until[0]


def ask_gemini(prompt, system=None, max_tokens=2048):
    if gemini_blocked():
        raise RuntimeError("Gemini cooling down after rate limits")
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": max_tokens}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    last = None
    for wait in (0, 20, 45, 90):  # retry overloads (503), rate limits (429) and timeouts
        time.sleep(wait)
        _gemini_pace()
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, json=body, timeout=AI_TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"HTTP {r.status_code}")
                if r.status_code == 429 and r.headers.get("Retry-After", "").isdigit():
                    time.sleep(min(int(r.headers["Retry-After"]), 120))
                continue
            r.raise_for_status()
            return extract_json("".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"]))
        except requests.exceptions.Timeout as ex:
            last = ex
    if "429" in str(last):  # quota is gone: stop hammering it for a while
        _gemini_blocked_until[0] = time.time() + GEMINI_COOLDOWN
        print(f"Gemini rate-limited, pausing it for {GEMINI_COOLDOWN // 60} min")
    raise last


def ask_claude(prompt, system=None, max_tokens=800):
    import anthropic
    kwargs = {"system": system} if system else {}
    msg = anthropic.Anthropic(timeout=AI_TIMEOUT, max_retries=3).messages.create(
        model=CLAUDE_MODEL, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}], **kwargs)
    return extract_json("".join(b.text for b in msg.content if b.type == "text"))


def enabled_ais():
    ais = {}
    if os.environ.get("GEMINI_API_KEY"):
        ais["gemini"] = ask_gemini
    if os.environ.get("ANTHROPIC_API_KEY"):
        ais["claude"] = ask_claude
    return ais
