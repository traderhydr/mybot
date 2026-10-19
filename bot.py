"""AI trade-signal bot: Claude or DeepSeek analyses Binance futures data and posts
signals (4 entries, 1 TP, SL, cross 10x) to a Telegram channel."""
import json
import os
import random
import subprocess
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHANNEL = os.environ.get("TELEGRAM_CHANNEL_ID", "")
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
SYMBOLS = [s.strip().upper() for s in os.environ.get("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",") if s.strip()]
TIMEFRAME = os.environ.get("TIMEFRAME", "1h")
INTERVAL_MIN = int(os.environ.get("INTERVAL_MINUTES", "240"))

LEVERAGE = 10
MARGIN_MODE = "Cross"
BINANCE = "https://fapi.binance.com"


# ---------- market data ----------

def fetch_candles(symbol, limit=200):
    r = requests.get(f"{BINANCE}/fapi/v1/klines",
                     params={"symbol": symbol, "interval": TIMEFRAME, "limit": limit}, timeout=15)
    r.raise_for_status()
    return [{"o": float(k[1]), "h": float(k[2]), "l": float(k[3]), "c": float(k[4]), "v": float(k[5])}
            for k in r.json()]


def ema(values, period):
    k, out = 2 / (period + 1), values[0]
    for v in values[1:]:
        out = v * k + out * (1 - k)
    return out


def rsi(closes, period=14):
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0)
        losses += max(-d, 0)
    ag, al = gains / period, losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (period - 1) + max(d, 0)) / period
        al = (al * (period - 1) + max(-d, 0)) / period
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def atr(c, period=14):
    trs = [max(c[i]["h"] - c[i]["l"], abs(c[i]["h"] - c[i - 1]["c"]), abs(c[i]["l"] - c[i - 1]["c"]))
           for i in range(1, len(c))]
    return sum(trs[-period:]) / period


def snapshot(symbol):
    c = fetch_candles(symbol)
    closes = [x["c"] for x in c]
    return {
        "symbol": symbol,
        "timeframe": TIMEFRAME,
        "price": closes[-1],
        "ema20": round(ema(closes, 20), 6),
        "ema50": round(ema(closes, 50), 6),
        "ema200": round(ema(closes, 200), 6),
        "rsi14": round(rsi(closes), 2),
        "atr14": round(atr(c), 6),
        "high_50": max(x["h"] for x in c[-50:]),
        "low_50": min(x["l"] for x in c[-50:]),
        "last_10_closes": closes[-10:],
    }


# ---------- AI ----------

PROMPT = """You are a crypto futures analyst. Given the market snapshots below, pick the ONE
best setup (long or short) and return it as strict JSON, nothing else:

{{"symbol": "...", "side": "LONG"|"SHORT", "entries": [e1, e2, e3, e4], "tp": number, "sl": number, "reason": "<=200 chars"}}

Rules:
- Exactly 4 entry prices (a scaled/laddered entry): e1 closest to current price, each next one further away.
- For LONG: entries descend, below or at current price; tp above all entries; sl below all entries.
- For SHORT: entries ascend, above or at current price; tp below all entries; sl above all entries.
- Exactly ONE take-profit. Leverage is fixed at {lev}x cross, so keep the SL tight enough that the
  loss at the last entry stays modest (use ATR; roughly 0.5-1.5 ATR beyond the last entry).
- Use realistic prices for the symbol's price scale.

Snapshots:
{data}"""


def ask_ai(snaps):
    prompt = PROMPT.format(lev=LEVERAGE, data=json.dumps(snaps, indent=1))
    ds_key = os.environ.get("DEEPSEEK_API_KEY")
    oa_base = os.environ.get("OPENAI_BASE_URL")
    if oa_base:  # any OpenAI-compatible endpoint
        r = requests.post(
            oa_base.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
            json={"model": os.environ["OPENAI_MODEL"],
                  "messages": [{"role": "user", "content": prompt}],
                  "max_completion_tokens": 1024},
            timeout=90)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    elif os.environ.get("PUTER_AUTH_TOKEN"):  # Claude via Puter.js (Node helper)
        p = subprocess.run(["node", os.path.join(os.path.dirname(os.path.abspath(__file__)), "puter_ai.cjs")],
                           input=prompt, capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            raise RuntimeError(f"puter: {p.stderr.strip()}")
        text = p.stdout
    elif os.environ.get("GEMINI_API_KEY"):  # Google Gemini
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": 2048}},
            timeout=90)
        r.raise_for_status()
        text = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
    elif ds_key:  # DeepSeek (OpenAI-compatible API)
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {ds_key}"},
            json={"model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                  "messages": [{"role": "user", "content": prompt}],
                  "response_format": {"type": "json_object"},
                  "max_tokens": 800},
            timeout=60)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    else:  # Claude
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=MODEL, max_tokens=800, messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in msg.content if b.type == "text")
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def validate(sig, snaps):
    """Reject anything that doesn't have 4 entries / 1 TP / SL in a sane order."""
    price = {s["symbol"]: s["price"] for s in snaps}
    if sig.get("symbol") not in price:
        raise ValueError("unknown symbol")
    side = sig.get("side")
    e, tp, sl = sig.get("entries"), sig.get("tp"), sig.get("sl")
    if side not in ("LONG", "SHORT") or not isinstance(e, list) or len(e) != 4:
        raise ValueError("need side and exactly 4 entries")
    e = [float(x) for x in e]
    tp, sl = float(tp), float(sl)
    p = price[sig["symbol"]]
    if side == "LONG":
        ok = e == sorted(e, reverse=True) and len(set(e)) == 4 and sl < e[-1] and tp > e[0] and e[0] <= p * 1.002
    else:
        ok = e == sorted(e) and len(set(e)) == 4 and sl > e[-1] and tp < e[0] and e[0] >= p * 0.998
    if not ok:
        raise ValueError("price levels inconsistent with side")
    # sanity: loss at SL must stay under 50% of margin at 10x (~5% price move)
    avg = sum(e) / 4
    if abs(avg - sl) / avg * LEVERAGE >= 0.5:
        raise ValueError("SL too wide for 10x")
    sig.update(entries=e, tp=tp, sl=sl)
    return sig


def fmt(x):
    """Plain decimal, no thousands separators or exponents (Cornix can't parse those)."""
    return f"{x:.8f}".rstrip("0").rstrip(".")


def format_message(sig):
    """Cornix-compatible signal format. Keep it plain text with no extra numbers."""
    base = sig["symbol"][:-4] if sig["symbol"].endswith("USDT") else sig["symbol"]
    e = sig["entries"]
    return "\n".join([
        f"#{base}/USDT",
        sig["side"].capitalize(),
        f"Leverage: {MARGIN_MODE} ({LEVERAGE}x)",
        "",
        "Entry Targets:",
        *[f"{i}) {fmt(x)}" for i, x in enumerate(e, 1)],
        "",
        "Take-Profit Targets:",
        f"1) {fmt(sig['tp'])}",
        "",
        "Stop Targets:",
        f"1) {fmt(sig['sl'])}",
    ])


# ---------- telegram ----------

def post(text):
    r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      json={"chat_id": TG_CHANNEL, "text": text}, timeout=15)
    r.raise_for_status()


def run_once(dry=False):
    snaps = []
    for s in SYMBOLS:
        try:
            snaps.append(snapshot(s))
        except Exception as ex:
            print(f"skip {s}: {ex}")
    if not snaps:
        raise RuntimeError("no market data")
    sig = None
    for attempt in range(3):  # AI output is re-asked if it fails validation
        try:
            sig = validate(ask_ai(snaps), snaps)
            break
        except Exception as ex:
            print(f"attempt {attempt + 1} rejected: {ex}")
    if sig is None:
        print("no valid signal this round")
        return
    msg = format_message(sig)
    print(msg)
    print(f"reason: {sig.get('reason', '')}")
    if not dry:
        post(msg)


def main():
    dry = "--dry" in sys.argv
    if not dry and not (TG_TOKEN and TG_CHANNEL):
        sys.exit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID (see .env.example)")
    if "--once" in sys.argv or dry:
        return run_once(dry)
    while True:
        try:
            run_once()
        except Exception as ex:
            print(f"error: {ex}")
        time.sleep(INTERVAL_MIN * 60 + random.randint(0, 60))


if __name__ == "__main__":
    main()
