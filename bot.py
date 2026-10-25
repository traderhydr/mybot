"""AI trade-signal bot: Claude analyses every Binance futures coin and posts
signals (4 entries, 1 TP, SL, cross 10x) to a Telegram channel."""
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv

load_dotenv()

# LLM clients and Telegram/Cornix output live in their own modules (shared with the v2 engine).
from llm import AI_TIMEOUT, CLAUDE_MODEL, _extract_json, ask_claude, enabled_ais  # noqa: E402,F401
from notify import LEVERAGE, MARGIN_MODE, fmt, format_message, post  # noqa: E402,F401

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHANNEL = os.environ.get("TELEGRAM_CHANNEL_ID", "")
# "ALL" = every USDT perpetual on Binance futures; or a comma list like BTCUSDT,ETHUSDT
SYMBOLS = [s.strip().upper() for s in os.environ.get("SYMBOLS", "ALL").split(",") if s.strip()]
MIN_VOLUME = float(os.environ.get("MIN_24H_VOLUME_USDT", "0"))  # skip illiquid coins (0 = keep all)
MAX_COINS = int(os.environ.get("MAX_COINS", "0"))  # 0 = no cap; otherwise the N most-traded coins
COIN_DELAY = float(os.environ.get("COIN_DELAY_SECONDS", "3"))  # pause between coins (rate limits)
TIMEFRAME = os.environ.get("TIMEFRAME", "1h")
INTERVAL_MIN = int(os.environ.get("INTERVAL_MINUTES", "240"))

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


def list_symbols():
    """All trading USDT perpetuals, most-traded first (or the SYMBOLS list from .env)."""
    if SYMBOLS != ["ALL"]:
        return SYMBOLS
    info = requests.get(f"{BINANCE}/fapi/v1/exchangeInfo", timeout=20)
    info.raise_for_status()
    live = {x["symbol"] for x in info.json()["symbols"]
            if x["status"] == "TRADING" and x["contractType"] == "PERPETUAL" and x["quoteAsset"] == "USDT"}
    tick = requests.get(f"{BINANCE}/fapi/v1/ticker/24hr", timeout=20)
    tick.raise_for_status()
    vol = {t["symbol"]: float(t["quoteVolume"]) for t in tick.json() if t["symbol"] in live}
    coins = [c for c, v in sorted(vol.items(), key=lambda kv: -kv[1]) if v >= MIN_VOLUME]
    return coins[:MAX_COINS] if MAX_COINS else coins


def snapshot(symbol):
    return snapshot_from_candles(symbol, fetch_candles(symbol))


def snapshot_from_candles(symbol, c):
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
    """Ask every configured AI. With two, they must agree on symbol + side; levels are averaged."""
    prompt = PROMPT.format(lev=LEVERAGE, data=json.dumps(snaps, indent=1))
    ais = enabled_ais()
    with ThreadPoolExecutor(len(ais)) as pool:  # ask all AIs at the same time
        futures = {name: pool.submit(fn, prompt) for name, fn in ais.items()}
        sigs = {}
        for name, fut in futures.items():
            try:
                sig = validate(fut.result(), snaps)  # each AI's own answer must be valid first
            except Exception as ex:
                raise ValueError(f"{name} failed: {ex}") from ex
            print(f"{name}: {sig['symbol']} {sig['side']}")
            sigs[name] = sig
    vals = list(sigs.values())
    if len(vals) == 1:
        return vals[0]
    g, c = vals
    if (g["symbol"], g["side"]) != (c["symbol"], c["side"]):
        raise ValueError("AIs disagree, no trade")
    return {
        "symbol": g["symbol"], "side": g["side"],
        "entries": [(x + y) / 2 for x, y in zip(g["entries"], c["entries"])],
        "tp": (g["tp"] + c["tp"]) / 2, "sl": (g["sl"] + c["sl"]) / 2,
        "reason": " | ".join(f"{n}: {x.get('reason', '')}" for n, x in sigs.items()),
    }


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


def run_once(dry=False, limit=None):
    """One cycle: analyse every coin and post a signal for each one that gets a valid setup."""
    coins = list_symbols()
    if limit:
        coins = coins[:limit]
    print(f"cycle: {len(coins)} coins")
    posted = 0
    for i, coin in enumerate(coins, 1):
        try:
            snaps = [snapshot(coin)]
            sig = None
            for attempt in range(2):  # AI output is re-asked if it fails validation
                try:
                    sig = validate(ask_ai(snaps), snaps)  # merged signal is re-checked too
                    break
                except Exception as ex:
                    print(f"[{i}/{len(coins)}] {coin} attempt {attempt + 1} rejected: {ex}")
            if sig is None:
                continue
            msg = format_message(sig)
            print(f"[{i}/{len(coins)}] {coin}\n{msg}\nreason: {sig.get('reason', '')}")
            if not dry:
                post(msg)
            posted += 1
        except Exception as ex:
            print(f"[{i}/{len(coins)}] {coin} error: {ex}")
        time.sleep(COIN_DELAY)
    print(f"cycle done: {posted}/{len(coins)} signals")


def main():
    if os.environ.get("STRATEGY", "v1").lower() == "v2":  # multi-timeframe / derivatives engine (execution.py)
        import execution
        return execution.main(sys.argv[1:])
    dry = "--dry" in sys.argv
    if not enabled_ais():
        sys.exit("Set ANTHROPIC_API_KEY (see .env.example)")
    print("AIs:", ", ".join(enabled_ais()))
    if not dry and not (TG_TOKEN and TG_CHANNEL):
        sys.exit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID (see .env.example)")
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else (3 if dry else None)
    if "--once" in sys.argv or dry:
        return run_once(dry, limit)
    while True:
        started = time.time()
        try:
            run_once(limit=limit)
        except Exception as ex:
            print(f"error: {ex}")
        # next cycle starts INTERVAL_MIN after this one started (immediately if the cycle ran longer)
        time.sleep(max(0, INTERVAL_MIN * 60 - (time.time() - started)) + random.randint(0, 30))


if __name__ == "__main__":
    main()
