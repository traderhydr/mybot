"""Backtest the signal bot on historical candles.

For many (coin, time) points it builds the same snapshot the live bot sees, asks the same AI(s) with the same
prompt and validation, then simulates the 4-entry ladder / TP / SL on the candles that follow.

Examples:
  python backtest.py --ai bot --only claude --coins 5 --days 10 --yes   # one AI only (no Gemini quota needed)
  python backtest.py --ai rule --days 30                       # free baseline (no AI calls)
  python backtest.py --ai bot --coins 10 --days 14 --step-hours 12 --yes
Results are cached in backtest_cache.jsonl so an interrupted run resumes without paying twice.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

import requests

import bot

TF_MS = {"15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000}
WARMUP = 200  # candles the live snapshot uses
CACHE_FILE = "backtest_cache.jsonl"
URLS = {"futures": f"{bot.BINANCE}/fapi/v1/klines",
        "spot": "https://data-api.binance.vision/api/v3/klines"}


# ---------- data ----------

def klines(symbol, start_ms, end_ms, url):
    out, cur = [], start_ms
    while cur < end_ms:
        r = requests.get(url, params={"symbol": symbol, "interval": bot.TIMEFRAME, "startTime": cur,
                                      "endTime": end_ms, "limit": 1000}, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        out += [{"t": int(k[0]), "o": float(k[1]), "h": float(k[2]), "l": float(k[3]), "c": float(k[4]),
                 "v": float(k[5])} for k in rows]
        cur = int(rows[-1][0]) + 1
        if len(rows) < 1000:
            break
        time.sleep(0.2)
    return out


# ---------- simulation ----------

def simulate(side, entries, tp, sl, fwd, fee_rt, lev=bot.LEVERAGE):
    """Limit-order ladder, equal size per entry. Pessimistic: SL wins ties; a trade can't fill and take
    profit in the same candle. TP touched before any fill = cancelled (as Cornix does)."""
    short = side == "SHORT"
    filled = [False] * 4
    for j, c in enumerate(fwd):
        new = False
        for k, e in enumerate(entries):
            if not filled[k] and ((c["h"] >= e) if short else (c["l"] <= e)):
                filled[k], new = True, True
        n = sum(filled)
        tp_hit = c["l"] <= tp if short else c["h"] >= tp
        sl_hit = c["h"] >= sl if short else c["l"] <= sl
        if n == 0:
            if tp_hit:
                return {"outcome": "cancelled", "fills": 0, "ret_dep": 0.0, "ret_alloc": 0.0, "bars": j + 1}
            continue
        if sl_hit:
            exit_px, outcome = sl, "sl"
        elif tp_hit and not new:
            exit_px, outcome = tp, "tp"
        else:
            continue
        return _result(short, filled, entries, exit_px, outcome, j + 1, fee_rt, lev)
    n = sum(filled)
    if n == 0:
        return {"outcome": "nofill", "fills": 0, "ret_dep": 0.0, "ret_alloc": 0.0, "bars": len(fwd)}
    return _result(short, filled, entries, fwd[-1]["c"], "expired", len(fwd), fee_rt, lev)


def _result(short, filled, entries, exit_px, outcome, bars, fee_rt, lev):
    px = [e for e, f in zip(entries, filled) if f]
    avg = sum(px) / len(px)
    move = (avg - exit_px) / avg if short else (exit_px - avg) / avg
    ret_dep = (lev * move - lev * fee_rt) * 100   # % of the margin actually deployed
    return {"outcome": outcome, "fills": len(px), "ret_dep": ret_dep,
            "ret_alloc": ret_dep * len(px) / 4, "bars": bars}  # unfilled margin earns nothing


def mirror(sig, price):
    """Same trade flipped to the other side (levels reflected around the signal price)."""
    r = lambda x: 2 * price - x
    return {"side": "LONG" if sig["side"] == "SHORT" else "SHORT",
            "entries": [r(e) for e in sig["entries"]], "tp": r(sig["tp"]), "sl": r(sig["sl"])}


# ---------- signal sources ----------

def rule_signal(snap):
    """Free baseline: EMA trend-following ladder (not an AI)."""
    p, a = snap["price"], snap["atr14"]
    if snap["ema20"] < snap["ema50"] and p < snap["ema20"]:
        side, sgn = "SHORT", 1
    elif snap["ema20"] > snap["ema50"] and p > snap["ema20"]:
        side, sgn = "LONG", -1
    else:
        return None
    entries = [p + sgn * a * m for m in (0.1, 0.3, 0.5, 0.7)]
    sig = {"symbol": snap["symbol"], "side": side, "entries": entries,
           "sl": entries[-1] + sgn * 0.9 * a, "tp": entries[0] - sgn * 1.8 * a}
    try:
        return bot.validate(sig, [snap])
    except ValueError:
        return None


TRANSIENT = ("HTTP 429", "HTTP 5", "cooling", "imeout", "timed out", "Connection", "overloaded", "rate")


def ai_signal(snap, cache, key):
    if key in cache:
        return cache[key]
    hard, soft = 0, 0
    while hard < 2 and soft < 6:
        while "gemini" in bot.enabled_ais() and bot.gemini_blocked():
            print("  waiting out Gemini cooldown...")
            time.sleep(60)
        try:
            sig = bot.validate(bot.ask_ai([snap]), [snap])
            break
        except Exception as ex:
            if any(t in str(ex) for t in TRANSIENT):
                soft += 1
                time.sleep(30)
            else:
                hard += 1
    else:
        if soft >= 6:
            return "SKIP"  # not cached; will be retried on the next run
        sig = None
    cache[key] = sig
    with open(CACHE_FILE, "a") as f:
        f.write(json.dumps({"k": key, "sig": sig}) + "\n")
    return sig


def load_cache():
    cache = {}
    if os.path.exists(CACHE_FILE):
        for line in open(CACHE_FILE):
            try:
                d = json.loads(line)
                cache[d["k"]] = d["sig"]
            except Exception:
                pass
    return cache


# ---------- report ----------

def summarize(rows, title):
    done = [r for r in rows if r["outcome"] in ("tp", "sl", "expired")]
    print(f"\n=== {title} ===")
    print(f"signals {len(rows)} | traded {len(done)} | cancelled {sum(r['outcome']=='cancelled' for r in rows)}"
          f" | never filled {sum(r['outcome']=='nofill' for r in rows)}")
    if not done:
        return
    wins = [r["ret_alloc"] for r in done if r["ret_alloc"] > 0]
    losses = [r["ret_alloc"] for r in done if r["ret_alloc"] <= 0]
    out = {o: sum(r["outcome"] == o for r in done) for o in ("tp", "sl", "expired")}
    print(f"outcomes: TP {out['tp']} | SL {out['sl']} | expired {out['expired']}")
    print(f"win rate {len(wins) / len(done):.0%} | avg win {statistics.mean(wins) if wins else 0:+.1f}%"
          f" | avg loss {statistics.mean(losses) if losses else 0:+.1f}%"
          f" | profit factor {(sum(wins) / -sum(losses)) if losses and sum(losses) else float('inf'):.2f}")
    total = sum(r["ret_alloc"] for r in done)
    cum, peak, mdd = 0.0, 0.0, 0.0
    for r in sorted(done, key=lambda r: r["t"]):
        cum += r["ret_alloc"]
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    print(f"total {total:+.0f}% of one trade's margin | avg per trade {total / len(done):+.2f}% | max drawdown {mdd:.0f}%")
    for side in ("LONG", "SHORT"):
        s = [r for r in done if r["side"] == side]
        if s:
            print(f"  {side}: {len(s)} trades, win rate {sum(r['ret_alloc'] > 0 for r in s) / len(s):.0%},"
                  f" avg {statistics.mean(r['ret_alloc'] for r in s):+.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", choices=["bot", "rule"], default="rule", help="bot = your configured AI(s); rule = free EMA baseline")
    ap.add_argument("--symbols", help="comma list; default = most-traded coins")
    ap.add_argument("--coins", type=int, default=10)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--step-hours", type=int, default=12, help="how often to take a signal per coin")
    ap.add_argument("--expiry-hours", type=int, default=48, help="close open trades at market after this long")
    ap.add_argument("--fee", type=float, default=0.0007, help="round-trip fee as a fraction of notional")
    ap.add_argument("--source", choices=["futures", "spot"], default="futures")
    ap.add_argument("--out", default="backtest_results.csv")
    ap.add_argument("--yes", action="store_true", help="confirm spending AI calls")
    ap.add_argument("--only", choices=["claude", "gemini"], help="test just one AI instead of the pair")
    a = ap.parse_args()
    if a.only:  # drop the other AI's key so bot.ask_ai uses a single AI
        os.environ.pop("GEMINI_API_KEY" if a.only == "claude" else "ANTHROPIC_API_KEY", None)

    tf = TF_MS[bot.TIMEFRAME]
    step, expiry = max(1, a.step_hours * 3_600_000 // tf), a.expiry_hours * 3_600_000 // tf
    end = int(time.time() * 1000)
    start = end - a.days * 86_400_000 - WARMUP * tf
    symbols = [s.strip().upper() for s in a.symbols.split(",")] if a.symbols else bot.list_symbols()[:a.coins]
    cache = load_cache() if a.ai == "bot" else {}
    ais = ",".join(sorted(bot.enabled_ais())) + f"|{bot.GEMINI_MODEL}|{bot.CLAUDE_MODEL}" if a.ai == "bot" else "rule"

    data, points = {}, []
    for sym in symbols:
        try:
            c = klines(sym, start, end, URLS[a.source])
        except Exception as ex:
            print(f"skip {sym}: {ex}")
            continue
        data[sym] = c
        points += [(sym, i) for i in range(WARMUP - 1, len(c) - expiry - 1, step)]
    new = [p for p in points if f"{p[0]}|{data[p[0]][p[1]]['t']}|{ais}" not in cache]
    print(f"{len(symbols)} coins, {len(points)} signal points ({a.days}d, every {a.step_hours}h), mode={a.ai}")
    if a.ai == "bot":
        print(f"{len(new)} points need AI calls ({len(new) * len(bot.enabled_ais())} requests)")
        if new and not a.yes:
            sys.exit("Re-run with --yes to spend them (results are cached, so a stop/resume costs nothing extra).")

    rows, mrows = [], []
    for n, (sym, i) in enumerate(points, 1):
        c = data[sym]
        snap = bot.snapshot_from_candles(sym, c[i - WARMUP + 1:i + 1])
        t = c[i]["t"]
        sig = rule_signal(snap) if a.ai == "rule" else ai_signal(snap, cache, f"{sym}|{t}|{ais}")
        if a.ai == "bot" and n % 10 == 0:
            print(f"  {n}/{len(points)}")
        if sig in (None, "SKIP"):
            continue
        fwd = c[i + 1:i + 1 + expiry]
        base = {"symbol": sym, "t": t, "price": snap["price"]}
        res = simulate(sig["side"], sig["entries"], sig["tp"], sig["sl"], fwd, a.fee)
        rows.append({**base, "side": sig["side"], "entries": sig["entries"], "tp": sig["tp"], "sl": sig["sl"], **res})
        m = mirror(sig, snap["price"])
        mres = simulate(m["side"], m["entries"], m["tp"], m["sl"], fwd, a.fee)
        mrows.append({**base, **m, **mres})

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["variant"]) if rows else None
        if w:
            w.writeheader()
            for v, rs in (("signal", rows), ("mirror", mrows)):
                for r in rs:
                    w.writerow({**r, "variant": v})
    summarize(rows, f"BOT SIGNALS ({a.ai})")
    summarize(mrows, "CONTROL: same trades flipped to the opposite side")
    print(f"\nrows saved to {a.out}. Caveats: candle-level simulation, no slippage/funding, equal margin per trade,"
          " trades treated independently.")


if __name__ == "__main__":
    main()
