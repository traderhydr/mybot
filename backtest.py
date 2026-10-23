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
import random
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


def with_side(sig, price, side):
    """The AI's ladder geometry (distances from price) applied to a fixed side: the control trades."""
    d_e = [abs(e - price) for e in sig["entries"]]
    d_tp, d_sl = abs(sig["tp"] - price), abs(sig["sl"] - price)
    if side == "SHORT":
        return {"side": side, "entries": [price + d for d in d_e], "tp": price - d_tp, "sl": price + d_sl}
    return {"side": side, "entries": [price - d for d in d_e], "tp": price + d_tp, "sl": price - d_sl}


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

def stats(rows):
    done = [r for r in rows if r["outcome"] in ("tp", "sl", "expired")]
    if not done:
        return None
    rets = [r["ret_alloc"] for r in done]
    wins, losses = [x for x in rets if x > 0], [x for x in rets if x <= 0]
    return {"n": len(done), "shorts": sum(r["side"] == "SHORT" for r in done) / len(done),
            "win": len(wins) / len(done), "pf": (sum(wins) / -sum(losses)) if losses and sum(losses) else float("inf"),
            "avg": statistics.mean(rets), "se": statistics.stdev(rets) / len(rets) ** 0.5 if len(rets) > 1 else 0.0,
            "total": sum(rets), "avg_win": statistics.mean(wins) if wins else 0.0,
            "avg_loss": statistics.mean(losses) if losses else 0.0,
            "tp": sum(r["outcome"] == "tp" for r in done), "sl": sum(r["outcome"] == "sl" for r in done),
            "expired": sum(r["outcome"] == "expired" for r in done)}


def report(variants):
    sig = variants["signal"]
    st = stats(sig)
    print(f"\nsignals {len(sig)} | cancelled {sum(r['outcome'] == 'cancelled' for r in sig)}"
          f" | never filled {sum(r['outcome'] == 'nofill' for r in sig)}")
    if not st:
        print("no completed trades")
        return
    print(f"AI trades: {st['n']} | TP {st['tp']} | SL {st['sl']} | expired {st['expired']} | "
          f"avg win {st['avg_win']:+.1f}% | avg loss {st['avg_loss']:+.1f}%")
    print(f"\n{'variant':14s}{'trades':>7s}{'%short':>8s}{'win%':>6s}{'PF':>6s}{'avg/trade (±SE)':>20s}{'total':>9s}")
    labels = {"signal": "AI", "mirror": "AI flipped", "always_short": "always SHORT",
              "always_long": "always LONG", "random": "random side"}
    for v, rows in variants.items():
        x = stats(rows)
        if x:
            print(f"{labels[v]:14s}{x['n']:7d}{x['shorts']:8.0%}{x['win']:6.0%}{x['pf']:6.2f}"
                  f"{x['avg']:+12.2f}% ±{x['se']:4.2f}{x['total']:+9.0f}%")
    print("\nAI minus control, per signal, same points and same ladder geometry (z>2 = unlikely to be luck):")
    for v in ("mirror", "always_short", "always_long", "random"):
        diffs = [a["ret_alloc"] - b["ret_alloc"] for a, b in zip(sig, variants[v])]
        if len(diffs) > 1:
            m, se = statistics.mean(diffs), statistics.stdev(diffs) / len(diffs) ** 0.5
            print(f"  vs {labels[v]:13s} {m:+6.2f}% ± {se:4.2f}   z = {m / se if se else 0:+.1f}")
    print("Trades overlap in time, so true uncertainty is larger than these SEs suggest.")


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

    variants = {v: [] for v in ("signal", "mirror", "always_short", "always_long", "random")}
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
        price = snap["price"]
        base = {"symbol": sym, "t": t, "price": price}
        opposite = "LONG" if sig["side"] == "SHORT" else "SHORT"
        trades = {"signal": sig,
                  "mirror": with_side(sig, price, opposite),
                  "always_short": with_side(sig, price, "SHORT"),
                  "always_long": with_side(sig, price, "LONG"),
                  "random": with_side(sig, price, random.Random(f"{sym}{t}").choice(["LONG", "SHORT"]))}
        for v, tr in trades.items():
            res = simulate(tr["side"], tr["entries"], tr["tp"], tr["sl"], fwd, a.fee)
            variants[v].append({**base, "side": tr["side"], "entries": tr["entries"], "tp": tr["tp"],
                                "sl": tr["sl"], **res})

    rows = variants["signal"]
    with open(a.out, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["variant"])
            w.writeheader()
            for v, rs in variants.items():
                for r in rs:
                    w.writerow({**r, "variant": v})
    report(variants)
    print(f"\nrows saved to {a.out}. Caveats: candle-level simulation, no slippage/funding, equal margin per trade,"
          " trades treated independently.")


if __name__ == "__main__":
    main()
