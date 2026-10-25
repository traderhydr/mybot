"""Backtest of the v2 engine on historical data.

It runs the *live* code path (ingest.build_context -> decision.* -> state locks) against a client that can only see
data available at each simulated moment (no look-ahead), then simulates the 4-entry ladder on the following candles with
the same pessimistic rules as backtest.py. Controls (flipped / always SHORT / always LONG / random side) are traded on the
same points with the same ladder shape, and a direction-only test uses one standard ladder.

  python backtest_v2.py --ai rule   --coins 15 --days 28             # no LLM: gates + 4h-bias direction (free)
  python backtest_v2.py --ai claude --coins 15 --days 28 --yes       # the real thing (Claude calls, cached)

Open-interest history only exists for ~30 days, so days + end-days-ago must be <= 29 with --source futures.
Funnel, tables and rows (backtest_v2_results.csv) are printed/saved at the end.
"""
import argparse
import bisect
import collections
import csv
import hashlib
import json
import logging
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import backtest as bt1
import execution
import ingest
import llm
from decision import DecisionError, build_payload, build_prompt, build_signal, enforce_rules, parse_decision, system_instructions
from state import StateManager

H = 3_600_000
DAY = 24 * H
CACHE_FILE = "backtest_v2_cache.jsonl"
WARM_DAYS = {"5m": 3, "15m": 6, "30m": 10, "1h": 14, "4h": 45, "1d": 120}
STATUS = {"tp": "TP", "sl": "SL", "expired": "EXPIRED", "cancelled": "CANCELLED", "nofill": "CANCELLED"}
log = logging.getLogger("bt2")


# ---------------------------------------------------------------- historical data

def fetch_klines(client, symbol, interval, start, end):
    start, end = int(start), int(end)  # Binance rejects float timestamps like "1.7e12" / "123.0"
    out, cur = [], start
    while cur < end:
        rows = client.get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "startTime": cur,
                                              "endTime": end, "limit": 1000})
        if not rows:
            break
        out += ingest.parse_kline_rows(rows, symbol)
        nxt = int(rows[-1][0]) + 1
        if len(rows) < 1000 or nxt <= cur:
            break
        cur = nxt
    return sorted({r["t"]: r for r in out}.values(), key=lambda r: r["t"])  # tolerate duplicated pages


def fetch_oi(client, symbol, start, end):
    """[(available_at_ms, open_interest)] oldest first. A point is only usable one period after its timestamp."""
    start, end = int(start), int(end)
    out, cur = [], start
    while cur < end:
        rows = client.get("/futures/data/openInterestHist", {"symbol": symbol, "period": "15m", "limit": 500,
                                                             "startTime": cur, "endTime": end})
        if not rows:
            break
        pts = sorted((int(r["timestamp"]), float(r["sumOpenInterest"])) for r in rows)
        out += [(ts + 900_000, v) for ts, v in pts]
        nxt = pts[-1][0] + 1
        if len(rows) < 500 or nxt <= cur:
            break
        cur = nxt
    return out


def fetch_funding(client, symbol, start, end):
    rows = client.get("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": int(start), "endTime": int(end), "limit": 1000})
    return sorted((int(r["fundingTime"]), float(r["fundingRate"])) for r in rows)


class History:
    """Raw history for all symbols; HistoricalClient reads it 'as of' a moment."""

    def __init__(self):
        self.candles, self.closes, self.oi, self.funding = {}, {}, {}, {}

    def add_candles(self, symbol, interval, rows):
        tf = ingest.TF_MS[interval]
        self.candles[(symbol, interval)] = rows
        self.closes[(symbol, interval)] = [r["t"] + tf for r in rows]

    def forward(self, symbol, interval, t0, bars):
        rows = self.candles[(symbol, interval)]
        i = bisect.bisect_left([r["t"] for r in rows], t0)
        return rows[i:i + bars]


class HistoricalClient:
    """Same interface the live engine uses, restricted to data that existed at self.T."""

    def __init__(self, hist):
        self.hist, self.T = hist, 0

    def klines(self, symbol, interval, limit):
        key = (symbol, interval)
        if key not in self.hist.candles:
            raise ingest.DataError(f"no history for {symbol} {interval}")
        i = bisect.bisect_right(self.hist.closes[key], self.T)  # only candles that had closed by T
        return self.hist.candles[key][max(0, i - limit):i]

    def open_interest_hist(self, symbol, period="15m", limit=30):
        pts = self.hist.oi.get(symbol, [])
        i = bisect.bisect_right([p[0] for p in pts], self.T)
        return [v for _, v in pts[max(0, i - limit):i]]


_shared_cache, _shared_lock = {}, threading.Lock()


def shared_at(hist, T, symbols):
    with _shared_lock:
        if T in _shared_cache:
            return _shared_cache[T]
    c = HistoricalClient(hist)
    c.T = T
    shared = {"btc": None, "premium": {}, "intervals": {}, "errors": []}
    try:
        shared["btc"] = ingest.btc_context(c.klines("BTCUSDT", "1h", 300), c.klines("BTCUSDT", "4h", 300))
    except (ingest.DataError, ValueError) as ex:
        shared["errors"].append(f"btc: {ex}")
    for s in symbols:
        f = hist.funding.get(s, [])
        i = bisect.bisect_right([x[0] for x in f], T)
        if i:
            gaps = [(f[j][0] - f[j - 1][0]) / H for j in range(max(1, i - 3), i)]
            shared["premium"][s] = {"rate": f[i - 1][1], "mark": 0.0}
            shared["intervals"][s] = round(sorted(gaps)[len(gaps) // 2]) if gaps else 8
    with _shared_lock:
        _shared_cache[T] = shared
    return shared


# ---------------------------------------------------------------- decisions

def rule_decision(ctx, lock):
    """No-LLM baseline: trade the 4h structure bias when the entry-timeframe structure agrees (gates still apply)."""
    htf = ctx["market_structure"]["htf_bias"]["4h"]
    ltf = ctx["market_structure"]["ltf_structure"]["bias"]
    side = {"BULLISH": "LONG", "BEARISH": "SHORT"}.get(htf)
    none = {"action": "NO_TRADE", "confidence_score": 0.0, "confluence_factors": [], "invalidation_level": 0.0,
            "entry_price": 0.0, "take_profit_levels": [], "risk_reward_ratio": 0.0, "reasoning": "no aligned bias"}
    if not side or ltf != htf:
        return none
    sgn = 1 if side == "LONG" else -1
    p, a = ctx["price"], ctx["atr_entry"]
    return {"action": f"OPEN_{side}", "confidence_score": 0.7, "confluence_factors": ["4h bias", "ltf agrees", "rvol"],
            "invalidation_level": p - sgn * 1.5 * a, "entry_price": p, "take_profit_levels": [p + sgn * 3 * a],
            "risk_reward_ratio": 2.0, "reasoning": "rule baseline"}


_cache, _cache_lock = {}, threading.Lock()


def load_cache():
    try:
        for line in open(CACHE_FILE):
            d = json.loads(line)
            _cache[d["k"]] = d["d"]
    except (OSError, ValueError, KeyError):
        pass


def cache_key(system, prompt):
    return hashlib.sha1(f"{llm.CLAUDE_MODEL}\n{system}\n{prompt}".encode()).hexdigest()


def claude_decision(system, prompt):
    """Cached Claude call. Bad replies are cached as {"_error"}; transport errors raise (and are not cached)."""
    key = cache_key(system, prompt)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    try:
        d = parse_decision(llm.ask_claude(prompt, system=system, max_tokens=1200))
    except (DecisionError, ValueError, KeyError) as ex:
        d = {"_error": str(ex)[:120]}
    with _cache_lock:
        _cache[key] = d
        with open(CACHE_FILE, "a") as f:
            f.write(json.dumps({"k": key, "d": d}) + "\n")
    return d


# ---------------------------------------------------------------- per-coin timeline

def prepare(symbol, times, hist, shared_for, cfg):
    """Build the context at every sample time (CPU only). Returns [(T, ctx | None, why_not)]."""
    client = HistoricalClient(hist)
    out = []
    for T in times:
        client.T = T
        try:
            out.append((T, ingest.build_context(symbol, client, shared_for(T), cfg.entry_tf), None))
        except (ingest.DataError, ValueError, ZeroDivisionError, IndexError) as ex:
            out.append((T, None, f"data: {str(ex)[:40]}"))
    return out


def run_coin(symbol, prepared, hist, cfg, mode, a):
    rules = cfg.rules
    tf = ingest.TF_MS[cfg.entry_tf]
    expiry_bars = int(cfg.expiry_hours * H // tf)
    vt = [0]
    state = StateManager(":memory:", cfg.cooldown_hours, cfg.flip_lock_hours, clock=lambda: vt[0] / 1000)
    system = system_instructions(rules)
    funnel, rejects, rows = collections.Counter(), collections.Counter(), {v: [] for v in bt1.LABELS}
    pending = None
    for T, ctx, why in prepared:
        vt[0] = T
        if pending and pending["closed_at"] <= T:  # the simulated position has finished by now
            r = pending
            state.close_position(r["pid"], STATUS[r["outcome"]], r["closed_at"], None, r["ret_dep"])
            pending = None
        funnel["points"] += 1
        if ctx is None:
            funnel["data_error"] += 1
            continue
        lock = state.lock_status(symbol, T)
        if lock["locked"]:
            funnel["locked"] += 1
            continue
        if cfg.pregate_rvol and (ctx["rvol"] is None or ctx["rvol"] < rules.rvol_min):
            funnel["rvol_gate"] += 1
            continue
        if mode == "rule":
            d = rule_decision(ctx, lock)
        else:
            try:
                d = claude_decision(system, build_prompt(build_payload(ctx, lock)))
            except Exception as ex:  # transport problem: skip the point, rerun later to fill it in (not cached)
                funnel["llm_error"] += 1
                log.warning("%s @%s: %s", symbol, T, ex)
                continue
            if "_error" in d:
                funnel["bad_reply"] += 1
                continue
            funnel["asked"] += 1
        if d["action"] in ("HOLD", "NO_TRADE"):
            funnel[d["action"]] += 1
            continue
        bad = enforce_rules(d, ctx, lock, rules)
        sig = None if bad else build_signal(d, ctx, rules)
        if sig and sig["rr"] < rules.min_rr_ladder:
            bad = [f"ladder R:R below {rules.min_rr_ladder}"]
        if bad:
            funnel["rejected"] += 1
            rejects[re.sub(r"[\d.]+", "#", bad[0])[:60]] += 1
            continue
        funnel["signals"] += 1
        price, atr_v = ctx["price"], ctx["atr_entry"]
        fwd = hist.forward(symbol, cfg.entry_tf, T, expiry_bars)
        base = {"symbol": symbol, "t": T, "price": price}
        opposite = "LONG" if sig["side"] == "SHORT" else "SHORT"
        trades = {"signal": sig,
                  "mirror": bt1.with_side(sig, price, opposite),
                  "always_short": bt1.with_side(sig, price, "SHORT"),
                  "always_long": bt1.with_side(sig, price, "LONG"),
                  "random": bt1.with_side(sig, price, random.Random(f"{symbol}{T}").choice(["LONG", "SHORT"]))}
        rnd = random.Random(f"fx{symbol}{T}").choice(["LONG", "SHORT"])
        for v, side in (("fixed_ai", sig["side"]), ("fixed_flipped", opposite), ("fixed_short", "SHORT"),
                        ("fixed_long", "LONG"), ("fixed_random", rnd)):
            trades[v] = bt1.fixed_trade(side, price, atr_v, a.fx_tp, a.fx_sl)
        for v, tr in trades.items():
            res = bt1.simulate(tr["side"], tr["entries"], tr["tp"], tr["sl"], fwd, a.fee)
            rows[v].append({**base, "side": tr["side"], "entries": tr["entries"], "tp": tr["tp"], "sl": tr["sl"], **res})
            if v == "signal":
                pid = state.add_position(sig, T - 1)  # lock this coin until the simulated position ends
                pending = {"pid": pid, "outcome": res["outcome"], "ret_dep": res["ret_dep"],
                           "closed_at": T + res["bars"] * tf}
    return rows, funnel, rejects


# ---------------------------------------------------------------- main

class SpotClient(ingest.BinanceClient):
    """Harness testing only: public spot candles via the data mirror; no futures data (funding / OI stay empty)."""

    def __init__(self):
        super().__init__(base="https://data-api.binance.vision")

    def get(self, path, params=None):
        if path.startswith("/fapi/v1/klines"):
            return super().get("/api/v3/klines", params)
        raise ingest.DataError("not available on spot")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", choices=["claude", "rule"], default="rule")
    ap.add_argument("--symbols", help="comma list; default = most-traded perpetuals")
    ap.add_argument("--coins", type=int, default=15)
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--end-days-ago", type=int, default=0)
    ap.add_argument("--step-hours", type=int, default=4, help="how often each coin is evaluated")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fee", type=float, default=0.0007)
    ap.add_argument("--fx-tp", type=float, default=1.5)
    ap.add_argument("--fx-sl", type=float, default=1.6)
    ap.add_argument("--source", choices=["futures", "spot"], default="futures")
    ap.add_argument("--out", default="backtest_v2_results.csv")
    ap.add_argument("--yes", action="store_true", help="confirm spending Claude calls")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if a.source == "futures" and a.days + a.end_days_ago > 29:
        sys.exit("Open-interest history only covers ~30 days: use --days + --end-days-ago <= 29 (or --source spot to test without it).")
    cfg = execution.Config.from_env()
    tf = ingest.TF_MS[cfg.entry_tf]
    client = SpotClient() if a.source == "spot" else ingest.BinanceClient()
    end = int(time.time() * 1000) - a.end_days_ago * DAY
    start = end - a.days * DAY
    if a.symbols:
        symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    elif a.source == "spot":
        sys.exit("--source spot needs --symbols (it has no coin list).")
    else:
        symbols = client.universe(["ALL"], a.coins, cfg.min_volume)
    symbols = list(dict.fromkeys(symbols))
    need = set(symbols) | {"BTCUSDT"}

    hist = History()
    print(f"loading history for {len(need)} symbols ({a.source}) ...")
    for sym in sorted(need):
        try:
            for iv in {cfg.entry_tf, "4h", "1d"} | ({"1h"} if sym == "BTCUSDT" else set()):
                warm = WARM_DAYS.get(iv, 30) * DAY
                hist.add_candles(sym, iv, fetch_klines(client, sym, iv, start - warm, end + cfg.expiry_hours * H))
            if a.source == "futures":
                hist.oi[sym] = fetch_oi(client, sym, start - DAY, end)
                hist.funding[sym] = fetch_funding(client, sym, start - 2 * DAY, end)
        except ingest.DataError as ex:
            print(f"  skip {sym}: {ex}")
            symbols = [s for s in symbols if s != sym]
    if "BTCUSDT" not in {k[0] for k in hist.candles}:
        sys.exit("BTCUSDT history is required (market regime) but could not be loaded.")

    step = a.step_hours * H
    last = end - int(cfg.expiry_hours * H)  # leave room for every position to resolve
    times = list(range(-(-start // step) * step, last + 1, step))
    print(f"{len(symbols)} coins x {len(times)} sample times ({a.days}d, every {a.step_hours}h), mode={a.ai}, entry TF {cfg.entry_tf}")
    shared_for = lambda T: shared_at(hist, T, symbols)
    with ThreadPoolExecutor(a.workers) as pool:
        prepared = dict(zip(symbols, pool.map(lambda s: prepare(s, times, hist, shared_for, cfg), symbols)))

    if a.ai == "claude":
        load_cache()
        system = system_instructions(cfg.rules)
        fresh = collections.Counter()
        for s, pts in prepared.items():
            for T, ctx, _ in pts:
                if ctx and ctx["rvol"] is not None and ctx["rvol"] >= cfg.rules.rvol_min:
                    prompt = build_prompt(build_payload(ctx, {"allowed": ["LONG", "SHORT"], "bias": None}))
                    fresh["cached" if cache_key(system, prompt) in _cache else "new"] += 1
        print(f"points passing the RVOL gate: {sum(fresh.values())} (upper bound on Claude calls; "
              f"{fresh['new']} not cached; locks will remove some)")
        if fresh["new"] and not a.yes:
            sys.exit("Re-run with --yes to spend the Claude calls (answers are cached, so stop/resume is free).")

    variants = {v: [] for v in bt1.LABELS}
    funnel, rejects = collections.Counter(), collections.Counter()
    with ThreadPoolExecutor(a.workers) as pool:
        for rows, f, r in pool.map(lambda s: run_coin(s, prepared[s], hist, cfg, a.ai, a), symbols):
            for v in variants:
                variants[v] += rows[v]
            funnel.update(f)
            rejects.update(r)
    for v in variants:
        variants[v].sort(key=lambda r: (r["t"], r["symbol"]))

    print("\nfunnel:", ", ".join(f"{k} {funnel[k]}" for k in
          ("points", "data_error", "locked", "rvol_gate", "asked", "llm_error", "bad_reply", "NO_TRADE", "HOLD", "rejected", "signals") if funnel[k]))
    if rejects:
        print("top rejection reasons:", "; ".join(f"{k} x{n}" for k, n in rejects.most_common(4)))
    rows = variants["signal"]
    if rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) + ["variant"])
            w.writeheader()
            for v, rs in variants.items():
                for r in rs:
                    w.writerow({**r, "variant": v})
    bt1.report(variants)
    print(f"\nrows saved to {a.out}. Same caveats as backtest.py; open interest/funding are as-of the sample time.")


if __name__ == "__main__":
    main()
