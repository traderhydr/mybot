"""Tests for the v2 engine (python -m unittest -v test_v2). No network, no API keys."""
import json
import math
import random
import time
import unittest

import execution
import ingest
from decision import (DecisionError, Rules, build_payload, build_prompt, build_signal, enforce_rules,
                      merge_decisions, parse_decision)
from state import StateManager, advance_position

TF = 900_000
H = 3_600_000


def C(t, o, h, l, c, v=100.0, tb=None):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v, "tb": v / 2 if tb is None else tb}


def series(closes, tf=TF, t0=0, spread=0.01, vol=100.0):
    out, prev = [], closes[0]
    for i, c in enumerate(closes):
        out.append(C(t0 + i * tf, prev, max(prev, c) + spread, min(prev, c) - spread, c, vol))
        prev = c
    return out


class Indicators(unittest.TestCase):
    def test_rvol(self):
        c = series([100] * 21)
        c[-1]["v"] = 300.0
        self.assertAlmostEqual(ingest.rvol(c), 3.0)
        self.assertIsNone(ingest.rvol(series([100] * 5)))

    def test_atr_rsi(self):
        c = series([100 + (i % 2) for i in range(60)], spread=0.5)
        self.assertGreater(ingest.atr(c), 1.0)
        self.assertEqual(ingest.rsi([float(i) for i in range(1, 40)]), 100.0)
        with self.assertRaises(ingest.DataError):
            ingest.atr(series([1, 2, 3]))

    def test_price_change(self):
        c = series([100] * 8 + [110], tf=H)
        self.assertAlmostEqual(ingest.price_change(c, 1), 0.10)
        self.assertIsNone(ingest.price_change(c, 100))

    def test_volume_profile(self):
        c = []
        for i in range(100):
            if i % 10 == 0:
                c.append(C(i * TF, 95, 110 if i % 20 == 0 else 90, 90 if i % 20 else 95, 100, v=10))  # wide, thin
            else:
                c.append(C(i * TF, 100, 100.5, 99.5, 100, v=1000))
        vp = ingest.volume_profile(c, 24)
        self.assertTrue(99.0 <= vp["poc"] <= 101.0, vp)
        self.assertLessEqual(vp["val"], vp["poc"])
        self.assertGreaterEqual(vp["vah"], vp["poc"])
        self.assertLess(vp["vah"] - vp["val"], 10)  # value area tighter than the full range
        self.assertIsNone(ingest.volume_profile(series([100] * 5), 24))


class Structure(unittest.TestCase):
    PATH = [10, 11, 12, 11, 10.5, 11.5, 13, 12, 11.2, 12.5, 10]

    def test_bos_then_choch(self):
        up = ingest.market_structure(series(self.PATH[:7]), 1, 1)
        self.assertEqual((up["bias"], up["last_event"]), ("BULLISH", "BOS_BULL"))
        full = ingest.market_structure(series(self.PATH), 1, 1)
        self.assertEqual((full["bias"], full["last_event"]), ("BEARISH", "CHOCH_BEAR"))
        self.assertEqual(full["bars_since_event"], 0)

    def test_range_when_no_break(self):
        self.assertEqual(ingest.market_structure(series([10, 10.1, 10, 10.1, 10]), 1, 1)["bias"], "RANGE")

    def test_fvg_and_mitigation(self):
        base = [C(0, 9, 10, 9, 9.5), C(TF, 9.5, 12, 9.5, 11.9), C(2 * TF, 11.5, 13, 11, 12.5)]
        far = base + [C(3 * TF, 12.5, 14, 12, 13.5)]
        z = ingest.find_fvgs(far, 1.0)
        self.assertEqual([(x["type"], x["bottom"], x["top"], x["mitigated"]) for x in z], [("BULL", 10, 11, False)])
        touched = base + [C(3 * TF, 12.5, 13, 10.5, 11)]
        self.assertTrue(ingest.find_fvgs(touched, 1.0)[0]["mitigated"])
        near = ingest.nearest_unmitigated_fvgs(z, 12.5, 1.0)
        self.assertEqual(near[0]["type"], "BULL_FVG")
        self.assertLess(near[0]["dist_atr"], 0)
        bear = [C(0, 20, 21, 20, 20.5), C(TF, 20.5, 20.5, 17, 17.5), C(2 * TF, 17.5, 18, 16, 17)]
        self.assertEqual(ingest.find_fvgs(bear, 1.0)[0]["type"], "BEAR")

    def test_liquidity_equal_highs(self):
        path = [9, 10, 12, 10, 9, 10, 12.02, 10, 9, 10, 9]
        zones = ingest.liquidity_zones(series(path, spread=0.0), [C(0, 9, 13, 8, 10)], 10.0, 1.0)
        kinds = {z["type"] for z in zones}
        self.assertIn("PDH", kinds)
        self.assertIn("EQH", kinds)


class MarketContext(unittest.TestCase):
    def test_states(self):
        up = series([100 + i * 0.5 for i in range(120)], spread=0.3)
        self.assertEqual(ingest.trend_state(up)["state"], "BULLISH_EXPANSION")
        down = series([160 - i * 0.5 for i in range(120)], spread=0.3)
        self.assertEqual(ingest.trend_state(down)["state"], "BEARISH_EXPANSION")
        quiet = series([100] * 100, spread=2.0) + series([100] * 20, spread=0.1, t0=100 * TF)
        self.assertEqual(ingest.trend_state(quiet)["state"], "COMPRESSION")
        rej = up[:-1] + [C(up[-1]["t"], 160, 175, 159, 159.5)]
        r = ingest.trend_state(rej)
        self.assertEqual((r["state"], r["rejection_side"]), ("REJECTION", "TOP"))
        rb = down[:-1] + [C(down[-1]["t"], 100, 101, 85, 100.5)]
        self.assertEqual(ingest.trend_state(rb)["rejection_side"], "BOTTOM")

    def test_correlation_risk(self):
        mk = lambda st, side=None: {"state": st, "rejection_side": side}
        r = lambda sym, btc: ingest.correlation_risk(sym, btc)
        self.assertTrue(r("ETHUSDT", mk("BEARISH_EXPANSION"))["long"])
        self.assertFalse(r("ETHUSDT", mk("BEARISH_EXPANSION"))["short"])
        self.assertTrue(r("ETHUSDT", mk("REJECTION", "TOP"))["long"])
        self.assertTrue(r("ETHUSDT", mk("REJECTION", "BOTTOM"))["short"])
        self.assertTrue(r("ETHUSDT", mk("BULLISH_EXPANSION"))["short"])
        self.assertFalse(r("ETHUSDT", mk("COMPRESSION"))["long"])
        self.assertEqual((r("BTCUSDT", mk("BEARISH_EXPANSION"))["long"], r("BTCUSDT", None)["known"]), (False, True))
        self.assertFalse(r("ETHUSDT", None)["known"])

    def test_positioning_regime(self):
        p = ingest.positioning_regime
        self.assertEqual(p(0.01, 0.02), "AGGRESSIVE_LONG_BUILD")
        self.assertEqual(p(-0.01, 0.02), "AGGRESSIVE_SHORT_BUILD")
        self.assertEqual(p(0.01, -0.02), "SHORT_SQUEEZE_COVERING")
        self.assertEqual(p(-0.01, -0.02), "LONG_LIQUIDATION")
        self.assertEqual(p(0.0001, 0.02), "NEUTRAL")
        self.assertEqual(p(None, 0.02), "NEUTRAL")

    def test_cvd_bearish_divergence(self):
        c = [C(i * TF, 100, 100.5, 99.5, 100, 100, 50) for i in range(36)]
        for i in range(24):
            hi = 110 if i == 10 else 112 if i == 20 else 101
            tb = 80 if i <= 10 else 50 if i <= 17 else 10
            c.append(C((36 + i) * TF, 100, hi, 99.5, 100, 100, tb))
        self.assertEqual(ingest.cvd_analysis(c)["divergence"], "BEARISH_DIVERGENCE")
        flat = [C(i * TF, 100, 100.5, 99.5, 100, 100, 50) for i in range(60)]
        self.assertEqual(ingest.cvd_analysis(flat)["divergence"], "NONE")
        self.assertEqual(ingest.cvd_analysis(flat[:10])["divergence"], "UNKNOWN")


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0)


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._b, self.headers, self.text = status, body, headers or {}, "x"

    def json(self):
        if self._b is None:
            raise ValueError("no json")
        return self._b


class ClientTests(unittest.TestCase):
    def client(self, responses):
        s = FakeSession(responses)
        return ingest.BinanceClient(session=s, sleep=lambda _: None, min_interval=0, retries=3), s

    def test_retries_then_succeeds(self):
        c, s = self.client([Resp(429, headers={"Retry-After": "1"}), Resp(503), Resp(200, [1])])
        self.assertEqual(c.get("/x"), [1])
        self.assertEqual(s.calls, 3)

    def test_blocked_region_and_client_errors_fail_fast(self):
        c, s = self.client([Resp(451)])
        with self.assertRaisesRegex(ingest.DataError, "451"):
            c.get("/x")
        c, s = self.client([Resp(400)])
        with self.assertRaises(ingest.DataError):
            c.get("/x")
        self.assertEqual(s.calls, 1)

    def test_gives_up_and_bad_json(self):
        c, _ = self.client([Resp(500)] * 4)
        with self.assertRaises(ingest.DataError):
            c.get("/x")
        c, _ = self.client([Resp(200, None)])
        with self.assertRaisesRegex(ingest.DataError, "JSON"):
            c.get("/x")

    def test_klines_drops_forming_candle_and_rejects_garbage(self):
        now = int(time.time() * 1000)
        row = lambda t, closed: [t, "1", "2", "0.5", "1.5", "10", t + 899_999 if closed else now + 60_000, "0", 1, "4", "0", "0"]
        c, _ = self.client([Resp(200, [row(now - 2 * TF, True), row(now - TF, True), row(now, False)])])
        out = c.klines("X", "15m", 3)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["tb"], 4.0)
        c, _ = self.client([Resp(200, [["bad"]])])
        with self.assertRaises(ingest.DataError):
            c.klines("X", "15m", 3)

    def test_open_interest_sorted_and_malformed(self):
        c, _ = self.client([Resp(200, [{"timestamp": 2, "sumOpenInterest": "5"}, {"timestamp": 1, "sumOpenInterest": "4"}])])
        self.assertEqual(c.open_interest_hist("X"), [4.0, 5.0])
        c, _ = self.client([Resp(200, [{"nope": 1}])])
        with self.assertRaises(ingest.DataError):
            c.open_interest_hist("X")


class StateTests(unittest.TestCase):
    def setUp(self):
        self.t = [1_000_000.0]
        self.s = StateManager(":memory:", cooldown_hours=8, flip_lock_hours=24, clock=lambda: self.t[0])
        self.sig = {"symbol": "ETHUSDT", "side": "LONG", "entries": [100, 99, 98, 97], "tp": 105, "sl": 95}

    def test_lock_lifecycle(self):
        self.assertEqual(self.s.lock_status("ETHUSDT")["allowed"], ["LONG", "SHORT"])
        pid = self.s.add_position(self.sig)
        self.assertEqual(self.s.lock_status("ETHUSDT")["reason"], "active position")
        self.assertFalse(self.s.lock_status("SOLUSDT")["locked"])  # other coins unaffected
        self.t[0] += 3600
        self.s.close_position(pid, "TP", self.s.now_ms(), 105, 20.0)
        self.assertIn("cooldown", self.s.lock_status("ETHUSDT")["reason"])
        self.t[0] += 9 * 3600  # past cooldown, inside flip window
        st = self.s.lock_status("ETHUSDT")
        self.assertEqual((st["locked"], st["allowed"], st["bias"]), (False, ["LONG"], "LONG"))
        self.t[0] += 20 * 3600
        self.assertEqual(self.s.lock_status("ETHUSDT")["allowed"], ["LONG", "SHORT"])

    def test_failed_signal_does_not_lock(self):
        pid = self.s.add_position(self.sig)
        self.s.close_position(pid, "FAILED", self.s.now_ms())
        self.assertFalse(self.s.lock_status("ETHUSDT")["locked"])

    def test_close_only_once_and_bad_status(self):
        pid = self.s.add_position(self.sig)
        self.s.close_position(pid, "SL", 5, 95, -50.0)
        self.s.close_position(pid, "TP", 6, 105, 50.0)  # ignored: already closed
        self.assertEqual(self.s.stats(), {"SL": 1})
        with self.assertRaises(ValueError):
            self.s.close_position(pid, "WHATEVER", 1)


class Tracking(unittest.TestCase):
    def pos(self, side="SHORT"):
        e = [101, 102, 103, 104] if side == "SHORT" else [99, 98, 97, 96]
        return {"side": side, "entries": e, "tp": 95 if side == "SHORT" else 105, "sl": 108 if side == "SHORT" else 92,
                "filled": [0, 0, 0, 0], "opened_at": 0, "checked_until": 0}

    def cs(self, *rows):
        return [C(i + 1, 0, h, l, (h + l) / 2) for i, (h, l) in enumerate(rows)]

    def test_take_profit_needs_a_later_candle(self):
        r = advance_position(self.pos(), self.cs((105, 99), (103, 94)), 10)
        self.assertEqual(r["status"], "TP")
        self.assertAlmostEqual(r["ret_pct"], (10 * (102.5 - 95) / 102.5 - 0.007) * 100, places=4)
        same = advance_position(self.pos(), self.cs((105, 94)), 10)  # fills and TP in one candle -> not allowed
        self.assertEqual(same["status"], "OPEN")

    def test_cancel_stop_and_progress(self):
        self.assertEqual(advance_position(self.pos(), self.cs((100, 94)), 10)["status"], "CANCELLED")
        self.assertEqual(advance_position(self.pos(), self.cs((101.5, 99), (109, 100)), 10)["status"], "SL")
        self.assertEqual(advance_position(self.pos(), self.cs((109, 94)), 10)["status"], "SL")  # SL wins ties
        r = advance_position(self.pos(), self.cs((101.5, 99)), 10)
        self.assertEqual((r["status"], r["filled"], r["checked_until"]), ("OPEN", [1, 0, 0, 0], 1))
        again = advance_position({**self.pos(), **r}, self.cs((101.5, 99)), 10)  # same candles are not replayed
        self.assertEqual(again["filled"], [1, 0, 0, 0])

    def test_long_and_expiry(self):
        self.assertEqual(advance_position(self.pos("LONG"), self.cs((98.5, 95), (106, 99)), 10)["status"], "TP")
        r = advance_position(self.pos(), self.cs((101.5, 99)), 49 * H)
        self.assertEqual(r["status"], "EXPIRED")
        self.assertEqual(advance_position(self.pos(), self.cs((100, 99)), 49 * H)["status"], "CANCELLED")


def ctx_fixture(**kw):
    ctx = {"symbol": "ETHUSDT", "price": 100.0, "atr_entry": 1.0, "rvol": 1.5,
           "macro_context": {"correlation_risk": {"long": False, "short": False, "reasons": []}}}
    ctx.update(kw)
    return ctx


LOCK = {"locked": False, "allowed": ["LONG", "SHORT"], "bias": None}


def dec(**kw):
    d = {"action": "OPEN_LONG", "confidence_score": 0.7, "confluence_factors": ["a", "b", "c"], "invalidation_level": 98.0,
         "entry_price": 100.0, "take_profit_levels": [104.0], "risk_reward_ratio": 2.0, "reasoning": "r"}
    d.update(kw)
    return d


class Decisions(unittest.TestCase):
    R = Rules()

    def test_parse(self):
        ok = parse_decision({"action": "OPEN_LONG", "confidence_score": 0.7, "confluence_factors": ["x"],
                             "invalidation_level": 98, "entry_price": 100, "take_profit_levels": [104], "risk_reward_ratio": 2})
        self.assertEqual(ok["take_profit_levels"], [104.0])
        hold = parse_decision({"action": "HOLD", "confidence_score": 0.9, "entry_price": 5})
        self.assertEqual((hold["confidence_score"], hold["entry_price"]), (0.0, 0.0))
        for bad in [None, {"action": "BUY"}, {"action": "OPEN_LONG", "confidence_score": 2, "invalidation_level": 1, "entry_price": 1,
                                              "take_profit_levels": [2]},
                    {"action": "OPEN_LONG", "confidence_score": 0.5, "invalidation_level": 1, "entry_price": 1, "take_profit_levels": []},
                    {"action": "OPEN_LONG", "confidence_score": float("nan"), "invalidation_level": 1, "entry_price": 1,
                     "take_profit_levels": [2]},
                    {"action": "OPEN_LONG", "confidence_score": 0.5, "invalidation_level": 1, "entry_price": 1,
                     "take_profit_levels": [2], "confluence_factors": "oops"}]:
            with self.assertRaises(DecisionError):
                parse_decision(bad)

    def test_rules(self):
        self.assertEqual(enforce_rules(dec(), ctx_fixture(), LOCK, self.R), [])
        self.assertTrue(enforce_rules(dec(), ctx_fixture(rvol=1.0), LOCK, self.R))
        self.assertTrue(enforce_rules(dec(), ctx_fixture(rvol=None), LOCK, self.R))
        risky = ctx_fixture(macro_context={"correlation_risk": {"long": True, "short": False, "reasons": ["BTC bearish"]}})
        self.assertTrue(any("BTC" in b for b in enforce_rules(dec(), risky, LOCK, self.R)))
        self.assertEqual(enforce_rules(dec(action="OPEN_SHORT", invalidation_level=102, take_profit_levels=[96]), risky, LOCK, self.R), [])
        self.assertTrue(enforce_rules(dec(), ctx_fixture(), {**LOCK, "allowed": ["SHORT"]}, self.R))
        self.assertTrue(enforce_rules(dec(invalidation_level=101), ctx_fixture(), LOCK, self.R))       # wrong side
        self.assertTrue(enforce_rules(dec(take_profit_levels=[101]), ctx_fixture(), LOCK, self.R))     # R:R
        self.assertTrue(enforce_rules(dec(invalidation_level=90, take_profit_levels=[130]), ctx_fixture(), LOCK, self.R))  # SL 10%
        self.assertTrue(enforce_rules(dec(entry_price=103), ctx_fixture(), LOCK, self.R))             # far from price
        self.assertTrue(enforce_rules(dec(confidence_score=0.4), ctx_fixture(), LOCK, self.R))
        self.assertTrue(enforce_rules(dec(confluence_factors=["a"]), ctx_fixture(), LOCK, self.R))
        self.assertTrue(enforce_rules(dec(invalidation_level=99.9), ctx_fixture(), LOCK, self.R))     # inside the noise
        self.assertTrue(enforce_rules(dec(action="NO_TRADE"), ctx_fixture(), LOCK, self.R))

    def test_ladder(self):
        for action, sl, tp in (("OPEN_LONG", 98.0, 104.0), ("OPEN_SHORT", 102.0, 96.0)):
            sig = build_signal(dec(action=action, invalidation_level=sl, take_profit_levels=[tp]), ctx_fixture(), self.R)
            e = sig["entries"]
            self.assertEqual(len(e), 4)
            self.assertEqual(len(set(e)), 4)
            if action == "OPEN_LONG":
                self.assertEqual(e, sorted(e, reverse=True))
                self.assertTrue(sig["sl"] < e[-1] and sig["tp"] > e[0])
            else:
                self.assertEqual(e, sorted(e))
                self.assertTrue(sig["sl"] > e[-1] and sig["tp"] < e[0])
        tight = build_signal(dec(invalidation_level=99.5, take_profit_levels=[103]), ctx_fixture(), self.R)
        self.assertGreater(tight["entries"][-1], 99.5 + 0.39 * 0.5)  # last entry stays inside the stop
        self.assertGreater(tight["entries"][-1], tight["sl"])

    def test_merge(self):
        a, b = dec(), dec(entry_price=100.2, invalidation_level=98.2, take_profit_levels=[104.4], confluence_factors=["c", "d"])
        m = merge_decisions({"g": a, "c": b})
        self.assertAlmostEqual(m["entry_price"], 100.1)
        self.assertEqual(m["confluence_factors"], ["a", "b", "c", "d"])
        self.assertEqual(merge_decisions({"g": a, "c": dec(action="OPEN_SHORT")})["action"], "NO_TRADE")
        self.assertEqual(merge_decisions({"g": a, "c": dec(action="HOLD")})["action"], "NO_TRADE")

    def test_payload_shape(self):
        ctx = ctx_fixture(derivatives_positioning={"funding": {"rate": 0.000123456789}}, market_structure={"rvol_20": 1.234567891})
        p = build_payload(ctx, {"allowed": ["LONG"], "bias": "LONG"})
        self.assertEqual(set(p), {"symbol", "current_price", "atr_entry_timeframe", "macro_context", "derivatives_positioning",
                                  "market_structure", "bot_state"})
        self.assertEqual(p["bot_state"], {"allowed_directions": ["LONG"], "current_bias": "LONG"})
        self.assertEqual(p["derivatives_positioning"]["funding"]["rate"], 0.000123457)
        json.loads(build_prompt(p).split("Payload:\n", 1)[1])


# ------------------------------------------------------------------ integration

class FakeClient:
    """Deterministic market: BTC in a clean uptrend, alts random walks with a volume spike on ETH only."""

    def __init__(self):
        self.extra = {}
        self.klines_calls = 0

    def klines(self, symbol, interval, limit):
        self.klines_calls += 1
        tf = ingest.TF_MS[interval]
        rnd = random.Random(f"{symbol}{interval}")
        end = (int(time.time() * 1000) // tf - 1) * tf
        px, out = 100.0, []
        for i in range(limit):
            t = end - (limit - 1 - i) * tf
            step = 0.4 if symbol == "BTCUSDT" else rnd.uniform(-0.6, 0.6)
            o, px = px, max(px + step, 5.0)
            out.append(C(t, o, max(o, px) + 0.3, min(o, px) - 0.3, px, rnd.uniform(90, 110), rnd.uniform(40, 60)))
        if symbol == "ETHUSDT" and interval == "15m":
            out[-1]["v"] *= 3
        return out + self.extra.get((symbol, interval), [])

    def universe(self, symbols, max_coins=0, min_volume=0):
        return ["ETHUSDT", "SOLUSDT"]

    def premium_index(self):
        return {"ETHUSDT": {"rate": 0.0001, "mark": 100.0}, "SOLUSDT": {"rate": -0.0004, "mark": 100.0}}

    def funding_intervals(self):
        return {"SOLUSDT": 4}

    def open_interest_hist(self, symbol, period="15m", limit=30):
        return [1000.0 * (1 + 0.002 * i) for i in range(30)]


def fake_ai(action="OPEN_LONG", calls=None, tp_mult=4.0, **over):
    def ask(prompt, system=None, max_tokens=None):
        if calls is not None:
            calls.append(1)
        payload = json.loads(prompt.split("Payload:\n", 1)[1])
        p, a = payload["current_price"], payload["atr_entry_timeframe"]
        long = action == "OPEN_LONG"
        d = {"action": action, "confidence_score": 0.75, "confluence_factors": ["4h bullish", "RVOL spike", "value area reclaim"],
             "invalidation_level": p - 2 * a if long else p + 2 * a, "entry_price": p,
             "take_profit_levels": [p + tp_mult * a if long else p - tp_mult * a], "risk_reward_ratio": 2.0, "reasoning": "test"}
        d.update(over)
        return d
    return ask


class Integration(unittest.TestCase):
    def build(self, ais, poster=None):
        cfg = execution.Config(symbols=("ALL",), coin_delay=0)
        self.client = FakeClient()
        self.state = StateManager(":memory:", cooldown_hours=8, flip_lock_hours=24)
        self.posted = [] if poster is None else poster
        return execution.ExecutionManager(cfg, self.state, self.client, ais=ais,
                                          poster=self.posted.append if isinstance(self.posted, list) else self.posted,
                                          sleep=lambda _: None)

    def test_full_lifecycle(self):
        calls = []
        mgr = self.build({"model_a": fake_ai(calls=calls), "model_b": fake_ai(calls=calls)})
        tally = mgr.run_cycle()
        self.assertEqual(len(self.posted), 1, tally)                 # ETH only; SOL fails the RVOL pre-gate
        self.assertEqual(len(calls), 2)                              # one call per AI, none for SOL
        msg = self.posted[0]
        self.assertTrue(msg.startswith("#ETH/USDT\nLong\nLeverage: Cross (10x)"))
        self.assertEqual(msg.count("Take-Profit Targets:\n1)"), 1)
        self.assertEqual(sum(1 for l in msg.splitlines() if l[:2] in ("1)", "2)", "3)", "4)")), 4 + 1 + 1)
        pos = self.state.open_positions("ETHUSDT")[0]
        # second cycle: coin is locked, no more LLM calls and no new post
        mgr.run_cycle()
        self.assertEqual((len(self.posted), len(calls)), (1, 2))
        # price walks through the ladder, then hits the target
        e, tp, t0 = pos["entries"], pos["tp"], int(time.time() * 1000) + 10_000
        self.client.extra[("ETHUSDT", "15m")] = [C(t0, e[0], e[0], e[-1] - 0.01, e[-1]), C(t0 + TF, e[-1], tp + 1, e[-1], tp)]
        mgr.reconcile()
        self.assertEqual(self.state.stats().get("TP"), 1)
        self.assertIn("cooldown", self.state.lock_status("ETHUSDT")["reason"])
        mgr.run_cycle()
        self.assertEqual(len(self.posted), 1)                         # still cooling down

    def test_ai_disagreement_posts_nothing(self):
        mgr = self.build({"model_a": fake_ai("OPEN_LONG"), "model_b": fake_ai("OPEN_SHORT")})
        mgr.run_cycle()
        self.assertEqual(self.posted, [])
        self.assertEqual(self.state.stats(), {})

    def test_rules_reject_low_rr_and_bad_reply(self):
        mgr = self.build({"model_a": fake_ai(tp_mult=0.4, risk_reward_ratio=3)})  # claims 3.0 but really 0.2
        mgr.run_cycle()
        self.assertEqual(self.posted, [])
        mgr = self.build({"model_a": lambda prompt, system=None, max_tokens=None: {"action": "YOLO"}})
        mgr.run_cycle()
        self.assertEqual(self.posted, [])
        self.assertEqual(self.state.stats(), {})

    def test_ai_outage_does_not_stop_cycle(self):
        def boom(prompt, system=None, max_tokens=None):
            raise TimeoutError("down")
        mgr = self.build({"model_a": boom})
        mgr.run_cycle()
        self.assertEqual(self.posted, [])

    def test_post_failure_voids_position(self):
        def bad_post(_):
            raise RuntimeError("telegram down")
        mgr = self.build({"model_a": fake_ai()}, poster=bad_post)
        mgr.run_cycle()
        self.assertEqual(self.state.stats(), {"FAILED": 1})
        self.assertFalse(self.state.lock_status("ETHUSDT")["locked"])  # can retry next cycle

    def test_btc_clash_blocks_altcoin_short(self):
        mgr = self.build({"model_a": fake_ai("OPEN_SHORT")})  # BTC fixture is a bullish expansion
        mgr.run_cycle()
        self.assertEqual(self.posted, [])
        rejected = [r["detail"] for r in self.state.db.execute("SELECT detail FROM events WHERE kind='REJECTED'")]
        self.assertTrue(any("BTC" in d for d in rejected), rejected)

    def test_degraded_shared_context(self):
        class NoBtc(FakeClient):
            def klines(self, symbol, interval, limit):
                if symbol == "BTCUSDT":
                    raise ingest.DataError("down")
                return super().klines(symbol, interval, limit)
        shared = ingest.build_shared(NoBtc())
        self.assertIsNone(shared["btc"])
        self.assertFalse(ingest.correlation_risk("ETHUSDT", shared["btc"])["known"])


if __name__ == "__main__":
    unittest.main()
