"""Tests for backtest_v2 (python -m unittest test_bt2). No network."""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import backtest_v2 as b2
import execution
import ingest
from decision import Rules

TF = 900_000
H = 3_600_000


def C(t, o, h, l, c, v=100.0):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v, "tb": v / 2}


class AsOf(unittest.TestCase):
    def setUp(self):
        self.h = b2.History()
        self.h.add_candles("X", "15m", [C(i * TF, 1, 2, 0.5, 1.5) for i in range(10)])
        self.h.oi["X"] = [(i * TF + TF, 100.0 + i) for i in range(10)]   # point i usable from (i+1)*TF
        self.c = b2.HistoricalClient(self.h)

    def test_only_closed_candles_are_visible(self):
        self.c.T = 5 * TF                       # candle 4 closed exactly now; candle 5 has just opened
        self.assertEqual([r["t"] for r in self.c.klines("X", "15m", 100)], [i * TF for i in range(5)])
        self.c.T = 5 * TF - 1                   # one ms earlier candle 4 is still forming
        self.assertEqual(self.c.klines("X", "15m", 100)[-1]["t"], 3 * TF)
        self.assertEqual(len(self.c.klines("X", "15m", 2)), 2)
        with self.assertRaises(ingest.DataError):
            self.c.klines("Y", "15m", 5)

    def test_open_interest_has_no_lookahead(self):
        self.c.T = 3 * TF
        self.assertEqual(self.c.open_interest_hist("X", limit=30), [100.0, 101.0, 102.0])  # points 0..2 usable
        self.assertEqual(self.c.open_interest_hist("X", limit=2), [101.0, 102.0])
        self.assertEqual(self.c.open_interest_hist("none"), [])

    def test_funding_as_of_and_interval(self):
        self.h.funding["X"] = [(i * 8 * H, 0.0001 * (i + 1)) for i in range(5)]
        self.h.add_candles("BTCUSDT", "1h", [C(i * H, 1, 2, 1, 1.5) for i in range(400)])
        self.h.add_candles("BTCUSDT", "4h", [C(i * 4 * H, 1, 2, 1, 1.5) for i in range(400)])
        b2._shared_cache.clear()
        sh = b2.shared_at(self.h, 20 * H, ["X"])
        self.assertAlmostEqual(sh["premium"]["X"]["rate"], 0.0003)  # funding at 16h applies, the 24h one does not yet
        self.assertEqual(sh["intervals"]["X"], 8)
        b2._shared_cache.clear()
        self.assertEqual(b2.shared_at(self.h, 0, ["X"])["premium"]["X"]["rate"], 0.0001)


class FakeClient:
    def __init__(self, total, page=1000):
        self.total, self.page, self.calls = total, page, []

    def get(self, path, params=None):
        self.calls.append(dict(params))
        i0 = -(-params["startTime"] // TF)  # Binance: candles with open time >= startTime
        n = min(params["limit"], self.total - i0)
        return [[(i0 + k) * TF, "1", "2", "0.5", "1.5", "10", (i0 + k) * TF + TF - 1, "0", 1, "4", "0", "0"]
                for k in range(max(n, 0))]


class Fetch(unittest.TestCase):
    def test_kline_pagination_covers_the_range_without_gaps(self):
        c = FakeClient(2500)
        rows = b2.fetch_klines(c, "X", "15m", 0, 2500 * TF)
        self.assertEqual([r["t"] for r in rows], [i * TF for i in range(2500)])
        self.assertEqual(len(c.calls), 3)

    def test_timestamps_sent_to_binance_are_integers(self):
        c = FakeClient(10)
        b2.fetch_klines(c, "X", "15m", 0.0, 10.0 * TF)       # floats in (e.g. from an hours*ms product)
        self.assertTrue(all(isinstance(p["startTime"], int) and isinstance(p["endTime"], int) for p in c.calls))

    def test_duplicate_pages_are_deduplicated(self):
        class Dup(FakeClient):
            def get(self, path, params=None):
                rows = super().get(path, params)
                return rows[:1] * 2 + rows if params["startTime"] > 0 and len(rows) == 1000 else rows
        rows = b2.fetch_klines(Dup(2500), "X", "15m", 0, 2500 * TF)
        ts = [r["t"] for r in rows]
        self.assertEqual(ts, sorted(set(ts)))

    def test_empty_and_short_responses_stop(self):
        self.assertEqual(b2.fetch_klines(FakeClient(0), "X", "15m", 0, 10 * TF), [])
        c = FakeClient(30)
        self.assertEqual(len(b2.fetch_klines(c, "X", "15m", 0, 100 * TF)), 30)
        self.assertEqual(len(c.calls), 1)

    def test_open_interest_availability_is_delayed_one_period(self):
        class OI:
            def get(self, path, params=None):
                return [{"timestamp": 1000, "sumOpenInterest": "5"}, {"timestamp": 500, "sumOpenInterest": "4"}]
        self.assertEqual(b2.fetch_oi(OI(), "X", 0, 10 ** 6), [(500 + 900_000, 4.0), (1000 + 900_000, 5.0)])


def ctx(price=100.0, rvol=1.5, htf="BULLISH", ltf="BULLISH"):
    return {"symbol": "X", "price": price, "atr_entry": 1.0, "rvol": rvol,
            "macro_context": {"correlation_risk": {"long": False, "short": False, "reasons": []}, "btc": None, "market_regime": "X"},
            "derivatives_positioning": {},
            "market_structure": {"htf_bias": {"daily": htf, "4h": htf}, "ltf_structure": {"bias": ltf}, "rvol_20": rvol}}


ARGS = SimpleNamespace(fee=0.0007, fx_tp=1.5, fx_sl=1.6)


class Timeline(unittest.TestCase):
    def setUp(self):
        self.cfg = execution.Config(entry_tf="15m", cooldown_hours=8, flip_lock_hours=24, expiry_hours=48, rules=Rules())
        self.hist = b2.History()
        T0 = 1000 * TF
        # candle at T0 fills the whole ladder (low 99.3), the next one reaches the 103 target
        fwd = [C(T0, 100, 100.5, 99.3, 100), C(T0 + TF, 100, 103.5, 99.9, 103)] + [C(T0 + (2 + i) * TF, 103, 103.2, 102.8, 103) for i in range(300)]
        self.hist.add_candles("X", "15m", fwd)
        self.T0 = T0

    def test_lock_cooldown_and_flip_window(self):
        T0 = self.T0
        times = [T0, T0 + TF, T0 + 2 * TF, T0 + 5 * H, T0 + 9 * H]
        prepared = [(T, ctx(), None) for T in times]
        rows, funnel, _ = b2.run_coin("X", prepared, self.hist, self.cfg, "rule", ARGS)
        # 1st: signal; 2nd: position still open; 3rd: just closed -> cooldown; 4th: cooldown; 5th: 9h later -> allowed again
        self.assertEqual((funnel["signals"], funnel["locked"], funnel["points"]), (2, 3, 5))
        first = rows["signal"][0]
        self.assertEqual((first["outcome"], first["fills"]), ("tp", 4))
        self.assertGreater(first["ret_alloc"], 0)
        self.assertEqual(len(rows["mirror"]), 2)
        self.assertEqual({v: len(r) for v, r in rows.items()}, {v: 2 for v in b2.bt1.LABELS})

    def test_gates_and_funnel(self):
        T0 = self.T0
        prepared = [(T0, ctx(rvol=0.5), None), (T0 + TF, None, "data: x"), (T0 + 2 * TF, ctx(htf="RANGE", ltf="RANGE"), None)]
        rows, funnel, rejects = b2.run_coin("X", prepared, self.hist, self.cfg, "rule", ARGS)
        self.assertEqual((funnel["rvol_gate"], funnel["data_error"], funnel["NO_TRADE"], funnel["signals"]), (1, 1, 1, 0))
        # BTC clash: rule says LONG, correlation forbids it -> rejected, never traded
        bad = ctx()
        bad["macro_context"]["correlation_risk"] = {"long": True, "short": False, "reasons": ["BTC bearish"]}
        rows, funnel, rejects = b2.run_coin("X", [(T0, bad, None)], self.hist, self.cfg, "rule", ARGS)
        self.assertEqual((funnel["rejected"], funnel["signals"]), (1, 0))
        self.assertIn("BTC bearish", list(rejects)[0])


class ClaudeCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.old = (b2.CACHE_FILE, dict(b2._cache))
        b2.CACHE_FILE = os.path.join(self.tmp, "c.jsonl")
        b2._cache.clear()

    def tearDown(self):
        b2.CACHE_FILE = self.old[0]
        b2._cache.clear()
        b2._cache.update(self.old[1])

    def test_cached_after_first_call(self):
        from unittest import mock
        reply = {"action": "NO_TRADE", "reasoning": "x"}
        with mock.patch("llm.ask_claude", return_value=reply) as m:
            d1 = b2.claude_decision("sys", "prompt")
            d2 = b2.claude_decision("sys", "prompt")
        self.assertEqual((m.call_count, d1, d2), (1, d2, d1))
        self.assertEqual(d1["action"], "NO_TRADE")
        self.assertEqual(len(open(b2.CACHE_FILE).read().splitlines()), 1)
        b2._cache.clear()
        b2.load_cache()                          # survives a restart
        with mock.patch("llm.ask_claude", side_effect=AssertionError("must not call")):
            self.assertEqual(b2.claude_decision("sys", "prompt")["action"], "NO_TRADE")
        self.assertNotEqual(b2.cache_key("sys", "a"), b2.cache_key("sys", "b"))

    def test_bad_reply_cached_transport_error_not(self):
        from unittest import mock
        with mock.patch("llm.ask_claude", return_value={"action": "YOLO"}):
            self.assertIn("_error", b2.claude_decision("s", "p1"))
        with mock.patch("llm.ask_claude", side_effect=TimeoutError("down")):
            with self.assertRaises(TimeoutError):
                b2.claude_decision("s", "p2")
        self.assertEqual(len(b2._cache), 1)


if __name__ == "__main__":
    unittest.main()
