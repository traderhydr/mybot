"""Persistent state for the v2 engine: signal-level positions, exits and cooldowns (SQLite).

A "position" is a posted signal that Cornix is (presumably) trading. The bot never talks to the exchange, so it
follows each signal on candle data: entries fill when price touches them, then TP / SL / expiry end it. While a
position is open, or a coin is cooling down, the LLM is not called for that coin.
"""
import json
import logging
import sqlite3
import threading
import time

log = logging.getLogger("state")

OPEN = "OPEN"
TERMINAL = ("TP", "SL", "EXPIRED", "CANCELLED", "FAILED")
HOUR_MS = 3_600_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, side TEXT NOT NULL,
    entries TEXT NOT NULL, tp REAL NOT NULL, sl REAL NOT NULL,
    filled TEXT NOT NULL DEFAULT '[0,0,0,0]',
    status TEXT NOT NULL DEFAULT 'OPEN',
    opened_at INTEGER NOT NULL, checked_until INTEGER NOT NULL,
    closed_at INTEGER, exit_price REAL, ret_pct REAL, meta TEXT
);
CREATE INDEX IF NOT EXISTS idx_pos_symbol ON positions(symbol, status);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, symbol TEXT, kind TEXT NOT NULL, detail TEXT
);
"""


class StateManager:
    def __init__(self, path="bot_state.db", cooldown_hours=8.0, flip_lock_hours=24.0, clock=time.time):
        self.cooldown_ms = int(cooldown_hours * HOUR_MS)
        self.flip_ms = int(flip_lock_hours * HOUR_MS)
        self.clock = clock
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        with self._lock, self.db:
            self.db.executescript(SCHEMA)

    def now_ms(self):
        return int(self.clock() * 1000)

    # -- writes

    def add_position(self, sig, now_ms=None):
        now_ms = now_ms or self.now_ms()
        with self._lock, self.db:
            cur = self.db.execute(
                "INSERT INTO positions(symbol, side, entries, tp, sl, opened_at, checked_until, meta) VALUES(?,?,?,?,?,?,?,?)",
                (sig["symbol"], sig["side"], json.dumps(sig["entries"]), sig["tp"], sig["sl"], now_ms, now_ms,
                 json.dumps({k: sig.get(k) for k in ("confidence", "rr", "reason")})))
            return cur.lastrowid

    def update_progress(self, pos_id, filled, checked_until):
        with self._lock, self.db:
            self.db.execute("UPDATE positions SET filled=?, checked_until=? WHERE id=? AND status='OPEN'",
                            (json.dumps(filled), checked_until, pos_id))

    def close_position(self, pos_id, status, closed_at, exit_price=None, ret_pct=None, filled=None):
        if status not in TERMINAL:
            raise ValueError(f"bad status {status}")
        with self._lock, self.db:
            self.db.execute(
                "UPDATE positions SET status=?, closed_at=?, exit_price=?, ret_pct=?, filled=COALESCE(?, filled) "
                "WHERE id=? AND status='OPEN'",
                (status, closed_at, exit_price, ret_pct, json.dumps(filled) if filled else None, pos_id))

    def log_event(self, kind, symbol=None, detail=None):
        with self._lock, self.db:
            self.db.execute("INSERT INTO events(ts, symbol, kind, detail) VALUES(?,?,?,?)",
                            (self.now_ms(), symbol, kind, detail if detail is None or isinstance(detail, str) else json.dumps(detail)))

    # -- reads

    @staticmethod
    def _row(r):
        if r is None:
            return None
        d = dict(r)
        d["entries"], d["filled"] = json.loads(d["entries"]), json.loads(d["filled"])
        return d

    def open_positions(self, symbol=None):
        q, args = "SELECT * FROM positions WHERE status='OPEN'", ()
        if symbol:
            q, args = q + " AND symbol=?", (symbol,)
        with self._lock:
            return [self._row(r) for r in self.db.execute(q + " ORDER BY id", args)]

    def last_position(self, symbol):
        with self._lock:
            return self._row(self.db.execute(
                "SELECT * FROM positions WHERE symbol=? AND status != 'FAILED' ORDER BY id DESC LIMIT 1", (symbol,)).fetchone())

    def stats(self):
        with self._lock:
            return {r["status"]: r["n"] for r in self.db.execute("SELECT status, COUNT(*) n FROM positions GROUP BY status")}

    def lock_status(self, symbol, now_ms=None):
        """Decide whether the LLM may be called for `symbol`, and which directions it may trade.

        - open position           -> locked
        - within cooldown of the last signal's exit/open -> locked
        - within the flip window  -> unlocked but only the last signal's direction is allowed (no whipsaw)
        """
        now_ms = now_ms or self.now_ms()
        if self.open_positions(symbol):
            return {"locked": True, "reason": "active position", "allowed": [], "bias": None}
        last = self.last_position(symbol)
        if not last:
            return {"locked": False, "reason": None, "allowed": ["LONG", "SHORT"], "bias": None}
        ref = last["closed_at"] or last["opened_at"]
        if now_ms - ref < self.cooldown_ms:
            left = (self.cooldown_ms - (now_ms - ref)) / HOUR_MS
            return {"locked": True, "reason": f"cooldown ({left:.1f}h left)", "allowed": [], "bias": last["side"]}
        if now_ms - ref < self.flip_ms:
            return {"locked": False, "reason": None, "allowed": [last["side"]], "bias": last["side"]}
        return {"locked": False, "reason": None, "allowed": ["LONG", "SHORT"], "bias": None}


def advance_position(pos, candles, now_ms, expiry_hours=48.0, fee_rt=0.0007, leverage=10):
    """Replay closed candles newer than `checked_until` against an open position (pure function, no I/O).

    Same rules as backtest.py: entries are limit orders filled on touch; stop-loss wins ties; a trade cannot fill and
    take profit in the same candle; TP touched before any fill cancels the signal; expiry closes at market.
    Returns a dict: {"status": "OPEN"|terminal, "filled", "checked_until", and for closes: closed_at, exit_price, ret_pct}.
    """
    short = pos["side"] == "SHORT"
    entries, tp, sl = pos["entries"], pos["tp"], pos["sl"]
    filled = list(pos["filled"])
    checked = pos["checked_until"]
    last_close = None
    for c in candles:
        if c["t"] <= checked:
            continue
        checked = c["t"]
        last_close = c["c"]
        new = False
        for k, e in enumerate(entries):
            if not filled[k] and ((c["h"] >= e) if short else (c["l"] <= e)):
                filled[k], new = 1, True
        n = sum(filled)
        tp_hit = c["l"] <= tp if short else c["h"] >= tp
        sl_hit = c["h"] >= sl if short else c["l"] <= sl
        end = c["t"] + 1
        if n == 0:
            if tp_hit:
                return {"status": "CANCELLED", "filled": filled, "checked_until": checked, "closed_at": end,
                        "exit_price": None, "ret_pct": 0.0}
            continue
        if sl_hit:
            return _close("SL", pos, filled, checked, end, sl, fee_rt, leverage)
        if tp_hit and not new:
            return _close("TP", pos, filled, checked, end, tp, fee_rt, leverage)
    if now_ms - pos["opened_at"] >= expiry_hours * HOUR_MS:
        if sum(filled) == 0:
            return {"status": "CANCELLED", "filled": filled, "checked_until": checked, "closed_at": now_ms,
                    "exit_price": None, "ret_pct": 0.0}
        return _close("EXPIRED", pos, filled, checked, now_ms, last_close if last_close is not None else entries[0],
                      fee_rt, leverage)
    return {"status": "OPEN", "filled": filled, "checked_until": checked}


def _close(status, pos, filled, checked, closed_at, exit_price, fee_rt, leverage):
    px = [e for e, f in zip(pos["entries"], filled) if f]
    avg = sum(px) / len(px)
    move = (avg - exit_price) / avg if pos["side"] == "SHORT" else (exit_price - avg) / avg
    return {"status": status, "filled": filled, "checked_until": checked, "closed_at": closed_at,
            "exit_price": exit_price, "ret_pct": (leverage * move - leverage * fee_rt) * 100}
