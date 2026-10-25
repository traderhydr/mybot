"""v2 execution manager: one scan cycle = reconcile exits -> lock check -> ingest -> LLM -> rules -> post -> record.

Run with `python bot.py` and STRATEGY=v2 (or `python execution.py`). Flags: --dry (no posting, in-memory state),
--once, --limit N.
"""
import argparse
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import ingest
import llm
import notify
from decision import (DecisionError, Rules, SYSTEM_INSTRUCTIONS, build_payload, build_prompt, build_signal,
                      enforce_rules, merge_decisions, parse_decision)
from state import StateManager, advance_position

log = logging.getLogger("exec")


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        raise SystemExit(f"{name} must be a number")


@dataclass
class Config:
    symbols: tuple = ("ALL",)
    max_coins: int = 0
    min_volume: float = 0.0
    entry_tf: str = "15m"
    interval_min: float = 60.0
    coin_delay: float = 2.0
    cooldown_hours: float = 8.0
    flip_lock_hours: float = 24.0
    expiry_hours: float = 48.0
    pregate_rvol: bool = True
    state_db: str = "bot_state.db"
    rules: Rules = field(default_factory=Rules)

    @classmethod
    def from_env(cls):
        tf = os.environ.get("ENTRY_TIMEFRAME", "15m")
        if tf not in ingest.TF_MS:
            raise SystemExit(f"ENTRY_TIMEFRAME must be one of {sorted(ingest.TF_MS)}")
        symbols = tuple(s.strip().upper() for s in os.environ.get("SYMBOLS", "ALL").split(",") if s.strip()) or ("ALL",)
        return cls(
            symbols=symbols, max_coins=int(_f("MAX_COINS", 0)), min_volume=_f("MIN_24H_VOLUME_USDT", 0),
            entry_tf=tf, interval_min=_f("INTERVAL_MINUTES", 60), coin_delay=_f("COIN_DELAY_SECONDS", 2),
            cooldown_hours=_f("COOLDOWN_HOURS", 8), flip_lock_hours=_f("FLIP_LOCK_HOURS", 24),
            expiry_hours=_f("EXPIRY_HOURS", 48), pregate_rvol=os.environ.get("PREGATE_RVOL", "1") != "0",
            state_db=os.environ.get("STATE_DB", "bot_state.db"),
            rules=Rules(rvol_min=_f("RVOL_MIN", 1.2), min_confidence=_f("MIN_CONFIDENCE", 0.6), min_rr=_f("MIN_RR", 1.5),
                        min_rr_ladder=_f("MIN_RR_LADDER", 1.2), max_sl_pct=_f("MAX_SL_PCT", 0.05),
                        ladder_step_atr=_f("LADDER_STEP_ATR", 0.2)))


class ExecutionManager:
    def __init__(self, cfg, state, client, ais=None, poster=None, sleep=None):
        self.cfg, self.state, self.client = cfg, state, client
        self.ais = ais if ais is not None else llm.enabled_ais()
        self.poster = poster  # None = dry run
        self._stop_evt = threading.Event()
        self._sleep = sleep or self._stop_evt.wait  # interruptible by SIGTERM
        if not self.ais:
            raise SystemExit("Set ANTHROPIC_API_KEY")

    def stop(self, *_):
        self._stop_evt.set()

    # ---------------------------------------------------------------- exits

    def reconcile(self):
        """Follow every open signal on fresh candles and close it on TP / SL / cancel / expiry."""
        for pos in self.state.open_positions():
            try:
                candles = self.client.klines(pos["symbol"], self.cfg.entry_tf, 500)
                res = advance_position(pos, candles, self.state.now_ms(), self.cfg.expiry_hours)
            except (ingest.DataError, ValueError) as ex:
                log.warning("reconcile %s #%s failed: %s", pos["symbol"], pos["id"], ex)
                continue
            if res["status"] == "OPEN":
                self.state.update_progress(pos["id"], res["filled"], res["checked_until"])
                continue
            self.state.close_position(pos["id"], res["status"], res["closed_at"], res["exit_price"], res["ret_pct"], res["filled"])
            self.state.log_event("EXIT", pos["symbol"], f"#{pos['id']} {pos['side']} -> {res['status']} {res['ret_pct']:+.1f}%")
            log.info("%s #%s %s closed: %s (%+.1f%% of margin)", pos["symbol"], pos["id"], pos["side"], res["status"], res["ret_pct"])

    # ---------------------------------------------------------------- decisions

    def _ask(self, ctx, lock):
        payload = build_payload(ctx, lock)
        prompt = build_prompt(payload)
        decisions, errors = {}, []
        with ThreadPoolExecutor(len(self.ais)) as pool:  # ask every AI at the same time
            futs = {name: pool.submit(fn, prompt, system=SYSTEM_INSTRUCTIONS, max_tokens=1200) for name, fn in self.ais.items()}
            for name, fut in futs.items():
                try:
                    decisions[name] = parse_decision(fut.result())
                except (DecisionError, ValueError, KeyError) as ex:
                    errors.append(f"{name}: bad reply ({ex})")
                except Exception as ex:  # network / quota: this coin is skipped, the cycle goes on
                    errors.append(f"{name}: {type(ex).__name__}: {ex}")
        if errors or not decisions:
            raise RuntimeError("; ".join(errors) or "no decision")  # require every configured AI, like v1 consensus
        return merge_decisions(decisions), decisions

    def process_symbol(self, symbol, shared):
        """Returns a short outcome string for the cycle summary."""
        lock = self.state.lock_status(symbol)
        if lock["locked"]:
            return f"locked: {lock['reason']}"
        ctx = ingest.build_context(symbol, self.client, shared, self.cfg.entry_tf)
        rv = ctx["rvol"]
        if self.cfg.pregate_rvol and (rv is None or rv < self.cfg.rules.rvol_min):
            return f"skip: RVOL {'n/a' if rv is None else round(rv, 2)} < {self.cfg.rules.rvol_min} (no LLM call)"
        decision, _ = self._ask(ctx, lock)
        if decision["action"] in ("HOLD", "NO_TRADE"):
            self.state.log_event(decision["action"], symbol, decision["reasoning"])
            return f"{decision['action']}: {decision['reasoning'][:90]}"
        bad = enforce_rules(decision, ctx, lock, self.cfg.rules)
        if bad:
            self.state.log_event("REJECTED", symbol, "; ".join(bad))
            return "rejected: " + "; ".join(bad)
        sig = build_signal(decision, ctx, self.cfg.rules)
        if sig["rr"] < self.cfg.rules.min_rr_ladder:
            self.state.log_event("REJECTED", symbol, f"ladder R:R {sig['rr']:.2f}")
            return f"rejected: ladder R:R {sig['rr']:.2f} < {self.cfg.rules.min_rr_ladder}"
        msg = notify.format_message(sig)
        log.info("SIGNAL %s\n%s\n(confidence %.2f, R:R %.2f) %s", symbol, msg, sig["confidence"], sig["rr"], sig["reason"])
        pos_id = self.state.add_position(sig)  # record first: a crash after posting can't cause a duplicate signal
        if self.poster:
            try:
                self.poster(msg)
            except Exception as ex:
                self.state.close_position(pos_id, "FAILED", self.state.now_ms())
                raise RuntimeError(f"post failed, position #{pos_id} voided: {ex}") from ex
        return f"SIGNAL {sig['side']} (conf {sig['confidence']:.2f}, R:R {sig['rr']:.2f})"

    def run_cycle(self, limit=None):
        started = time.time()
        shared = ingest.build_shared(self.client)
        self.reconcile()
        try:
            coins = self.client.universe(list(self.cfg.symbols), self.cfg.max_coins, self.cfg.min_volume)
        except ingest.DataError as ex:
            log.error("cannot load the coin list: %s", ex)
            return {}
        if limit:
            coins = coins[:limit]
        log.info("cycle: %d coins | BTC 4h: %s | open positions: %d", len(coins),
                 (shared["btc"] or {}).get("state", "unknown"), len(self.state.open_positions()))
        tally = {}
        for i, coin in enumerate(coins, 1):
            if self._stop_evt.is_set():
                break
            try:
                outcome = self.process_symbol(coin, shared)
            except ingest.DataError as ex:
                outcome = f"data error: {ex}"
            except RuntimeError as ex:  # expected: AI outage / bad reply / Telegram failure
                outcome = f"error: {ex}"
            except Exception as ex:  # unexpected: keep the traceback, but one bad coin never stops the cycle
                log.exception("%s failed unexpectedly", coin)
                outcome = f"error: {type(ex).__name__}: {ex}"
            key = outcome.split(":")[0].split(" ")[0]
            tally[key] = tally.get(key, 0) + 1
            log.info("[%d/%d] %s -> %s", i, len(coins), coin, outcome)
            self._sleep(self.cfg.coin_delay)
        log.info("cycle done in %.0fs: %s | positions: %s", time.time() - started, tally, self.state.stats())
        return tally


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="no posting, in-memory state")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    cfg = Config.from_env()
    if not a.dry and not (os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHANNEL_ID")):
        sys.exit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID (see .env.example)")
    state = StateManager(":memory:" if a.dry else cfg.state_db, cfg.cooldown_hours, cfg.flip_lock_hours)
    mgr = ExecutionManager(cfg, state, ingest.BinanceClient(), poster=None if a.dry else notify.post)
    log.info("v2 engine | AIs: %s | entry TF %s | dry=%s", ", ".join(mgr.ais), cfg.entry_tf, a.dry)
    signal.signal(signal.SIGTERM, mgr.stop)
    signal.signal(signal.SIGINT, mgr.stop)
    limit = a.limit if a.limit else (3 if a.dry else None)
    while not mgr._stop_evt.is_set():
        started = time.time()
        try:
            mgr.run_cycle(limit)
        except Exception:
            log.exception("cycle crashed; continuing")
        if a.once or a.dry:
            break
        mgr._stop_evt.wait(max(0.0, cfg.interval_min * 60 - (time.time() - started)))  # next cycle, or exit on SIGTERM


if __name__ == "__main__":
    main()
