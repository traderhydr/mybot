"""v2 prompt builder, decision schema, hard rule gates, consensus and Cornix ladder construction.

The model proposes a *direction, entry reference, invalidation and target*. Code then (1) validates the schema,
(2) re-checks every hard rule itself (models ignore instructions), and (3) builds the 4-entry ladder the Cornix
format requires from the entry reference and ATR, so the model never has to produce ladder levels.
"""
import json
import math
from dataclasses import dataclass

ACTIONS = ("OPEN_LONG", "OPEN_SHORT", "HOLD", "NO_TRADE")


class DecisionError(ValueError):
    """The model's reply does not match the required schema."""


@dataclass
class Rules:
    rvol_min: float = 1.2
    min_confidence: float = 0.6
    min_rr: float = 1.5               # on the reference entry
    min_rr_ladder: float = 1.2        # on the average of the 4 ladder entries (what a filled position really gets)
    min_factors: int = 2
    max_entry_dev_atr: float = 0.75   # reference entry must be near the current price
    min_sl_atr: float = 0.3           # stop closer than this is noise
    max_sl_pct: float = 0.05          # 5% price move = 50% of margin at 10x
    ladder_step_atr: float = 0.2


_INSTRUCTIONS_TEMPLATE = """You are a disciplined crypto-futures analyst. You receive ONE JSON payload describing a single
perpetual contract and must decide whether to open a trade now. Reply with a single JSON object and nothing else.

OUTPUT (strict JSON, exactly these keys):
{"action": "OPEN_LONG" | "OPEN_SHORT" | "HOLD" | "NO_TRADE",
 "confidence_score": number 0-1,
 "confluence_factors": [string, ...],
 "invalidation_level": number,          // stop-loss price: where the idea is proven wrong
 "entry_price": number,                 // reference entry near the current price
 "take_profit_levels": [number, ...],   // nearest first; the first one is the target that will be used
 "risk_reward_ratio": number,           // |first target - entry| / |entry - invalidation|
 "reasoning": string                    // <= 400 characters
}
For HOLD / NO_TRADE set confidence_score, prices and risk_reward_ratio to 0 and take_profit_levels to [].
HOLD = a valid setup is forming but its trigger has not printed yet. NO_TRADE = nothing to do.

HARD RULES (any violation means NO_TRADE; the system re-checks these and discards violations):
1. Volume confirmation: if market_structure.rvol_20 is null or below @RVOL_MIN@, you must answer NO_TRADE.
2. Only trade a direction listed in bot_state.allowed_directions. If it is empty, answer NO_TRADE.
3. macro_context.correlation_risk.long == true forbids OPEN_LONG; correlation_risk.short == true forbids OPEN_SHORT.
   Never open an altcoin long while BTC's higher-timeframe state is a bearish rejection or bearish expansion.
4. Require at least @MIN_FACTORS@ independent confluence factors from different groups (higher-timeframe bias, key zone,
   lower-timeframe trigger, volume, derivatives positioning). Do not count the same signal twice.
5. Take-profit and invalidation must be on the correct sides of entry_price, with risk_reward_ratio >= @MIN_RR@.
   Place the invalidation beyond a real structure level (swing, FVG edge, value-area edge), not at a random distance.

HOW TO READ THE DATA:
- htf_bias is the dominant direction; trade with it unless a CHoCH on the 4h with RVOL confirmation says otherwise.
- ltf_entry_triggers say what has actually happened on the entry timeframe. No trigger -> HOLD, not a trade.
- Value area (VAL / POC / VAH), unmitigated FVGs and liquidity pools are where price is likely to react or run stops.
- Derivatives: AGGRESSIVE_*_BUILD = new positions entering (trend fuel); SHORT_SQUEEZE_COVERING / LONG_LIQUIDATION =
  positions closing (move may be exhausted). Extreme funding on the side you would join = crowded; be skeptical.
  A CVD divergence against your direction is a warning. Null fields mean unknown, not neutral.
- Do not invent data that is not in the payload. If the evidence is mixed, answer NO_TRADE: skipping is a good outcome.
"""


def system_instructions(rules):
    """The system prompt with the configured thresholds filled in, so the model and the code enforce the same numbers."""
    return (_INSTRUCTIONS_TEMPLATE.replace("@RVOL_MIN@", f"{rules.rvol_min:g}")
            .replace("@MIN_FACTORS@", str(rules.min_factors)).replace("@MIN_RR@", f"{rules.min_rr:g}"))


SYSTEM_INSTRUCTIONS = system_instructions(Rules())  # default-threshold copy (kept for reference/tests)


def _round(x, sig=6):
    if isinstance(x, bool) or not isinstance(x, float):
        return x
    if x == 0 or not math.isfinite(x):
        return x if math.isfinite(x) else None
    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return _round(obj)


def build_payload(ctx, lock):
    """The structured JSON the model sees (macro / derivatives / structure / bot state)."""
    return _clean({
        "symbol": ctx["symbol"], "current_price": ctx["price"],
        "atr_entry_timeframe": ctx["atr_entry"],
        "macro_context": ctx["macro_context"],
        "derivatives_positioning": ctx["derivatives_positioning"],
        "market_structure": ctx["market_structure"],
        "bot_state": {"allowed_directions": lock["allowed"], "current_bias": lock["bias"]},
    })


def build_prompt(payload):
    return "Decide on this setup. Payload:\n" + json.dumps(payload, separators=(",", ":"))


def _num(x, name, allow_zero=True):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise DecisionError(f"{name} must be a finite number")
    if not allow_zero and x <= 0:
        raise DecisionError(f"{name} must be > 0")
    return float(x)


def parse_decision(raw):
    """Validate and normalise the model's JSON. Raises DecisionError on any schema violation."""
    if not isinstance(raw, dict):
        raise DecisionError("reply is not a JSON object")
    action = raw.get("action")
    if action not in ACTIONS:
        raise DecisionError(f"bad action {action!r}")
    factors = raw.get("confluence_factors", [])
    if not isinstance(factors, list) or not all(isinstance(f, str) for f in factors):
        raise DecisionError("confluence_factors must be a list of strings")
    tps = raw.get("take_profit_levels", [])
    if not isinstance(tps, list):
        raise DecisionError("take_profit_levels must be a list")
    d = {"action": action, "confluence_factors": [f.strip() for f in factors if f.strip()],
         "reasoning": str(raw.get("reasoning", ""))[:600]}
    if action in ("HOLD", "NO_TRADE"):
        d.update(confidence_score=0.0, invalidation_level=0.0, entry_price=0.0, take_profit_levels=[],
                 risk_reward_ratio=0.0)
        return d
    d.update(confidence_score=_num(raw.get("confidence_score"), "confidence_score"),
             invalidation_level=_num(raw.get("invalidation_level"), "invalidation_level", False),
             entry_price=_num(raw.get("entry_price"), "entry_price", False),
             take_profit_levels=[_num(t, "take_profit_levels[]", False) for t in tps],
             risk_reward_ratio=_num(raw.get("risk_reward_ratio", 0), "risk_reward_ratio"))
    if not 0 <= d["confidence_score"] <= 1:
        raise DecisionError("confidence_score outside 0-1")
    if not d["take_profit_levels"]:
        raise DecisionError("an OPEN action needs at least one take-profit")
    return d


def enforce_rules(d, ctx, lock, rules):
    """Return a list of rule violations (empty = tradable). Code-side re-check of every hard rule."""
    if d["action"] not in ("OPEN_LONG", "OPEN_SHORT"):
        return [f"action is {d['action']}"]
    side = "LONG" if d["action"] == "OPEN_LONG" else "SHORT"
    bad = []
    rv = ctx.get("rvol")
    if rv is None or rv < rules.rvol_min:
        bad.append(f"RVOL {rv if rv is None else round(rv, 2)} < {rules.rvol_min}")
    if side not in lock["allowed"]:
        bad.append(f"{side} not allowed (allowed: {lock['allowed']})")
    if ctx["macro_context"]["correlation_risk"]["long" if side == "LONG" else "short"]:
        bad.append("; ".join(ctx["macro_context"]["correlation_risk"]["reasons"]) or "BTC correlation risk")
    if d["confidence_score"] < rules.min_confidence:
        bad.append(f"confidence {d['confidence_score']:.2f} < {rules.min_confidence}")
    if len(d["confluence_factors"]) < rules.min_factors:
        bad.append(f"only {len(d['confluence_factors'])} confluence factors")
    price, atr_v = ctx["price"], ctx["atr_entry"]
    e, sl, tp = d["entry_price"], d["invalidation_level"], d["take_profit_levels"][0]
    if abs(e - price) > rules.max_entry_dev_atr * atr_v:
        bad.append(f"entry {e} too far from price {price}")
    ok_geo = (sl < e < tp) if side == "LONG" else (tp < e < sl)
    if not ok_geo:
        bad.append("entry / invalidation / take-profit on wrong sides")
        return bad
    risk, reward = abs(e - sl), abs(tp - e)
    if risk < rules.min_sl_atr * atr_v:
        bad.append("stop too tight (inside noise)")
    if risk / e > rules.max_sl_pct:
        bad.append(f"stop {risk / e:.1%} away: too wide for 10x")
    rr = reward / risk if risk else 0.0
    if rr < rules.min_rr:
        bad.append(f"R:R {rr:.2f} < {rules.min_rr}")
    return bad


def merge_decisions(decisions):
    """Consensus of several AIs: they must agree on the action; levels are averaged, factors united."""
    vals = list(decisions.values())
    if len(vals) == 1:
        return vals[0]
    actions = {v["action"] for v in vals}
    if len(actions) > 1:
        return {"action": "NO_TRADE", "confidence_score": 0.0, "confluence_factors": [], "invalidation_level": 0.0,
                "entry_price": 0.0, "take_profit_levels": [], "risk_reward_ratio": 0.0,
                "reasoning": "AIs disagree: " + ", ".join(f"{k}={v['action']}" for k, v in decisions.items())}
    if actions.isdisjoint({"OPEN_LONG", "OPEN_SHORT"}):
        return vals[0]
    mean = lambda xs: sum(xs) / len(xs)
    factors = list(dict.fromkeys(f for v in vals for f in v["confluence_factors"]))
    return {"action": vals[0]["action"], "confidence_score": mean([v["confidence_score"] for v in vals]),
            "confluence_factors": factors,
            "invalidation_level": mean([v["invalidation_level"] for v in vals]),
            "entry_price": mean([v["entry_price"] for v in vals]),
            "take_profit_levels": [mean([v["take_profit_levels"][0] for v in vals])],
            "risk_reward_ratio": mean([v["risk_reward_ratio"] for v in vals]),
            "reasoning": " | ".join(f"{k}: {v['reasoning']}" for k, v in decisions.items())[:600]}


def build_signal(d, ctx, rules):
    """Turn an approved decision into the Cornix signal: 4 entries, exactly 1 TP, 1 SL (cross 10x is added on output).

    Entry 1 is the reference entry; the other three step away from price by ladder_step_atr * ATR, compressed so the
    last entry stays well inside the stop (at most 60% of the way to it)."""
    side = "LONG" if d["action"] == "OPEN_LONG" else "SHORT"
    sgn = -1 if side == "LONG" else 1  # direction in which entries move away from price
    e, sl, tp = d["entry_price"], d["invalidation_level"], d["take_profit_levels"][0]
    step = min(rules.ladder_step_atr * ctx["atr_entry"], 0.6 * abs(e - sl) / 3)
    entries = [e + sgn * step * k for k in range(4)]
    risk = abs(sum(entries) / 4 - sl)
    rr = abs(tp - sum(entries) / 4) / risk if risk else 0.0
    why = "; ".join(d["confluence_factors"][:4])
    return {"symbol": ctx["symbol"], "side": side, "entries": [_round(x) for x in entries], "tp": _round(tp),
            "sl": _round(sl), "confidence": d["confidence_score"], "rr": rr,
            "reason": f"{why} | {d['reasoning']}"[:400]}
