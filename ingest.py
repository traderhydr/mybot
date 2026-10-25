"""v2 market data ingestion and metrics.

Everything here is deterministic code (no LLM): multi-timeframe structure, volume profile, RVOL, FVGs, liquidity
zones, BTC market context, funding / open interest / CVD. All candle inputs are *closed* candles only.

A candle is a dict: {"t": open_ms, "o", "h", "l", "c", "v": base volume, "tb": taker-buy base volume}.
"""
import logging
import os
import time

import requests

log = logging.getLogger("ingest")

TF_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
         "2h": 7_200_000, "4h": 14_400_000, "1d": 86_400_000}
HOUR_MS = 3_600_000
FAPI = os.environ.get("BINANCE_FAPI_URL", "https://fapi.binance.com")


class DataError(Exception):
    """Market data could not be fetched or is unusable. Callers skip the symbol; they never trade on it."""


def parse_kline_rows(rows, symbol="?", now_ms=None):
    """Binance kline rows -> candle dicts, dropping any candle that has not closed yet."""
    now = int(time.time() * 1000) if now_ms is None else now_ms
    out = []
    try:
        for k in rows:
            if int(k[6]) >= now:
                continue
            out.append({"t": int(k[0]), "o": float(k[1]), "h": float(k[2]), "l": float(k[3]),
                        "c": float(k[4]), "v": float(k[5]), "tb": float(k[9])})
    except (IndexError, TypeError, ValueError) as ex:
        raise DataError(f"malformed klines for {symbol}") from ex
    return out


# ---------------------------------------------------------------- Binance client

class BinanceClient:
    """Thin REST wrapper: spacing between calls, retries with backoff, rate-limit (429/418) handling."""

    def __init__(self, base=FAPI, session=None, min_interval=0.05, timeout=15, retries=4, sleep=time.sleep):
        self.base, self.timeout, self.retries, self._sleep = base.rstrip("/"), timeout, retries, sleep
        self.session = session or requests.Session()
        self.min_interval, self._last = min_interval, 0.0
        self._intervals = None

    def get(self, path, params=None):
        last_err = None
        for attempt in range(self.retries + 1):
            gap = self.min_interval - (time.time() - self._last)
            if gap > 0:
                self._sleep(gap)
            self._last = time.time()
            try:
                r = self.session.get(self.base + path, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as ex:
                last_err = ex
                self._sleep(min(2 ** attempt, 20))
                continue
            if r.status_code in (418, 429):  # rate limited / temporarily banned: honour Retry-After
                wait = int(r.headers["Retry-After"]) if r.headers.get("Retry-After", "").isdigit() else 5 * 2 ** attempt
                last_err = DataError(f"HTTP {r.status_code} rate limit")
                self._sleep(min(wait, 120))
                continue
            if r.status_code >= 500:
                last_err = DataError(f"HTTP {r.status_code}")
                self._sleep(min(2 ** attempt, 20))
                continue
            if r.status_code == 451:
                raise DataError("HTTP 451: Binance blocks this server's region")
            if r.status_code >= 400:
                raise DataError(f"HTTP {r.status_code} on {path}: {r.text[:160]}")
            try:
                return r.json()
            except ValueError as ex:
                raise DataError(f"bad JSON from {path}") from ex
        raise DataError(f"{path} failed after {self.retries + 1} tries: {last_err}")

    # -- endpoints

    def klines(self, symbol, interval, limit):
        """Closed candles only (the still-forming last candle is dropped)."""
        return parse_kline_rows(self.get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit}),
                                symbol)

    def universe(self, symbols, max_coins=0, min_volume=0.0):
        """Tradable USDT perpetuals, most-traded first. `symbols` = ['ALL'] or an explicit list."""
        if symbols != ["ALL"]:
            return list(symbols)
        info = self.get("/fapi/v1/exchangeInfo")
        live = {x["symbol"] for x in info.get("symbols", [])
                if x.get("status") == "TRADING" and x.get("contractType") == "PERPETUAL" and x.get("quoteAsset") == "USDT"}
        vol = {t["symbol"]: float(t["quoteVolume"]) for t in self.get("/fapi/v1/ticker/24hr") if t["symbol"] in live}
        coins = [s for s, v in sorted(vol.items(), key=lambda kv: -kv[1]) if v >= min_volume]
        return coins[:max_coins] if max_coins else coins

    def premium_index(self):
        """{symbol: {"rate": current funding rate, "mark": mark price}} for all symbols in one call."""
        out = {}
        for x in self.get("/fapi/v1/premiumIndex"):
            try:
                out[x["symbol"]] = {"rate": float(x["lastFundingRate"]), "mark": float(x["markPrice"])}
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def funding_intervals(self):
        """{symbol: funding interval hours}; symbols not listed use the 8h default."""
        if self._intervals is None:
            try:
                self._intervals = {x["symbol"]: int(x["fundingIntervalHours"]) for x in self.get("/fapi/v1/fundingInfo")}
            except (DataError, KeyError, TypeError, ValueError) as ex:
                log.warning("fundingInfo unavailable (%s); assuming 8h", ex)
                self._intervals = {}
        return self._intervals

    def open_interest_hist(self, symbol, period="15m", limit=30):
        """Oldest-first list of open interest values (contracts)."""
        rows = self.get("/futures/data/openInterestHist", {"symbol": symbol, "period": period, "limit": limit})
        try:
            return [float(r["sumOpenInterest"]) for r in sorted(rows, key=lambda r: r["timestamp"])]
        except (KeyError, TypeError, ValueError) as ex:
            raise DataError(f"malformed open interest for {symbol}") from ex


# ---------------------------------------------------------------- indicators

def ema_series(values, period):
    k, out = 2 / (period + 1), [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def true_ranges(c):
    return [max(c[i]["h"] - c[i]["l"], abs(c[i]["h"] - c[i - 1]["c"]), abs(c[i]["l"] - c[i - 1]["c"]))
            for i in range(1, len(c))]


def atr(c, period=14):
    """Wilder ATR."""
    trs = true_ranges(c)
    if len(trs) < period:
        raise DataError("not enough candles for ATR")
    a = sum(trs[:period]) / period
    for tr in trs[period:]:
        a = (a * (period - 1) + tr) / period
    return a


def rsi(closes, period=14):
    if len(closes) <= period:
        raise DataError("not enough candles for RSI")
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains, losses = gains + max(d, 0), losses + max(-d, 0)
    ag, al = gains / period, losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag, al = (ag * (period - 1) + max(d, 0)) / period, (al * (period - 1) + max(-d, 0)) / period
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def rvol(c, period=20):
    """Relative volume: last closed bar volume / mean volume of the `period` bars before it."""
    if len(c) < period + 1:
        return None
    base = sum(x["v"] for x in c[-period - 1:-1]) / period
    return c[-1]["v"] / base if base > 0 else None


def price_change(c, hours):
    """Fractional close-to-close change over the last `hours` (None if history is too short)."""
    target = c[-1]["t"] - hours * HOUR_MS
    past = [x for x in c if x["t"] <= target]
    if not past or past[-1]["c"] <= 0:
        return None
    return c[-1]["c"] / past[-1]["c"] - 1


# ---------------------------------------------------------------- volume profile

def volume_profile(c, hours=24, bins=48, value_area=0.70):
    """Rolling-window volume profile from candles (each candle's volume spread evenly over its high-low range).
    Returns POC, VAH, VAL or None when there is not enough data."""
    if len(c) < 2:
        return None
    window = [x for x in c if x["t"] > c[-1]["t"] - hours * HOUR_MS]
    if len(window) < 10:
        return None
    lo, hi = min(x["l"] for x in window), max(x["h"] for x in window)
    if hi <= lo:
        return None
    width = (hi - lo) / bins
    vol = [0.0] * bins
    for x in window:
        b0, b1 = min(bins - 1, int((x["l"] - lo) / width)), min(bins - 1, int((x["h"] - lo) / width))
        share = x["v"] / (b1 - b0 + 1)
        for b in range(b0, b1 + 1):
            vol[b] += share
    total = sum(vol)
    if total <= 0:
        return None
    poc = max(range(bins), key=vol.__getitem__)
    lo_i = hi_i = poc
    acc = vol[poc]
    while acc < value_area * total and (lo_i > 0 or hi_i < bins - 1):
        up = vol[hi_i + 1] + (vol[hi_i + 2] if hi_i + 2 < bins else 0.0) if hi_i < bins - 1 else -1.0
        dn = vol[lo_i - 1] + (vol[lo_i - 2] if lo_i - 2 >= 0 else 0.0) if lo_i > 0 else -1.0
        if up >= dn:
            hi_i += 1
            acc += vol[hi_i]
        else:
            lo_i -= 1
            acc += vol[lo_i]
    return {"poc": lo + (poc + 0.5) * width, "vah": lo + (hi_i + 1) * width, "val": lo + lo_i * width}


# ---------------------------------------------------------------- market structure

def swing_points(c, left=3, right=3):
    """Confirmed fractal swing highs / lows as [(index, price)]."""
    highs, lows = [], []
    for j in range(left, len(c) - right):
        if all(c[j]["h"] > c[k]["h"] for k in range(j - left, j)) and all(c[j]["h"] >= c[k]["h"] for k in range(j + 1, j + right + 1)):
            highs.append((j, c[j]["h"]))
        if all(c[j]["l"] < c[k]["l"] for k in range(j - left, j)) and all(c[j]["l"] <= c[k]["l"] for k in range(j + 1, j + right + 1)):
            lows.append((j, c[j]["l"]))
    return highs, lows


def market_structure(c, left=3, right=3):
    """Walk the candles once, tracking the active (unbroken) swing levels.
    A close beyond the active swing in the trend direction is a BOS; a close beyond the opposite swing is a CHoCH."""
    sh = sl = None
    trend = None
    last = None
    highs, lows = [], []
    for i in range(len(c)):
        j = i - right
        if j >= left:
            if all(c[j]["h"] > c[k]["h"] for k in range(j - left, j)) and all(c[j]["h"] >= c[k]["h"] for k in range(j + 1, j + right + 1)):
                sh = (c[j]["h"], j)
                highs.append(sh)
            if all(c[j]["l"] < c[k]["l"] for k in range(j - left, j)) and all(c[j]["l"] <= c[k]["l"] for k in range(j + 1, j + right + 1)):
                sl = (c[j]["l"], j)
                lows.append(sl)
        close = c[i]["c"]
        if sh and close > sh[0]:
            last = {"event": "BOS_BULL" if trend in (None, "bull") else "CHOCH_BULL", "idx": i, "level": sh[0]}
            trend, sh = "bull", None
        elif sl and close < sl[0]:
            last = {"event": "BOS_BEAR" if trend in (None, "bear") else "CHOCH_BEAR", "idx": i, "level": sl[0]}
            trend, sl = "bear", None
    return {"bias": {"bull": "BULLISH", "bear": "BEARISH", None: "RANGE"}[trend],
            "last_event": last["event"] if last else None,
            "bars_since_event": len(c) - 1 - last["idx"] if last else None,
            "event_level": last["level"] if last else None,
            "swing_high": highs[-1][1] if highs else None, "swing_low": lows[-1][1] if lows else None}


def find_fvgs(c, atr_v, min_size_atr=0.1, lookback=120):
    """3-candle imbalances in the last `lookback` candles, flagged mitigated once price later trades back into them."""
    out = []
    for i in range(max(2, len(c) - lookback), len(c)):
        a, d = c[i - 2], c[i]
        if d["l"] > a["h"] and d["l"] - a["h"] >= min_size_atr * atr_v:
            kind, bottom, top = "BULL", a["h"], d["l"]
        elif d["h"] < a["l"] and a["l"] - d["h"] >= min_size_atr * atr_v:
            kind, bottom, top = "BEAR", d["h"], a["l"]
        else:
            continue
        mitigated = any(x["l"] <= top and x["h"] >= bottom for x in c[i + 1:])
        out.append({"type": kind, "bottom": bottom, "top": top, "idx": i, "mitigated": mitigated})
    return out


def nearest_unmitigated_fvgs(fvgs, price, atr_v, n=2):
    res = []
    for kind in ("BULL", "BEAR"):
        zs = [z for z in fvgs if z["type"] == kind and not z["mitigated"]]
        zs.sort(key=lambda z: abs((z["bottom"] + z["top"]) / 2 - price))
        res += [{"type": f"{kind}_FVG", "bottom": z["bottom"], "top": z["top"],
                 "dist_atr": round(((z["bottom"] + z["top"]) / 2 - price) / atr_v, 2)} for z in zs[:n]]
    return res


def liquidity_zones(c_htf, c_daily, price, atr_htf, tol_atr=0.15, n=6):
    """Equal highs / equal lows on the HTF (resting stops) plus the prior-day high / low."""
    zones = []
    highs, lows = swing_points(c_htf, 2, 2)
    for kind, pts, pick in (("EQH", highs, max), ("EQL", lows, min)):
        levels = sorted(p for _, p in pts[-12:])
        used = set()
        for i, p in enumerate(levels):
            if i in used:
                continue
            cluster = [q for j, q in enumerate(levels) if abs(q - p) <= tol_atr * atr_htf and j not in used]
            if len(cluster) >= 2:
                used.update(j for j, q in enumerate(levels) if q in cluster)
                zones.append({"type": kind, "price": pick(cluster)})
    if c_daily:
        zones += [{"type": "PDH", "price": c_daily[-1]["h"]}, {"type": "PDL", "price": c_daily[-1]["l"]}]
    for z in zones:
        z["dist_atr"] = round((z["price"] - price) / atr_htf, 2)
    zones.sort(key=lambda z: abs(z["dist_atr"]))
    return zones[:n]


# ---------------------------------------------------------------- BTC / market regime

def trend_state(c):
    """Classify one timeframe: REJECTION (with TOP/BOTTOM), BULLISH_EXPANSION, BEARISH_EXPANSION or COMPRESSION."""
    if len(c) < 60:
        raise DataError("not enough candles for trend state")
    closes = [x["c"] for x in c]
    e20, e50 = ema_series(closes, 20), ema_series(closes, 50)
    a14 = atr(c, 14)
    trs = true_ranges(c)[-100:]
    ratio = (sum(trs[-14:]) / 14) / (sum(trs) / len(trs)) if trs else 1.0
    st = market_structure(c)
    price = closes[-1]
    out = {"atr_ratio": round(ratio, 2), "structure": st["bias"], "last_event": st["last_event"],
           "above_ema50": price > e50[-1], "rejection_side": None}
    for x in (c[-1], c[-2]):  # a rejection candle stays relevant for one more bar
        rng = x["h"] - x["l"]
        if rng < 1.2 * a14:
            continue
        prior = c[-22:-2]
        if (x["h"] - max(x["o"], x["c"])) / rng >= 0.6 and (x["c"] - x["l"]) / rng <= 0.4 \
                and x["h"] >= max(p["h"] for p in prior) - 0.5 * a14:
            out.update(state="REJECTION", rejection_side="TOP")
            return out
        if (min(x["o"], x["c"]) - x["l"]) / rng >= 0.6 and (x["h"] - x["c"]) / rng <= 0.4 \
                and x["l"] <= min(p["l"] for p in prior) + 0.5 * a14:
            out.update(state="REJECTION", rejection_side="BOTTOM")
            return out
    slope = e20[-1] / e20[-6] - 1
    if ratio <= 0.8:
        out["state"] = "COMPRESSION"
    elif price > e20[-1] > e50[-1] and slope > 0 and st["bias"] != "BEARISH":
        out["state"] = "BULLISH_EXPANSION"
    elif price < e20[-1] < e50[-1] and slope < 0 and st["bias"] != "BULLISH":
        out["state"] = "BEARISH_EXPANSION"
    else:
        out["state"] = "COMPRESSION"
    return out


def btc_context(c1h, c4h):
    s1, s4 = trend_state(c1h), trend_state(c4h)
    return {"1h": s1, "4h": s4, "state": s4["state"], "rejection_side": s4["rejection_side"]}


def correlation_risk(symbol, btc):
    """Would a long / short on this coin fight BTC's higher-timeframe structure?"""
    if symbol == "BTCUSDT":  # BTC cannot clash with itself
        return {"long": False, "short": False, "reasons": [], "known": True}
    if btc is None:
        return {"long": False, "short": False, "reasons": ["BTC context unavailable"], "known": False}
    st, side, reasons = btc["state"], btc["rejection_side"], []
    risk_long = st == "BEARISH_EXPANSION" or (st == "REJECTION" and side == "TOP")
    risk_short = st == "BULLISH_EXPANSION" or (st == "REJECTION" and side == "BOTTOM")
    if risk_long:
        reasons.append(f"BTC 4h {st}{' (' + side + ')' if side else ''} opposes altcoin longs")
    if risk_short:
        reasons.append(f"BTC 4h {st}{' (' + side + ')' if side else ''} opposes altcoin shorts")
    return {"long": risk_long, "short": risk_short, "reasons": reasons, "known": True}


# ---------------------------------------------------------------- derivatives

OI_THR = float(os.environ.get("OI_THRESHOLD", "0.005"))       # |dOI| below this = unchanged
PRICE_THR = float(os.environ.get("PRICE_THRESHOLD", "0.002"))  # |dPrice| below this = unchanged
FUNDING_EXTREME = float(os.environ.get("FUNDING_EXTREME", "0.0003"))  # per 8h


def positioning_regime(d_price, d_oi):
    if d_price is None or d_oi is None or abs(d_price) < PRICE_THR or abs(d_oi) < OI_THR:
        return "NEUTRAL"
    if d_price > 0:
        return "AGGRESSIVE_LONG_BUILD" if d_oi > 0 else "SHORT_SQUEEZE_COVERING"
    return "AGGRESSIVE_SHORT_BUILD" if d_oi > 0 else "LONG_LIQUIDATION"


def cvd_analysis(c, window=24):
    """Cumulative volume delta from taker-buy volume, plus divergence between the last two price extremes."""
    if len(c) < window * 2:
        return {"divergence": "UNKNOWN", "slope": None}
    delta = [2 * x["tb"] - x["v"] for x in c]
    cvd, run = [], 0.0
    for d in delta:
        run += d
        cvd.append(run)
    seg = c[-window:]
    cseg = cvd[-window:]
    split = int(window * 0.75)
    hi_new = max(range(window), key=lambda i: seg[i]["h"])
    lo_new = min(range(window), key=lambda i: seg[i]["l"])
    div = "NONE"
    if hi_new >= split:  # price printed its high recently: was it confirmed by buying pressure?
        hi_old = max(range(split), key=lambda i: seg[i]["h"])
        if seg[hi_new]["h"] > seg[hi_old]["h"] and cseg[hi_new] < cseg[hi_old]:
            div = "BEARISH_DIVERGENCE"
    if div == "NONE" and lo_new >= split:
        lo_old = min(range(split), key=lambda i: seg[i]["l"])
        if seg[lo_new]["l"] < seg[lo_old]["l"] and cseg[lo_new] > cseg[lo_old]:
            div = "BULLISH_DIVERGENCE"
    return {"divergence": div, "slope": "UP" if cseg[-1] > cseg[0] else "DOWN", "last_bar_delta": delta[-1]}


def derivatives_block(symbol, entry, shared, client):
    """Funding, open interest deltas, positioning regime and CVD for one symbol. Missing pieces become None."""
    out = {"funding": None, "open_interest": None, "positioning_regime": None, "cvd": cvd_analysis(entry)}
    prem = (shared.get("premium") or {}).get(symbol)
    if prem:
        interval = (shared.get("intervals") or {}).get(symbol, 8)
        rate = prem["rate"]
        eq8 = rate * 8 / interval
        out["funding"] = {"rate": rate, "interval_h": interval, "rate_8h_equiv": eq8,
                          "crowding": "LONGS_CROWDED" if eq8 >= FUNDING_EXTREME else
                                      "SHORTS_CROWDED" if eq8 <= -FUNDING_EXTREME else "NORMAL"}
    try:
        oi = client.open_interest_hist(symbol, "15m", 30)
    except DataError as ex:
        log.warning("%s: open interest unavailable (%s)", symbol, ex)
        oi = []
    if len(oi) >= 5 and oi[-5] > 0:
        d1 = oi[-1] / oi[-5] - 1
        d4 = oi[-1] / oi[-17] - 1 if len(oi) >= 17 and oi[-17] > 0 else None
        p1, p4 = price_change(entry, 1), price_change(entry, 4)
        out["open_interest"] = {"delta_1h": d1, "delta_4h": d4}
        out["positioning_regime"] = {"1h": positioning_regime(p1, d1), "4h": positioning_regime(p4, d4),
                                     "price_change_1h": p1, "price_change_4h": p4}
    return out


# ---------------------------------------------------------------- context assembly

def build_shared(client):
    """Once per cycle: BTC context plus the all-symbol funding tables. Failures degrade to None, never raise."""
    shared = {"btc": None, "premium": None, "intervals": None, "errors": []}
    try:
        shared["btc"] = btc_context(client.klines("BTCUSDT", "1h", 300), client.klines("BTCUSDT", "4h", 300))
    except (DataError, ValueError) as ex:
        shared["errors"].append(f"btc: {ex}")
    try:
        shared["premium"] = client.premium_index()
        shared["intervals"] = client.funding_intervals()
    except DataError as ex:
        shared["errors"].append(f"funding: {ex}")
    for e in shared["errors"]:
        log.warning("shared context degraded: %s", e)
    return shared


def build_context(symbol, client, shared, entry_tf="15m", htf="4h", daily="1d", limit=400):
    """All metrics for one symbol. Raises DataError when the core candle data is missing or too short."""
    entry = client.klines(symbol, entry_tf, limit)
    h4 = client.klines(symbol, htf, 250)
    d1 = client.klines(symbol, daily, 120)
    if len(entry) < 150 or len(h4) < 60 or len(d1) < 20:
        raise DataError(f"{symbol}: not enough history")
    price = entry[-1]["c"]
    a_entry, a_htf = atr(entry), atr(h4)
    closes = [x["c"] for x in entry]
    vp = volume_profile(entry, 24)
    rv = rvol(entry, 20)
    st_entry, st_htf, st_d1 = market_structure(entry), market_structure(h4), market_structure(d1)
    fvgs = nearest_unmitigated_fvgs(find_fvgs(entry, a_entry), price, a_entry) + \
        nearest_unmitigated_fvgs(find_fvgs(h4, a_htf), price, a_htf, 1)
    triggers = []
    if st_entry["bars_since_event"] is not None and st_entry["bars_since_event"] <= 12:
        triggers.append(f"LTF {st_entry['last_event']} {st_entry['bars_since_event']} bars ago")
    if vp:
        loc = "ABOVE_VAH" if price > vp["vah"] else "BELOW_VAL" if price < vp["val"] else "INSIDE_VALUE_AREA"
        triggers.append(f"price {loc}")
    for z in fvgs:
        if z["bottom"] <= price <= z["top"]:
            triggers.append(f"price inside unmitigated {z['type']}")
    deriv = derivatives_block(symbol, entry, shared, client)
    return {
        "symbol": symbol, "price": price, "atr_entry": a_entry, "atr_htf": a_htf, "rvol": rv,
        "macro_context": {
            "btc": shared.get("btc"),
            "correlation_risk": correlation_risk(symbol, shared.get("btc")),
            "market_regime": (shared.get("btc") or {}).get("state", "UNKNOWN"),
        },
        "derivatives_positioning": deriv,
        "market_structure": {
            "htf_bias": {"daily": st_d1["bias"], "4h": st_htf["bias"]},
            "htf_last_event": {"daily": st_d1["last_event"], "4h": st_htf["last_event"]},
            "key_zones": {"volume_profile_24h": vp, "unmitigated_fvgs": fvgs,
                          "liquidity": liquidity_zones(h4, d1, price, a_htf)},
            "ltf_entry_triggers": triggers,
            "ltf_structure": {"bias": st_entry["bias"], "last_event": st_entry["last_event"],
                              "swing_high": st_entry["swing_high"], "swing_low": st_entry["swing_low"]},
            "rvol_20": rv,
            "momentum": {"rsi14": rsi(closes), "ema20": ema_series(closes, 20)[-1], "ema50": ema_series(closes, 50)[-1]},
        },
    }
