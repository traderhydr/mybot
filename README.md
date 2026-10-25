# AI Telegram Trade Signal Bot

Claude analyses Binance futures candles (EMA20/50/200, RSI, ATR) and posts a signal to your Telegram channel
in Cornix format. Every signal has **4 entries, 1 TP, 1 SL, Cross 10x**; output that doesn't meet that
structure is rejected and regenerated. It only posts signals; it does not place orders.

## Setup
1. Create a bot with @BotFather and add it to your channel as an admin (post permission).
2. Get a Claude API key (console.anthropic.com) and set `ANTHROPIC_API_KEY` (model: `CLAUDE_MODEL`, default `claude-haiku-5-5`).
3. `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
4. `cp .env.example .env` and fill it in (no quotes, no trailing comments).
5. `python bot.py --dry` prints a signal; `--once` posts one; no flag runs on a loop (`INTERVAL_MINUTES`).

## Cornix
Messages use Cornix's format (`#BTC/USDT`, `Long`, `Leverage: Cross (10x)`, `Entry Targets`,
`Take-Profit Targets`, `Stop Targets`). Link your channel to a Cornix signals group and connect a demo account first.

## Modal
```
modal secret create mybot-secrets TELEGRAM_BOT_TOKEN=... TELEGRAM_CHANNEL_ID=@chan ANTHROPIC_API_KEY=sk-ant-... SYMBOLS=BTCUSDT,ETHUSDT
modal run modal_app.py      # post one now
modal deploy modal_app.py   # every 4 hours
```

## Backtest
`python backtest.py --ai rule --days 30` runs a free EMA-trend baseline for comparison (no AI calls).
`python backtest.py --ai bot --coins 10 --days 14 --step-hours 12 --yes` replays your real AI setup on past candles
(each point = same snapshot, prompt and validation as live) and simulates the 4-entry ladder, TP, SL and a 48h expiry.
Results are cached in `backtest_cache.jsonl` (stop/resume safely) and written to `backtest_results.csv`. It also runs
controls on the same points with the same ladder geometry: AI flipped, always SHORT, always LONG and a random side,
and prints a paired AI-minus-control comparison with a z-score.
Caveats: candle-level simulation, no slippage/funding, SL wins ties, use recent data (after the AI models' training cutoff).

Backtest options: `--end-days-ago N` tests an earlier window (e.g. `--days 14 --end-days-ago 14`); `--fx-tp/--fx-sl` set the
Test 2 ladder (ATR multiples). Test 1 = the AI's own levels vs controls; Test 2 = only its long/short choice on one standard
ladder vs the same controls (is the direction any good?). Sampling is aligned to UTC clock times so the cache is reused.

## v2 engine (`STRATEGY=v2`)
Adds what v1 was blind to. Modules: `ingest.py` (data + metrics), `state.py` (SQLite positions/cooldowns),
`decision.py` (prompt, schema, rule gates, ladder), `execution.py` (cycle manager), `llm.py` / `notify.py` (shared).
- **Ingestion:** entry timeframe (default 15m) + 4h + 1d; RVOL(20); rolling 24h volume profile (POC/VAH/VAL); swing
  structure (BOS/CHoCH); unmitigated FVGs; liquidity pools (equal highs/lows, prior-day high/low); BTC 1h/4h state
  (Bullish/Bearish Expansion, Compression, Rejection) with a correlation-risk flag; funding, 1h/4h open-interest delta,
  positioning regime and CVD divergence.
- **State machine:** every posted signal is tracked on candles until TP / SL / cancel / expiry. No LLM call while a coin
  has an open signal or is cooling down, and no direction flip inside the flip window.
- **Model contract:** JSON in (`macro_context`, `derivatives_positioning`, `market_structure`, `bot_state`), strict JSON out
  (`action`, `confidence_score`, `confluence_factors`, `invalidation_level`, `entry_price`, `take_profit_levels`,
  `risk_reward_ratio`, `reasoning`). Code re-checks every hard rule (RVOL >= 1.2, BTC clash, allowed directions, geometry,
  R:R, stop width) and builds the 4-entry ladder / single TP / SL itself. (If several models are ever configured, they must agree on the action.)
- **Run:** `STRATEGY=v2 python bot.py --dry --limit 3` (no posting, in-memory state), then drop `--dry`. Tests: `python -m unittest test_v2`.
- **Not validated:** v2 has not been backtested; earlier v1 backtests showed no edge. Treat it as a hypothesis and keep Cornix on demo.

## v2 backtest (`backtest_v2.py`)
Replays the *live* v2 code (ingest -> decision -> state locks) on history with no look-ahead (a client that only sees data
closed/available at each sample time; open interest is usable one period after its timestamp), then simulates the 4-entry
ladder with the same pessimistic rules as `backtest.py`. Locks/cooldowns run on simulated time, so a coin with an open
simulated position is skipped just like live. Prints a funnel (points / locked / RVOL gate / asked / NO_TRADE / rejected /
signals + top rejection reasons), the controls table and the paired AI-minus-control z-scores.
```
python backtest_v2.py --ai rule   --coins 15 --days 28        # free: same gates, direction = 4h structure bias (no LLM)
python backtest_v2.py --ai claude --coins 15 --days 28        # prints the planned Claude calls; add --yes to run them
```
Open-interest history only exists for ~30 days: `--days + --end-days-ago <= 29`. Claude answers are cached by prompt hash
(`backtest_v2_cache.jsonl`), so reruns, rule tweaks and interrupted runs cost nothing extra.
