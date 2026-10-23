# AI Telegram Trade Signal Bot

Gemini and Claude both analyse Binance futures candles (EMA20/50/200, RSI, ATR) and posts a signal to your Telegram channel
in Cornix format. Every signal has **4 entries, 1 TP, 1 SL, Cross 10x**; output that doesn't meet that
structure is rejected and regenerated. It only posts signals; it does not place orders.

## Setup
1. Create a bot with @BotFather and add it to your channel as an admin (post permission).
2. Get a Gemini key (aistudio.google.com/apikey) and a Claude key (console.anthropic.com). With both set, they must agree on symbol and direction; levels are averaged, otherwise no signal is posted. With only one key set, that AI is used alone.
3. `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
4. `cp .env.example .env` and fill it in (no quotes, no trailing comments).
5. `python bot.py --dry` prints a signal; `--once` posts one; no flag runs on a loop (`INTERVAL_MINUTES`).

## Cornix
Messages use Cornix's format (`#BTC/USDT`, `Long`, `Leverage: Cross (10x)`, `Entry Targets`,
`Take-Profit Targets`, `Stop Targets`). Link your channel to a Cornix signals group and connect a demo account first.

## Modal
```
modal secret create mybot-secrets TELEGRAM_BOT_TOKEN=... TELEGRAM_CHANNEL_ID=@chan GEMINI_API_KEY=... ANTHROPIC_API_KEY=sk-ant-... SYMBOLS=BTCUSDT,ETHUSDT
modal run modal_app.py      # post one now
modal deploy modal_app.py   # every 4 hours
```

## Backtest
`python backtest.py --ai rule --days 30` runs a free EMA-trend baseline for comparison (no AI calls).
`python backtest.py --ai bot --coins 10 --days 14 --step-hours 12 --yes` replays your real AI setup on past candles
(each point = same snapshot, prompt and validation as live) and simulates the 4-entry ladder, TP, SL and a 48h expiry.
Results are cached in `backtest_cache.jsonl` (stop/resume safely) and written to `backtest_results.csv`. It also runs
controls on the same points with the same ladder geometry: AI flipped, always SHORT, always LONG and a random side,
and prints a paired AI-minus-control comparison with a z-score. Use `--only claude|gemini` to test a single AI.
Caveats: candle-level simulation, no slippage/funding, SL wins ties, use recent data (after the AI models' training cutoff).

Backtest options: `--end-days-ago N` tests an earlier window (e.g. `--days 14 --end-days-ago 14`); `--fx-tp/--fx-sl` set the
Test 2 ladder (ATR multiples). Test 1 = the AI's own levels vs controls; Test 2 = only its long/short choice on one standard
ladder vs the same controls (is the direction any good?). Sampling is aligned to UTC clock times so the cache is reused.
