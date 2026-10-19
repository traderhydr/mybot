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
