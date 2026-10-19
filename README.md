# AI Telegram Trade Signal Bot

Claude analyses Binance futures candles (EMA20/50/200, RSI, ATR) and posts a signal to your
Telegram channel. Every signal has **4 entries, 1 TP, 1 SL, Cross 10x**. Output that doesn't
meet that structure is rejected and regenerated.

It only *posts* signals; it does not place orders.

## Setup
1. Create a bot with [@BotFather](https://t.me/BotFather), add it to your channel as an admin (post permission).
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and fill in the keys.
4. `python bot.py --dry` – prints a signal without posting. `python bot.py --once` posts one. `python bot.py` runs on a loop (`INTERVAL_MINUTES`).

## Cornix
Messages use Cornix's signal format (`#BTC/USDT`, `Long`, `Leverage: Cross (10x)`, `Entry Targets`,
`Take-Profit Targets`, `Stop Targets`), so Cornix can parse and execute them.
1. In Cornix, create a signals group/bot linked to your channel (add your Cornix-connected account as a channel
   member/the Cornix bot per their channel-connect flow) and connect a **demo/testnet** exchange account first.
2. Set the Cornix trade config to Cross 10x with entries "as given" so the 4 entries are used.
3. Verify a few posted signals are recognised before going live.

## Hosting on Modal
```
pip install modal && python -m modal setup
modal secret create mybot-secrets TELEGRAM_BOT_TOKEN=... TELEGRAM_CHANNEL_ID=@chan ANTHROPIC_API_KEY=sk-ant-... SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
modal run modal_app.py      # post one signal now
modal deploy modal_app.py   # run every 4 hours (edit the Cron in modal_app.py)
```

## DeepSeek instead of Claude
Set `DEEPSEEK_API_KEY` (from platform.deepseek.com) and the bot uses DeepSeek (`deepseek-chat`, override with
`DEEPSEEK_MODEL`). If it is not set, Claude is used.

## Any OpenAI-compatible API
Set `OPENAI_BASE_URL` (up to and including `/v1`), `OPENAI_API_KEY` and `OPENAI_MODEL`. Priority: OpenAI-compatible,
then DeepSeek, then Claude.

## Claude via Puter.js
Needs Node.js 18+ and a Puter account (usage is billed to that account's allowance; there is no truly keyless
mode outside a browser).
```
npm install
node puter_ai.cjs --login      # on a machine with a browser; prints your auth token
```
Put the token in `.env` as `PUTER_AUTH_TOKEN=...` and run `python bot.py --dry`.

## Gemini
Get a key at aistudio.google.com/apikey, set `GEMINI_API_KEY` (optionally `GEMINI_MODEL`, default `gemini-2.5-flash`).
