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
