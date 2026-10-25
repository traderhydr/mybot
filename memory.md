# Project memory: AI Telegram trade-signal bot

Log of each step taken, in order. Repo: `traderhydr/mybot`, branch `claude/practical-sagan-h9x7co`.

## Goal
Telegram bot that uses AI to post trade signals to a channel. Every signal has **4 entries, 1 TP, 1 SL, Cross 10x**.
Execution is done by **Cornix** (on a demo account first). The bot only posts signals; it never places orders.

## Steps

1. **Initial build.** Empty repo. Wrote `bot.py` (Python): pulls Binance futures candles, computes EMA20/50/200, RSI14, ATR14, asks an AI for one setup as JSON, validates it, posts to Telegram. Added `requirements.txt`, `.env.example`, `.gitignore`, `README.md`.
2. **Validation rules.** Exactly 4 entries ordered away from price, one TP and SL on the correct side, SL no wider than ~5% price move (50% of margin at 10x). Bad AI output is rejected and re-asked (3 attempts).
3. **Cornix format.** Switched to Cornix's signal format: `#BTC/USDT`, `Long`/`Short`, `Leverage: Cross (10x)`, `Entry Targets`, `Take-Profit Targets`, `Stop Targets`. Plain text, plain decimals (no commas, no HTML, no emoji) so Cornix can parse it. Reason text is printed to console only.
4. **Server setup.** Server (`mybot`, Debian/Ubuntu) had no git or Python. Installed `git python3 python3-pip python3-venv`, cloned the repo, created a `venv`, `pip install -r requirements.txt`, copied `.env.example` to `.env`.
5. **API key debugging (401).** Anthropic key in `.env` was only 10 characters (truncated). Checked shell env vars and `.env` length with commands that don't print the key. Lesson: no quotes, spaces or `# comments` in `.env` values; real keys are 100+ chars.
6. **Modal.** User installed Modal inside the venv (system pip is blocked by PEP 668). Added `modal_app.py`: scheduled every 4 hours, pinned to an EU region because Binance blocks some US IPs, secrets in a Modal secret named `mybot-secrets`. Not deployed or tested.
7. **Extra AI providers tried.** Added DeepSeek, a generic OpenAI-compatible endpoint, Claude via Puter.js (Node helper), then Gemini. Puter needs a Puter account token on a server (not keyless). The unofficial OpenAI-compatible proxy was flagged as risky.
8. **Leftover env bug.** An old `OPENAI_BASE_URL` in `.env` caused a `KeyError: 'OPENAI_MODEL'` and hid Gemini. Fixed by commenting out unused provider lines and logging the provider.
9. **Gemini only.** At the user's request removed all other providers (and the Node/Puter files). Gemini model name `gemini-2.5-flash` returned 404; model `gemini-3.8-flash` works. It was missing from `.env`, so added `GEMINI_MODEL=gemini-3.8-flash` and made it the code default.
10. **First working signal.** `python bot.py --dry` printed a valid BTC short (4 entries, 1 TP, SL, Cross 10x).
11. **Running continuously.** Started the bot in `tmux` session `bot` (`source venv/bin/activate && python bot.py`; detach Ctrl+B then D; reattach `tmux attach -t bot`). It posts a signal on start, then every `INTERVAL_MINUTES` (default 240).
12. **Gemini + Claude together.** Both AIs analyse the same data independently. If they agree on symbol and direction, levels are averaged into one signal; if they disagree, no trade that round. With one key set, that AI is used alone. Re-added the `anthropic` package and `CLAUDE_MODEL` (default `claude-sonnet-5-5`).
13. **Timeouts and speed.** A Gemini call timed out at 90s. Added `AI_TIMEOUT` (default 180s), 1 retry on Claude, and run both AI calls in parallel.

## Current state
- Code: `bot.py`, `modal_app.py`, `requirements.txt`, `.env.example`, `README.md`, `memory.md` on the branch above.
- Server: `~/mybot`, venv active with `source venv/bin/activate`; `.env` holds Telegram token + channel, `GEMINI_API_KEY`, `GEMINI_MODEL`, `ANTHROPIC_API_KEY`, `SYMBOLS`.
- Last run: Gemini call timed out once at 90s; latest push (parallel calls, 180s timeout) not yet confirmed on the server.
- The bot running in tmux uses old code until restarted (Ctrl+C, then `git pull`, then `python bot.py`).
- Not yet confirmed: that a post reached the channel, and that Cornix read all 4 entries, TP, SL and 10x Cross on the demo account.

## Next steps
- `git pull origin claude/practical-sagan-h9x7co`, `python bot.py --dry`, then `--once`, then restart the tmux bot.
- Verify the first posted signal in Cornix (demo account only).
- Optional: add more pairs to `SYMBOLS`; deploy on Modal (`modal secret create mybot-secrets ...` then `modal deploy modal_app.py`).
- Revisit validation strictness if too many signals are rejected.

## Notes
- Never paste API keys or tokens into chat or commit them; `.env` is git-ignored.
- Signals are AI-generated and not financial advice; stay on the demo account until results are proven.

## Later steps (Oct 20-25)
14. **Haiku as the Claude model; Gemini free quota** keeps running out -> circuit breaker + `GEMINI_RPM` pacing (llm.py).
15. **Live results on Cornix demo (Oct 19-22):** ~22 closed signals, ~30% win rate, avg win ~+31% vs avg loss ~-16% (about break-even
    to slightly negative); stops clustered on short signals when the market bounced; same coin re-signalled while open.
16. **Backtest harness (backtest.py):** replays the real prompt/validation on past candles, simulates the ladder (SL wins ties,
    fees, 48h expiry) and compares against controls (flipped, always SHORT/LONG, random side) on the same ladder shape, plus a
    direction-only test on one standard ladder. `--only claude|gemini`, `--end-days-ago`, cached in backtest_cache.jsonl.
17. **Backtest verdict (v1):** Haiku looked good in the recent 14 days (Test 2 z~3 vs always-long, 1.9 vs always-short) but failed
    the earlier window (-0.84%/trade); Sonnet was worse (-2.47%/trade, z=-2.5 vs always-short). Profit tracked market drift, not
    skill. Conclusion: no proven edge; demo only.
18. **v2 engine built** (ingest.py, state.py, decision.py, execution.py, llm.py, notify.py, test_v2.py; `STRATEGY=v2`, default v1).
    Adds RVOL, volume profile, MTF structure/FVG/liquidity, BTC regime + correlation gate, funding/OI/CVD, SQLite state with
    cooldown and flip lock, strict JSON decision schema with code-side rule enforcement. Unit tests pass; futures endpoints for
    funding/OI were not reachable from the dev environment, so run `--dry --limit 3` on the server first. NOT backtested.

19. **Gemini removed entirely** (free-tier quota kept failing; Claude-only). llm.py now only has Claude; bot.py v1, backtest.py, docs and Modal setup updated. Old backtest cache keys are migrated on load.
