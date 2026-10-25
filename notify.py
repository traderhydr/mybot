"""Cornix-compatible signal formatting and Telegram delivery."""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

LEVERAGE = 10
MARGIN_MODE = "Cross"


def fmt(x):
    """Plain decimal, no thousands separators or exponents (Cornix can't parse those)."""
    return f"{x:.8f}".rstrip("0").rstrip(".")


def format_message(sig):
    """Cornix-compatible signal format. Keep it plain text with no extra numbers."""
    base = sig["symbol"][:-4] if sig["symbol"].endswith("USDT") else sig["symbol"]
    e = sig["entries"]
    return "\n".join([
        f"#{base}/USDT",
        sig["side"].capitalize(),
        f"Leverage: {MARGIN_MODE} ({LEVERAGE}x)",
        "",
        "Entry Targets:",
        *[f"{i}) {fmt(x)}" for i, x in enumerate(e, 1)],
        "",
        "Take-Profit Targets:",
        f"1) {fmt(sig['tp'])}",
        "",
        "Stop Targets:",
        f"1) {fmt(sig['sl'])}",
    ])


def post(text):
    token, channel = os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHANNEL_ID", "")
    if not (token and channel):
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHANNEL_ID not set")
    for _ in range(3):
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": channel, "text": text}, timeout=15)
        if r.status_code == 429:  # Telegram flood limit: wait as told, then retry
            time.sleep(r.json().get("parameters", {}).get("retry_after", 5) + 1)
            continue
        r.raise_for_status()
        return
    raise RuntimeError("telegram rate limit")
