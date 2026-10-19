"""Run the signal bot on Modal on a schedule.

Setup (once):
    modal secret create mybot-secrets \
        TELEGRAM_BOT_TOKEN=... TELEGRAM_CHANNEL_ID=@your_channel \
        GEMINI_API_KEY=... ANTHROPIC_API_KEY=sk-ant-... SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
Deploy:   modal deploy modal_app.py
Test now: modal run modal_app.py
"""
import modal

app = modal.App("mybot")

image = (
    modal.Image.debian_slim()
    .pip_install("anthropic>=0.40.0", "requests>=2.31.0", "python-dotenv>=1.0.0")
    .add_local_python_source("bot")
)

# Binance futures blocks US IPs, so run in Europe.
REGION = "eu"


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("mybot-secrets")],
    schedule=modal.Cron("0 */4 * * *"),  # every 4 hours (UTC)
    region=REGION,
    timeout=300,
)
def post_signal():
    import bot  # imported here so env vars from the secret are set first
    bot.run_once()


@app.local_entrypoint()
def main():
    post_signal.remote()
