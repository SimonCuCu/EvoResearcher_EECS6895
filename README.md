# EvoResearcher

EvoResearcher is an interactive deep-research proposal system with:

- Rich TUI with animated live phases
- LangGraph orchestration
- DeepSeek API as the model backend
- Dual memory plus an Evolution Memory Agent (EMA)
- Tree-search based ideation
- LaTeX report generation and direct PDF rendering

## Quick start

```bash
python -m evoresearcher.main --mode general
python -m evoresearcher.main --mode ml
python -m evoresearcher.main --mode general --goal "Investigate why social protection programs in South Asia often fail the ultra-poor as of September 2023."
```

Outputs are written under `outputs/<run-id>/`.

## Environment

Create `.env` with:

```bash
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/chat/completions
```

## Telegram channel

Install the optional dependency and add Telegram settings to `.env`:

```bash
pip install '.[telegram]'
```

```bash
TELEGRAM_BOT_TOKEN=...
TELEGRAM_ALLOWED_USER_IDS=123456789
```

Then start the long-polling bot:

```bash
evoresearcher-telegram
```

In Telegram, send `/start`, choose `General research` or `ML research`, then type the goal as a normal message. For ML runs, the bot will ask the intake questions with selectable Telegram buttons and an optional custom answer.

Utility commands:

```text
/status
/last
/cancel
/help
```

`/cancel` is intentionally not destructive in this MVP because the current research pipeline is a synchronous long-running job. Add cooperative cancellation to the graph before enabling hard interruption.
