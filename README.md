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
python -m evoresearcher.main --mode general --global-model deepseek-v4-flash --goal "Draft a fast evidence scan."
```

Outputs are written under `outputs/<run-id>/`.

## Environment

Create `.env` with:

```bash
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_REASONING_MODEL=deepseek-v4-pro
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

Telegram also asks for a model profile before the goal:

- `Default (.env)`: use `DEEPSEEK_MODEL` and `DEEPSEEK_REASONING_MODEL`.
- `Flash`: use `deepseek-v4-flash` globally for intake, research reasoning, proposal, and memory updates.
- `Pro reasoning`: use the default model for intake/proposal and `deepseek-v4-pro` for research reasoning.

The Telegram bot and TUI both support lightweight human-in-the-loop guidance:

- Add phrases like `ask me clarifying questions before...` to your goal to make the agent generate 2-3 clarification questions before building the brief.
- After candidate ideas are ranked, choose the direction that should anchor the final report.
- Before report writing, choose the emphasis: novelty, feasibility, evidence, risks, or a custom instruction.

Utility commands:

```text
/status
/last
/cancel
/help
```

`/cancel` is intentionally not destructive in this MVP because the current research pipeline is a synchronous long-running job. Add cooperative cancellation to the graph before enabling hard interruption.
