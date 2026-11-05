# llmterm

**A minimal terminal client for OpenAI-compatible LLM servers.**

`llmterm` is a small Python command-line application for interacting with
local or self-hosted LLM servers that expose an OpenAI-compatible API.

## Features

* Interactive endpoint selection (local + hosted + custom URLs)
* Automatic model discovery
* Interactive model selection (or `--model` for scripts)
* Asynchronous streaming responses
* Conversation history (with `clear` to reset)
* Optional custom system prompt
* Optional Markdown conversation logging
* `.env` support for API keys — **keys are never stored**
* Small, modular implementation

## Project structure

```text
llmterm/
├── pyproject.toml
├── README.md
├── .gitignore
└── src/
    └── llmterm/
        ├── __init__.py
        ├── endpoint.py
        ├── endpoints.py
        ├── main.py
        └── utils.py
```

`src/` contains only package source code. Runtime-generated files are not
stored inside the Python package.

## Markdown output location

When Markdown saving is enabled, responses are stored in:

```text
~/.llmterm/
├── config.json         # custom endpoint URLs only — never API keys
└── markdown-outputs/
    ├── model-name-2026-08-15.md
    └── another-model-2026-08-15.md
```

This location is independent of the directory from which `llmterm` is run.

Keeping generated data outside the package is important for a distributable
Python package because installed package directories should not be assumed
to be writable.

## Installation

### Using uv

Install `llmterm` as a standalone CLI tool:

```bash
uv tool install llmterm
```

Run:

```bash
llmterm
```

### Using pip

You can also install `llmterm` using pip:

```bash
python -m pip install llmterm
```

Run:

```bash
llmterm
```

## Development

Clone the repository and enter the project directory:

```bash
git clone https://github.com/prasannaba/llmterm.git
cd llmterm
```

Install the development environment and dependencies:

```bash
uv sync
```

Run the application:

```bash
uv run llmterm
```

## API keys: never stored

**llmterm never stores API keys.** Keys live in memory only for the selected
session and are never written to config files, logs, or markdown output.
You will see this notice at startup and before every key prompt.

Provide keys via the environment (recommended for scripts) or enter them
when prompted (input is hidden):

```env
OPENAI_API_KEY=your-key
META_API_KEY=your-key
BASETEN_API_KEY=your-key
UNSLOTH_STUDIO_API_KEY=your-key   # only when your server requires auth
```

There is deliberately no `--api-key` flag: keys on the command line leak via
shell history and process listings. Use `--api-key-env NAME` to point at a
different variable, or `--no-prompt` to fail fast instead of prompting.

Custom endpoint *URLs* may be remembered in `~/.llmterm/config.json`
(`{custom_endpoints: [{name, base_url, key_env}]}`) — keys are never saved
there, by construction and by test.

### Using llmterm installed via `uv tool`

`uv tool install` (or `pip install`) only installs the `llmterm` package —
repo files such as `.env.example` are **not** installed and won't be on disk.
That file is just a template for people running from a cloned repo. As a
tool user, provide keys any of these ways:

1. **Environment variables** (recommended): set them in your shell so every
   run picks them up, e.g. `OPENAI_API_KEY`, `META_API_KEY`,
   `BASETEN_API_KEY` (see `llmterm --show-env-template` for the full list).
2. **Your own `.env` file**: create a file named `.env` in the directory
   from which you run `llmterm` — it is read from the current working
   directory on startup. Generate a starting point with:
   ```bash
   llmterm --show-env-template > .env
   ```
   then uncomment the lines you need and fill in your keys. Do not commit
   this file anywhere; it holds secrets. (The template contains variable
   names only, no values, so saving its output is safe until you edit it.)
3. **Enter the key when prompted**: input is hidden and the key lives in
   memory only for that session.

## Configured endpoints

| Endpoint         | Base URL (overridable)          | Key env                  |
| ---------------- | ------------------------------- | ------------------------ |
| Llama.cpp        | `http://localhost:8080/v1`     | —                        |
| LiteRTM          | `http://localhost:9379/v1`     | —                        |
| UnSloth-Studio   | `http://127.0.0.1:8888/v1`     | `UNSLOTH_STUDIO_API_KEY` |
| LM-Studio        | `http://127.0.0.1:1234/v1`     | —                        |
| Ollama           | `http://localhost:11434/v1`    | —                        |
| OpenAI           | `https://api.openai.com/v1`    | `OPENAI_API_KEY`         |
| Meta-AI          | `https://api.meta.ai/v1`        | `META_API_KEY`           |
| Baseten          | `https://inference.baseten.co/v1` | `BASETEN_API_KEY`      |
| Custom           | *(you provide the URL)*        | *(you provide env var)*  |

Local servers must be running and expose an OpenAI-compatible API. Hosted
providers (OpenAI, Meta, Baseten) expose OpenAI-compatible `/v1` endpoints
(Baseten's shared inference is `https://inference.baseten.co/v1`, i.e. base
URL without the `/chat/completions` path);
override any default with `--endpoint-url`, `LLMTERM_BASE_URL`, or the
interactive Custom entry (dedicated Baseten deployments need their own URL).

Non-interactive examples:

```bash
llmterm --list-endpoints
llmterm --endpoint openai --list-models
llmterm --endpoint baseten --endpoint-url https://your-deployment.api.baseten.co/v1 --model your-model-id
llmterm --endpoint ollama --model llama3.1 --no-save --system-prompt "Be concise."
```

## Architecture

```text
Terminal
   │
   ▼
llmterm
   │
   ▼
OpenAI Python client
   │
   ▼
OpenAI-compatible /v1 endpoint
   │
   ▼
Local/self-hosted LLM
```

### `endpoint.py`

Owns the asynchronous OpenAI-compatible client, model discovery, chat
completion calls, and client cleanup.

### `endpoints.py`

Owns the endpoint registry (local + hosted presets + custom URLs),
URL validation/normalization, and URL-only persistence. Never stores keys.

### `main.py`

Handles endpoint selection, model selection, conversation history, streaming,
and API errors.

### `utils.py`

Handles configuration, system prompts, response cleanup, and Markdown
persistence.

The project intentionally avoids a large agent framework or abstraction layer.
