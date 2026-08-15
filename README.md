# llmterm

**A minimal terminal client for OpenAI-compatible LLM servers.**

`llmterm` is a small Python command-line application for interacting with
local or self-hosted LLM servers that expose an OpenAI-compatible API.

## Features

* Interactive endpoint selection
* Automatic model discovery
* Interactive model selection
* Asynchronous streaming responses
* Conversation history
* Optional custom system prompt
* Optional Markdown conversation logging
* `.env` support for API keys
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
        ├── main.py
        └── utils.py
```

`src/` contains only package source code. Runtime-generated files are not
stored inside the Python package.

## Markdown output location

When Markdown saving is enabled, responses are stored in:

```text
~/.llmterm/
└── markdown-outputs/
    ├── model-name-08-15-2026.md
    └── another-model-08-15-2026.md
```

This location is independent of the directory from which `llmterm` is run.

Keeping generated data outside the package is important for a distributable
Python package because installed package directories should not be assumed
to be writable.

## Installation

### Using uv

Install `llmterm-prasannaba` as a standalone CLI tool:

```bash
uv tool install llmterm-prasannaba
```

Run:

```bash
llmterm
```

### Using pip

You can also install `llmterm-prasannaba` using pip:

```bash
python -m pip install llmterm-prasannaba
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

## Optional API key

Create a `.env` file in the project directory when Unsloth Studio requires
an API key:

```env
UNSLOTH_STUDIO_API_KEY=your-key
```

## Configured endpoints

| Endpoint         | Base URL                    |
| ---------------- | --------------------------- |
| Llama.cpp        | `http://localhost:8080/v1`  |
| Google-Litert-LM | `http://localhost:9379/v1`  |
| UnSloth-Studio   | `http://127.0.0.1:8888/v1`  |
| LM-Studio        | `http://127.0.0.1:1234/v1`  |
| Ollama           | `http://localhost:11434/v1` |

The corresponding server must be running and expose an OpenAI-compatible API.

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

### `main.py`

Handles endpoint selection, model selection, conversation history, streaming,
and API errors.

### `utils.py`

Handles configuration, system prompts, response cleanup, and Markdown
persistence.

The project intentionally avoids a large agent framework or abstraction layer.
