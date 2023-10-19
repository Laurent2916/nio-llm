# Nio LLM

[![GitHub](https://img.shields.io/github/license/Laurent2916/nio-llm)](https://github.com/Laurent2916/nio-llm/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)

You own little LLM in your matrix chatroom.

## Usage

This project is split in two parts: the client and the server.

The server simply downloads an LLM and starts a llama-cpp-python server (which mimics an openai server).

The client connects to the matrix server and queries the llama-cpp-python server to create matrix messages.

## Special thanks

- https://github.com/abetlen/llama-cpp-python
- https://github.com/ggerganov/llama.cpp/
