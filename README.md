# Nio LLM

[![GitHub](https://img.shields.io/github/license/Laurent2916/nio-llm)](https://github.com/Laurent2916/nio-llm/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)

You own little LLM in your matrix chatroom.

## Installation

```bash
pip install git+https://github.com/Laurent2916/nio-llm.git
```

## Usage

This project uses [jsonargparse](https://github.com/omni-us/jsonargparse/) to help with the command line arguments.

To see the available options, run:

```bash
nio_llm --help
```

To run the bot, you can either use command line arguments, environment variables or a config file. (or a mix of all three)

### Command line arguments

```bash
nio_llm \
  # required \
  --room <YOUR ROOM> \
  --password <YOUR PASSWORD> \
  --username <YOUR USERNAME> \
  --preprompt <YOUR PREPROMPT> \
  # optional \
  --device-id nio-llm \
  --homeserver https://matrix.org \
  --ggml-repoid TheBloke/stable-vicuna-13B-GGML \
  --ggml-filename stable-vicuna-13B.ggmlv3.q5_1.bin \
  --sync-timeout 30000
```

### Environment variables

```bash
# required
export NIO_LLM_ROOM=<YOUR ROOM>
export NIO_LLM_PASSWORD=<YOUR PASSWORD>
export NIO_LLM_USERNAME=<YOUR USERNAME>
export NIO_LLM_PREPROMPT=<YOUR PREPROMPT>

# optional
export NIO_LLM_DEVICE_ID=nio-llm
export NIO_LLM_HOMESERVER=https://matrix.org
export NIO_LLM_GGML_REPOID=TheBloke/stable-vicuna-13B-GGML
export NIO_LLM_GGML_FILENAME=stable-vicuna-13B.ggmlv3.q5_1.bin
export NIO_LLM_SYNC_TIMEOUT=30000

nio_llm
```


### Config file

Create a config file with the following content:

```yaml
# config_file.yaml

# required
room: <YOUR ROOM>
password: <YOUR PASSWORD>
username: <YOUR USERNAME>
preprompt: <YOUR PREPROMPT>

# optional
device_id: nio-llm
homeserver: https://matrix.org
ggml_repoid: TheBloke/stable-vicuna-13B-GGML
ggml_filename: stable-vicuna-13B.ggmlv3.q5_1.bin
sync_timeout: 30000
```

Then run:

```bash
nio_llm --config config_file.yaml
```

## Special thanks

- https://github.com/abetlen/llama-cpp-python
- https://github.com/ggerganov/llama.cpp/
