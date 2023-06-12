"""The main program for nio-llm."""

import asyncio
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download
from jsonargparse import CLI
from rich.logging import RichHandler

from nio_llm.client import LLMClient

logger = logging.getLogger("nio-llm.main")


def main(
    room: str,
    password: str,
    username: str,
    device_id: str,
    preprompt: str,
    ggml_repoid: str = "TheBloke/stable-vicuna-13B-GGML",
    ggml_filename: str = "stable-vicuna-13B.ggmlv3.q5_1.bin",
    homeserver: str = "https://matrix.org",
    sync_timeout: int = 30000,
) -> None:
    """Download llama model from HuggingFace and start the client.

    Args:
        room (`str`):
            The room to join.
        password (`str`):
            The password to log in with.
        username (`str`):
            The username to log in as.
        device_id (`str`):
            The device ID to use.
        preprompt (`str`):
            The preprompt to use.
        ggml_repoid (`str`, default `"TheBloke/stable-vicuna-13B-GGML"`):
            The HuggingFace Hub repo ID to download the model from.
        ggml_filename (`str`, default `"stable-vicuna-13B.ggmlv3.q5_1.bin"`):
            The HuggingFace Hub filename to download the model from.
        homeserver (`str`, default `"matrix.org"`):
            The homeserver to connect to.
        sync_timeout (`int`, default `30000`):
            The timeout to use when syncing with the homeserver.
    """
    # download the model
    ggml_path = Path(
        hf_hub_download(
            repo_id=ggml_repoid,
            filename=ggml_filename,
        ),
    )

    # create the client
    client = LLMClient(
        room=room,
        username=username,
        device_id=device_id,
        ggml_path=ggml_path,
        preprompt=preprompt,
        homeserver=homeserver,
    )

    # start the client
    asyncio.get_event_loop().run_until_complete(
        client.start(
            password=password,
            sync_timeout=sync_timeout,
        ),
    )


if __name__ == "__main__":
    # set up logging
    logging.captureWarnings(True)
    logging.basicConfig(
        level="DEBUG",
        format="%(name)s: %(message)s",
        handlers=[RichHandler(markup=True)],
    )

    # run the main program (with environment variables)
    CLI(main)
