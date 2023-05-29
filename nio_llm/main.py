"""The main program for nio-llm."""

import asyncio
import logging
from pathlib import Path

import click
from huggingface_hub import hf_hub_download
from rich.logging import RichHandler

from nio_llm.client import LLMClient

logger = logging.getLogger("nio-llm.main")


@click.command()
@click.option(
    "--homeserver",
    "-h",
    help="The homeserver to connect to.",
    default="https://matrix.org",
    show_default=True,
)
@click.option(
    "--username",
    "-u",
    help="The username to log in as.",
    required=True,
)
@click.option(
    "--password",
    "-p",
    help="The password to log in with.",
    required=True,
)
@click.option(
    "--room",
    "-r",
    help="The room to join.",
    required=True,
)
@click.option(
    "--device-id",
    "-d",
    help="The device ID to use.",
    default="nio-llm",
    show_default=True,
)
@click.option(
    "--preprompt",
    "-t",
    help="The preprompt to use.",
    required=True,
)
@click.option(
    "--ggml-repoid",
    "-g",
    help="The HuggingFace Hub repo ID to download the model from.",
    default="TheBloke/stable-vicuna-13B-GGML",
    show_default=True,
)
@click.option(
    "--ggml-filename",
    "-f",
    help="The HuggingFace Hub filename to download the model from.",
    default="stable-vicuna-13B.ggmlv3.q5_1.bin",
    show_default=True,
)
@click.option(
    "--sync-timeout",
    "-s",
    help="The timeout to use when syncing with the homeserver.",
    default=30000,
    show_default=True,
)
def main(
    *,
    room: str,
    password: str,
    username: str,
    device_id: str,
    preprompt: str,
    homeserver: str,
    ggml_repoid: str,
    ggml_filename: str,
    sync_timeout: int,
) -> None:
    """Run the main program.

    Download the model from HuggingFace Hub and start the async loop.
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
    main(auto_envvar_prefix="NIOLLM")
