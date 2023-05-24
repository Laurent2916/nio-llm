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
@click.option("--homeserver", "-h", help="The homeserver to connect to.", required=True)
@click.option("--device-id", "-d", help="The device ID to use.", required=True)
@click.option("--username", "-u", help="The username to log in as.", required=True)
@click.option("--password", "-p", help="The password to log in with.", required=True)
@click.option("--room", "-r", help="The room to join.", required=True)
@click.option("--preprompt", "-t", help="The preprompt to use.", required=True)
def main(
    homeserver: str,
    device_id: str,
    username: str,
    password: str,
    room: str,
    preprompt: str,
) -> None:
    """Run the main program.

    Download the model from HuggingFace Hub and start the async loop.
    """
    # download the model
    ggml_path = Path(
        hf_hub_download(
            repo_id="TheBloke/stable-vicuna-13B-GGML",
            filename="stable-vicuna-13B.ggmlv3.q5_1.bin",
        ),
    )

    # start the async loop
    asyncio.get_event_loop().run_until_complete(
        _main(
            ggml_path=ggml_path,
            homeserver=homeserver,
            device_id=device_id,
            username=username,
            password=password,
            preprompt=preprompt,
            room=room,
        ),
    )


async def _main(
    homeserver: str,
    device_id: str,
    username: str,
    password: str,
    preprompt: str,
    ggml_path: Path,
    room: str,
) -> None:
    """Run the async main program.

    Create the client, login, join the room, and sync forever.
    """
    # create the client
    client = LLMClient(
        homeserver=homeserver,
        device_id=device_id,
        username=username,
        room=room,
        preprompt=preprompt,
        ggml_path=ggml_path,
    )

    # Login to the homeserver
    logger.debug(await client.login(password))

    # Join the room, if not already joined
    logger.debug(await client.join(room))

    # Sync with the server forever
    await client.sync_forever(timeout=30000)


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
