"""The main program for nio-llm."""

import asyncio
import logging

from jsonargparse import CLI
from rich.logging import RichHandler

from nio_llm.client import LLMClient

logger = logging.getLogger("nio-llm.main")


def main(
    room: str,
    username: str,
    password: str,
    preprompt: str = "You are a helpful assistant in a multi-agent conversation. Be as concise as possible.",
    device_id: str = "nio-llm",
    homeserver: str = "https://matrix.org",
    sync_timeout: int = 30000,
    openai_api_key: str = "osftw",
    openai_api_endpoint: str = "http://localhost:8000/v1",
    openai_temperature: float = 0,
    openai_max_tokens: int = 256,
) -> None:
    """Instantiate and start the client.

    Args:
        room (`str`):
            The room to join.
        username (`str`):
            The username to log in as.
        password (`str`):
            The password to log in with.
        preprompt (`str`):
            The preprompt to use.
            Defaults to `"You are a helpful assistant."`.
        device_id (`str`):
            The device ID to use.
            Defaults to `"nio-llm"`.
        homeserver (`str`):
            The matrix homeserver to connect to.
            Defaults to `"https://matrix.org"`.
        sync_timeout (`int`):
            The timeout to use when syncing with the homeserver.
            Defaults to `30000`.
        openai_api_key (`str`):
            The OpenAI API key to use.
            Defaults to `"osftw"`.
        openai_api_endpoint (`str`):
            The OpenAI API endpoint to use.
            Defaults to `"http://localhost:8000/v1"`.
        openai_temperature (`float`):
            The OpenAI temperature to use.
            Defaults to `0`.
        openai_max_tokens (`int`):
            The OpenAI max tokens to use.
            Defaults to `256`.
    """
    # create the client
    client = LLMClient(
        room=room,
        username=username,
        device_id=device_id,
        preprompt=preprompt,
        homeserver=homeserver,
        openai_api_key=openai_api_key,
        openai_api_endpoint=openai_api_endpoint,
        openai_temperature=openai_temperature,
        openai_max_tokens=openai_max_tokens,
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
    CLI(
        components=main,
        as_positional=False,
        env_prefix="NIOLLM",
        default_env=True,
    )
