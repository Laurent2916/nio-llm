"""A Matrix client that uses Llama to respond to messages."""

import asyncio
import logging
import time
from textwrap import dedent

import click
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from nio import AsyncClient, MatrixRoom, RoomMessageText

logger = logging.getLogger("nio-llm")


class LLMClient(AsyncClient):
    """A Matrix client that uses Llama to respond to messages."""

    def __init__(
        self,
        username: str,
        homeserver: str,
        device_id: str,
        preprompt: str,
        room: str,
        ggml_path: str,
    ):
        """Create a new LLMClient instance."""
        super().__init__(
            user=f"@{username}:{homeserver.removeprefix('https://')}",
            homeserver=homeserver,
            device_id=device_id,
        )

        self.spawn_time = time.time() * 1000
        self.username = username
        self.preprompt = preprompt
        self.room = room

        # create the Llama instance
        self.llm = Llama(
            model_path=ggml_path,
            n_threads=12,
        )

        # add callbacks
        self.add_event_callback(self.message_callback, RoomMessageText)  # type: ignore

    async def message_callback(self, room: MatrixRoom, event: RoomMessageText):
        """Process new messages as they come in."""
        logger.debug(f"Received new message in room {room.room_id}.")
        logger.debug(f"Message body: {event.body}")

        # ignore our own messages
        if event.sender == self.user:
            logger.debug("Ignoring our own message.")
            return

        # ignore messages pre-spawn
        if event.server_timestamp < self.spawn_time:
            logger.debug("Ignoring message pre-spawn.")
            return

        # ignore messages sent in other rooms
        if room.room_id != self.room:
            logger.debug("Ignoring message in different room.")
            return

        if self.username not in event.body:
            logger.debug("Ignoring message not directed at us.")
            return

        prompt = dedent(
            f"""
            {self.preprompt}
            <{event.sender}>: {event.body}
            <pipobot>:
            """,
        ).strip()

        logger.debug(f"Prompt: {prompt}")

        # enable typing indicator
        await self.room_typing(
            self.room,
            typing_state=True,
            timeout=100000000,
        )

        output = self.llm(
            prompt,
            max_tokens=100,
            stop=[f"<{event.sender}>"],
            echo=True,
        )

        # retreive the response
        output = output["choices"][0]["text"]  # type: ignore
        output = output.removeprefix(prompt).strip()

        # disable typing indicator
        await self.room_typing(self.room, typing_state=False)

        # send the response
        await self.room_send(
            room_id=self.room,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                "body": output,
            },
        )


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
    ggml_path = hf_hub_download(
        repo_id="TheBloke/stable-vicuna-13B-GGML",
        filename="stable-vicuna-13B.ggmlv3.q5_1.bin",
    )

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
    room: str,
    preprompt: str,
    ggml_path: str,
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
    print(await client.login(password))

    # Join the room, if not already joined
    print(await client.join(room))

    # Sync with the server forever
    await client.sync_forever(timeout=30000)


if __name__ == "__main__":
    # set up logging
    logging.basicConfig(level=logging.DEBUG)

    # run the main program (with environment variables)
    main(auto_envvar_prefix="NIOLLM")
