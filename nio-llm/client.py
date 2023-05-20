"""A Matrix client that uses Llama to respond to messages."""

import asyncio
import logging
from textwrap import dedent

from llama_cpp import Llama
from nio import AsyncClient, MatrixRoom, RoomMessageText

logger = logging.getLogger("nio-llm")


class LLMClient(AsyncClient):
    """A Matrix client that uses Llama to respond to messages."""

    def __init__(
        self,
        user: str,
        homeserver: str,
        device_id: str,
    ):
        """Create a new LLMClient instance."""
        super().__init__(
            user=user,
            homeserver=homeserver,
            device_id=device_id,
        )

        # create the Llama instance
        self.llm = Llama(
            model_path="../../../llama.cpp/models/sv13B/stable-vicuna-13B.ggml.q5_1.bin",
            n_threads=12,
        )

        # add callbacks
        self.add_event_callback(self.message_callback, RoomMessageText)  # type: ignore

    async def message_callback(self, room: MatrixRoom, event: RoomMessageText):
        """Process new messages as they come in."""
        # ignore messages sent in other rooms
        if room.room_id != ROOM:
            return

        if f"<{USERNAME}>" in event.body:
            logging.debug("Received message including our identifier")

            prompt = dedent(
                f"""
                {PREPROMPT}
                <{event.sender}>: {event.body}
                <{USERNAME}>:
                """,
            ).strip()

            # enable typing indicator
            await self.room_typing(ROOM, typing_state=True)

            output = self.llm(
                prompt,
                max_tokens=100,
                stop=["<{event.sender}>:", "\n"],
                echo=True,
            )

            # retreive the response
            output = output["choices"][0]["text"]  # type: ignore
            output = output.removeprefix(prompt).strip()

            # disable typing indicator
            await self.room_typing(ROOM, typing_state=False)

            # send the response
            await self.room_send(
                room_id=ROOM,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": output,
                },
            )


async def main() -> None:
    """Run the main program."""
    # create the client
    client = LLMClient(
        homeserver=HOMESERVER,
        device_id=DEVICE_ID,
        user=USERNAME,
    )

    # Login to the homeserver
    print(await client.login(PASSWORD))

    # Join the room, if not already joined
    print(await client.join(ROOM))

    # Sync with the server forever
    await client.sync_forever(timeout=30000)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
