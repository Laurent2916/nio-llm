"""A Matrix client that uses Llama to respond to messages."""

import logging
import time
from collections import deque
from pathlib import Path

from llama_cpp import Llama
from nio import AsyncClient, MatrixRoom, RoomMessageText

logger = logging.getLogger("nio-llm.client")


class LLMClient(AsyncClient):
    """A Matrix client that uses Llama to respond to messages."""

    def __init__(
        self,
        username: str,
        homeserver: str,
        device_id: str,
        preprompt: str,
        ggml_path: Path,
        room: str,
    ):
        """Create a new LLMClient instance.

        Args:
            username (`str`): The username to log in as.
            homeserver (`str`): The homeserver to connect to.
            device_id (`str`): The device ID to use.
            preprompt (`str`): The preprompt to use.
            ggml_path (`Path`): The path to the GGML model.
            room (`str`): The room to join.
        """
        self.uid = f"@{username}:{homeserver.removeprefix('https://')}"
        self.spawn_time = time.time() * 1000
        self.username = username
        self.preprompt = preprompt
        self.room = room

        # create the AsyncClient instance
        super().__init__(
            user=self.uid,
            homeserver=homeserver,
            device_id=device_id,
        )

        # create the Llama instance
        self.llm = Llama(
            model_path=str(ggml_path),
            n_threads=12,
            n_ctx=512 + 128,
        )

        # create message history queue
        self.history: deque[RoomMessageText] = deque(maxlen=10)

        # add callbacks
        self.add_event_callback(self.message_callback, RoomMessageText)  # type: ignore

    async def message_callback(self, room: MatrixRoom, event: RoomMessageText) -> None:
        """Process new messages as they come in.

        Args:
            room (`MatrixRoom`): The room the message was sent in.
            event (`RoomMessageText`): The message event.
        """
        logger.debug(f"New RoomMessageText: {event.source}")

        # ignore messages pre-dating our spawn time
        if event.server_timestamp < self.spawn_time:
            logger.debug("Ignoring message pre-spawn.")
            return

        # ignore messages not in our monitored room
        if room.room_id != self.room:
            logger.debug("Ignoring message in different room.")
            return

        # ignore edited messages
        if "m.new_content" in event.source["content"]:
            logger.debug("Ignoring edited message.")
            return

        # ignore thread messages
        if (
            "m.relates_to" in event.source["content"]
            and event.source["content"]["m.relates_to"]["rel_type"] == "m.thread"
        ):
            logger.debug("Ignoring thread message.")
            return

        # update history
        self.history.append(event)

        # ignore our own messages
        if event.sender == self.user:
            logger.debug("Ignoring our own message.")
            return

        # ignore messages not mentioning us
        if not (
            "format" in event.source["content"]
            and "formatted_body" in event.source["content"]
            and event.source["content"]["format"] == "org.matrix.custom.html"
            and f'<a href="https://matrix.to/#/{self.uid}">{self.username}</a>'
            in event.source["content"]["formatted_body"]
        ):
            logger.debug("Ignoring message not directed at us.")
            return

        # generate prompt from message and history
        history = "\n".join(f"<{message.sender}>: {message.body}" for message in self.history)
        prompt = "\n".join([self.preprompt, history, f"<{self.uid}>:"])
        tokens = self.llm.tokenize(str.encode(prompt))
        logger.debug(f"Prompt:\n{prompt}")
        logger.debug(f"Tokens: {len(tokens)}")

        # ignore prompts that are too long
        if len(tokens) > 512:
            logger.debug("Prompt too long, skipping.")
            await self.room_send(
                room_id=self.room,
                message_type="m.room.message",
                content={
                    "msgtype": "m.emote",
                    "body": "reached prompt token limit",
                },
            )
            return

        # enable typing indicator
        await self.room_typing(
            self.room,
            typing_state=True,
            timeout=100000000,
        )

        # generate response using llama.cpp
        senders = [f"<{message.sender}>" for message in self.history]
        output = self.llm(
            prompt,
            max_tokens=128,
            stop=[f"<{self.uid}>", "### Human", "### Assistant", *senders],
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

    async def start(self, password, sync_timeout=30000):
        """Start the client.

        Args:
            password (`str`): The password to log in with.
            sync_timeout (`int`, default `30000`): The sync timeout in milliseconds.
        """
        # Login to the homeserver
        logger.debug(await self.login(password))

        # Join the room, if not already joined
        logger.debug(await self.join(self.room))

        # Sync with the server forever
        await self.sync_forever(timeout=sync_timeout)
