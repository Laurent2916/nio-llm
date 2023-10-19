import asyncio
import logging
import time
from collections import deque

import openai
from nio import AsyncClient, MatrixRoom, RoomMessageText

logger = logging.getLogger("nio-llm.client")


class LLMClient(AsyncClient):
    """A Matrix client that uses llama.cpp to respond to messages."""

    def __init__(
        self,
        username: str,
        homeserver: str,
        device_id: str,
        preprompt: str,
        room: str,
        openai_api_key: str,
        openai_api_endpoint: str,
        openai_temperature: float,
        openai_max_tokens: int,
        history_size: int,
    ) -> None:
        """Create a new LLMClient instance.

        Args:
            username (`str`):
                The username to log in as.
            homeserver (`str`):
                The homeserver to connect to.
            device_id (`str`):
                The device ID to use.
            preprompt (`str`):
                The preprompt to use.
            room (`str`):
                The room to join.
            openai_api_key (`str`):
                The OpenAI API key to use.
            openai_api_endpoint (`str`):
                The OpenAI API endpoint to use.
            openai_temperature (`float`):
                The OpenAI temperature to use.
            openai_max_tokens (`int`):
                The OpenAI max tokens to use.
            history_size (`int`):
                The number of messages to keep in history.
        """
        self.uid = f"@{username}:{homeserver.removeprefix('https://')}"
        self.spawn_time = time.time() * 1000
        self.username = username
        self.preprompt = preprompt
        self.room = room

        # setup openai settings
        openai.api_base = openai_api_endpoint
        openai.api_key = openai_api_key
        self.openai_temperature = openai_temperature
        self.openai_max_tokens = openai_max_tokens

        # create nio AsyncClient instance
        super().__init__(
            user=self.uid,
            homeserver=homeserver,
            device_id=device_id,
        )

        # create message history queue
        self.history: deque[RoomMessageText] = deque(maxlen=history_size)

        # add callbacks
        self.add_event_callback(self.message_callback, RoomMessageText)  # type: ignore

    async def typing_loop(
        self,
        sleep_time: int = 10,
    ) -> None:
        """Send typing indicators every `sleep_time` seconds.

        Args:
            sleep_time (`int`, default `10`):
                The time to sleep between sending typing indicators.
        """
        logging.debug("Started typing indicator.")
        try:
            while True:
                logging.debug("Sending typing indicator.")
                await self.room_typing(self.room, True)
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            await self.room_typing(self.room, False)
            logging.debug("Stopped typing indicator.")

    async def message_callback(
        self,
        room: MatrixRoom,
        event: RoomMessageText,
    ) -> None:
        """Process new messages as they come in.

        Args:
            room (`MatrixRoom`):
                The room the message was sent in.
            event (`RoomMessageText`):
                The message event.
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
            and "rel_type" in event.source["content"]["m.relates_to"]
            and event.source["content"]["m.relates_to"]["rel_type"] == "m.thread"
        ):
            logger.debug("Ignoring thread message.")
            return

        # update history
        self.history.append(event)
        logger.debug(f"Updated history: {self.history}")

        # update read receipt
        await self.room_read_markers(
            room_id=self.room,
            fully_read_event=event.event_id,
            read_event=event.event_id,
        )
        logger.debug(f"Updated read receipt to event: {event.event_id}")

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
            logger.debug("Ignoring message not mentioning us.")
            return

        # start typing indicator loop
        typing_task = asyncio.create_task(self.typing_loop())

        # generate response using llama.cpp
        response = await openai.ChatCompletion.acreate(
            model="local-model",
            messages=[
                {
                    "content": self.preprompt,
                    "role": "system",
                },
                *[
                    {
                        "content": f"{message.sender}: {message.body}",
                        "role": "assistant" if message.sender == self.uid else "user",
                    }
                    for message in self.history
                ],
            ],
            stop=["<|im_end|>"],
            temperature=self.openai_temperature,
            max_tokens=self.openai_max_tokens,
        )
        logger.debug(f"Generated response: {response}")

        # retreive the response
        output = response["choices"][0]["message"]["content"]  # type: ignore
        output = output.strip().removeprefix(f"{self.uid}:").strip()

        # send the response
        await self.room_send(
            room_id=self.room,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                "body": output,
            },
        )
        logger.debug(f"Sent response: {output}")

        # stop typing indicator loop
        typing_task.cancel()

    async def start(
        self,
        password: str,
        sync_timeout: int = 30000,
    ) -> None:
        """Start the client.

        Args:
            password (`str`):
                The password to log in with.
            sync_timeout (`int`, default `30000`):
                The sync timeout in milliseconds.
        """
        # Login to the homeserver
        logger.debug(await self.login(password))

        # Join the room, if not already joined
        logger.debug(await self.join(self.room))

        # Sync with the server forever
        await self.sync_forever(timeout=sync_timeout)
