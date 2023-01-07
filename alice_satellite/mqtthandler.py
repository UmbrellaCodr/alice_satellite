"""mqtt handler"""
# MIT License

# Copyright (c) 2022 UmbrellaCodr

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import (
    Any,
    Optional,
    Callable,
    List,
    Dict,
    Awaitable,
)
from argparse import Namespace
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
import uuid
import asyncio
import logging
import asyncio_mqtt as aiomqtt
import dataclasses_json
from dataclasses_json import dataclass_json, Undefined, CatchAll, DataClassJsonMixin, LetterCase
from termcolor import cprint
from numpy import ndarray

from .audiohandler import AudioHandler

_log = logging.getLogger("alice.mqtt")


class MessageField:
    """Allowing optional fields in Message classes which don't serialize to json
    """

    def __init__(self, *args, exclude=True, **kwargs) -> None:
        if None in args:
            self.value = None
        else:
            self.value = str(*args)
        self.exclude = exclude

    def __str__(self) -> str:
        return f"{self.value}"

    def __repr__(self) -> str:
        return f"MessageField({self.value})"


def message_field_exclude(param: Any) -> bool:
    """predicate passed to dataclasses_json to handle field filtering

    Args:
        param (_type_): Annoyingly they only pass the values and not the keys

    Returns:
        bool: return True to exclue and False to include
    """
    if isinstance(param, MessageField):
        return param.exclude
    return False


@dataclass
class MessageBase(DataClassJsonMixin, metaclass=ABCMeta):
    dataclass_json_config = dataclasses_json.config(
        letter_case=LetterCase.CAMEL,
        undefined=Undefined.INCLUDE,
        exclude=message_field_exclude)['dataclasses_json']

    @abstractmethod
    def topic(self) -> Optional[str]:
        pass

    def peak(self, message: aiomqtt.Message) -> bool:
        return False

    def waitfor(self) -> Any:
        return None


@dataclass
class MessageTTSFinished(MessageBase):
    site_id: Optional[str] = None
    id: Optional[str] = None
    session_id: Optional[str] = None
    unknown: CatchAll = None

    def __post_init__(self):
        super().__init__()

    def topic(self) -> Optional[str]:
        return "hermes/tts/sayFinished"


@dataclass
class MessageTTS(MessageBase):
    text: Optional[str] = None
    site_id: Optional[str] = None
    lang: Optional[str] = None
    id: Optional[str] = None
    session_id: Optional[str] = None
    volume: Optional[float] = 1.0
    unknown: CatchAll = None

    def __post_init__(self):
        super().__init__()
        self.id = str(uuid.uuid4())

    def topic(self) -> Optional[str]:
        return "hermes/tts/say"

    def peak(self, message: aiomqtt.Message) -> bool:
        response = MessageTTSFinished().from_json(message.payload)
        return self.id == response.id

    def waitfor(self) -> Any:
        return MessageTTSFinished()


@dataclass
class MessageAsrStart(MessageBase):
    site_id: Optional[str] = None
    session_id: Optional[str] = None
    lang: Optional[str] = None

    # rhasspy settings
    stop_on_silence: bool = True
    send_audio_captured: bool = False
    wakeword_id: Optional[str] = None
    intent_filter: Optional[List[str]] = None
    unknown: CatchAll = None

    def topic(self) -> Optional[str]:
        return "hermes/asr/startListening"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class AsrTokenTime:
    """The time when an ASR token was detected."""

    start: float
    """Start time (in seconds) of token relative to beginning of utterance."""
    end: float
    """End time (in seconds) of token relative to beginning of utterance."""


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class AsrToken:
    """A token from an automated speech recognizer."""

    value: str
    """Text value of the token."""
    confidence: float
    """Confidence score of the token, between 0 and 1 (1 being confident)."""
    range_start: int
    """The start of the range in which the token is in the original input."""
    range_end: int
    """The end of the range in which the token is in the original input."""
    time: Optional[AsrTokenTime] = None
    """Structured time when this token was detected."""


@dataclass
class MessageAsrText(MessageBase):
    text: Optional[str] = None
    likelihood: Optional[float] = 0
    seconds: Optional[float] = 0

    site_id: Optional[str] = None
    session_id: Optional[str] = None

    # rhasspy
    wakeword_id: Optional[str] = None
    asr_tokens: Optional[List[List[AsrToken]]] = None
    lang: Optional[str] = None
    unknown: CatchAll = None

    def topic(self) -> Optional[str]:
        return "hermes/asr/textCaptured"


@dataclass
class MessageAsrStop(MessageBase):
    site_id: Optional[str] = None
    session_id: Optional[str] = None
    unknown: CatchAll = None

    def topic(self) -> Optional[str]:
        return "hermes/asr/stopListening"

    def peak(self, message: aiomqtt.Message) -> bool:
        response = MessageAsrText().from_json(message.payload)
        _log.debug("MessageAsrStop %s -> %s",
                   self.session_id, response.session_id)
        return self.session_id == response.session_id

    def waitfor(self) -> Any:
        return MessageAsrText()


@dataclass
class MessagePlayFinished(MessageBase):
    site_id: Optional[MessageField] = None
    id: Optional[str] = None
    session_id: Optional[str] = None
    unknown: CatchAll = None

    def __post_init__(self):
        super().__init__()
        # handle passing strings for site_id
        if not isinstance(self.site_id, MessageField):
            self.site_id = MessageField(self.site_id)

    def topic(self) -> Optional[str]:
        return f"hermes/audioServer/{self.site_id}/playFinished"


@dataclass
class MessageHotwordDetected(MessageBase):
    model_id: str = None
    model_version: str = ""
    model_type: str = "personal"
    current_sensitivity: float = 1.0
    site_id: Optional[str] = None

    # rhasspy
    session_id: Optional[str] = None
    send_audio_captured: Optional[bool] = None
    lang: Optional[str] = None
    custom_entities: Optional[Dict[str, Any]] = None
    unknown: CatchAll = None

    def topic(self) -> Optional[str]:
        return f"hermes/hotword/{self.model_id}/detected"


@dataclass
class MessageHandlerEvent:
    topic: str
    peak: Callable[[aiomqtt.Message], bool]
    queue: asyncio.Queue = None

    def __post_init__(self):
        if not self.queue:
            self.queue = asyncio.Queue()


class MessageHandler():
    def __init__(self, config: Namespace) -> None:
        self.mqtt_settings: dict = vars(config.mqtt)
        self.mqtt_client: aiomqtt.Client = None
        self.mqtt_initialized: asyncio.Event = asyncio.Event()
        self.mqtt_events: list[MessageHandlerEvent] = []
        self.debug = config.debug
        self.subscribed: list[str] = []

    async def task(self, callback: Callable[[aiomqtt.Message], Awaitable[None]] = None, reconnect_interval: int = 5):
        while True:
            try:
                async with aiomqtt.Client(**self.mqtt_settings) as client:
                    self.mqtt_client = client
                    self.mqtt_initialized.set()
                    _log.info("starting connection to %s", self.mqtt_settings['hostname'])
                    for topic in self.subscribed:
                        await self.mqtt_client.subscribe(topic)
                    async with self.mqtt_client.messages() as messages:
                        async for message in messages:
                            if self.debug:
                                cprint(message.topic, "magenta")
                            if callback:
                                await callback(message)
                            for event in [x for x in self.mqtt_events if message.topic.matches(x.topic)]:
                                if event.peak(message):
                                    await event.queue.put(message)
            except aiomqtt.MqttError as error:
                self.mqtt_initialized.clear()
                _log.error(
                    'Error %s. Reconnecting in %i seconds.', error, reconnect_interval)
                await asyncio.sleep(reconnect_interval)

    def subscribe(self, topic:str):
        self.subscribed.append(topic)

    def unsubscribe(self, topic:str):
        self.subscribed.remove(topic)

    async def wait(self):
        # wait a moment for the handler to initialize
        if not self.mqtt_initialized.is_set():
            _log.debug("waiting for mqtt connection")
            async def should_initialize(event: asyncio.Event):
                await event.wait()
            wait_task = asyncio.create_task(should_initialize(self.mqtt_initialized))
            await asyncio.wait_for(wait_task, 5)
            _log.debug("mqtt initialized %i", self.mqtt_initialized.is_set())
            if not self.mqtt_initialized.is_set():
                raise TimeoutError("timeout waiting for client to initialize")

    async def send_message(self, message: MessageBase) -> aiomqtt.Message:
        await self.wait()

        # process message
        event: MessageHandlerEvent = None
        result: aiomqtt.Message = None
        response_topic = None
        try:
            if message.waitfor():
                response_topic = message.waitfor().topic()
            if response_topic:
                event = MessageHandlerEvent(response_topic, message.peak)
                self.mqtt_events.append(event)
                await self.mqtt_client.subscribe(response_topic)

            await self.mqtt_client.publish(message.topic(), message.to_json())

            if response_topic:
                get_await = event.queue.get()
                result = await asyncio.wait_for(get_await, 5)
        except asyncio.TimeoutError:
            _log.error('gave up waiting on %s', response_topic)
        finally:
            if response_topic:
                self.mqtt_events.remove(event)
                await self.mqtt_client.unsubscribe(response_topic)

        return result

    async def hotword_detected(self, site_id: str) -> None:
        model_id = "Alice"
        await self.send_message(MessageHotwordDetected(model_id=model_id, site_id=site_id))


    async def transcribe_audio(self, wavform: ndarray, site_id: str, session_id: str = None) -> MessageAsrText:
        chunk_topic = f"hermes/audioServer/{site_id}/audioFrame"
        if session_id is None:
            session_id = str(uuid.uuid4())

        await self.send_message(MessageAsrStart(
            site_id=site_id, session_id=session_id))
        for chunk in AudioHandler.chunk_audio(wavform):
            await self.mqtt_client.publish(chunk_topic, payload=chunk)
        result = await self.send_message(
            MessageAsrStop(site_id=site_id, session_id=session_id))

        asr_msg = MessageAsrText.from_json(result.payload)

        return asr_msg.to_dict()

    async def play_finished(self, site_id: str, session_id: str):
        await self.send_message(MessagePlayFinished(site_id=site_id,
                                                    id=session_id, session_id=session_id))

    async def tts(self, text: str, site_id: str) -> bytes:
        await self.wait()
        
        msg_tts = MessageTTS(text=text, site_id=site_id)
        audio_topic = f"hermes/audioServer/{site_id}/playBytes/{msg_tts.id}"
        try:
            event: MessageHandlerEvent = MessageHandlerEvent(audio_topic, lambda x:True)
            self.mqtt_events.append(event)
            await self.mqtt_client.subscribe(audio_topic)
            await self.send_message(msg_tts)

            get_await = event.queue.get()
            result = await asyncio.wait_for(get_await, 5)
        except asyncio.TimeoutError:
            _log.error('gave up waiting on %s', audio_topic)
        finally:
            self.mqtt_events.remove(event)
            await self.mqtt_client.unsubscribe(audio_topic)

        return result.payload if result else None
        