"""Alice main program"""
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

import sys
import os
import argparse
import logging
import asyncio
import time
import json
import numpy as np
from termcolor import colored, cprint
import sounddevice as sd
import asyncio_mqtt as aiomqtt
import noisereduce as nr

from . import ALICE_MODULE_PATH
from .audiohandler import AudioHandler
from .mqtthandler import MessageHandler, MessageAsrStart, MessageAsrStop
from .config import load_config, save_config
from .generate import analyze, stitch, listen, detect, verify, generate, morph
from .generate import get_kw, get_kw_samples
from .utils import alice_model, download_model

_log = logging.getLogger("alice")


async def list_audio_devices(config: argparse.Namespace, **kwargs) -> None:
    device_info = sd.query_devices(device=None)
    print(device_info)


async def train_model(config: argparse.Namespace, **kwargs) -> None:
    import alice_satellite.model.kws as kws

    model_handler = kws.AliceKWSTrain(config, training=True)
    model_handler.train()


async def predict(config: argparse.Namespace, **kwargs) -> None:
    import alice_satellite.model.kwslite as kws

    model_handler = kws.AliceKWS(config)
    audio_handler = AudioHandler(config, training=True, **kwargs)
    class_id, label, prob, perf = model_handler.predict(
        audio_handler.load_audio(config.file))
    print("id: {} label: {} ({:4.1f}%) time: {:.3f}ms".format(
        class_id, label, prob, perf))


async def info(config: argparse.Namespace, **kwargs) -> None:
    """info"""

    print("module path: {}".format(ALICE_MODULE_PATH))
    index_to_label = {}
    try:
        labels_file = os.path.join(config.alice.model_path,  "labels.txt")
        with open(labels_file, encoding="utf-8") as fd:
            index_to_label = fd.read().split('\n')
    except FileNotFoundError:
        pass

    if len(index_to_label) > 0:
        print("Model loaded with the following labels:")
        for i, v in enumerate(index_to_label):
            print("{}:{};".format(
                colored(i, "blue", attrs=['bold']), v), end="")
        print("")
    else:
        print("no model has been trained yet")

    if os.path.exists(config.alice.samples_path):
        print("keyword samples:")
        wanted_words = ""
        for k in get_kw(config):
            k = str(k)
            if k.lower() == "_background_noise_":
                continue
            total = 0
            morphed = 0
            for f, _, _ in get_kw_samples(config, keyword=k):
                total += 1
                if f.startswith("m"):
                    morphed += 1
            print("  {:>12} -> original: {:>6} + morphed({:>6}) == total: {:>6}".format(k,
                                                                                        int(total-morphed), morphed, total))
            if len(wanted_words) > 0:
                wanted_words += f",{k}"
            else:
                wanted_words = f"{k}"
        print(f"wanted_words: {wanted_words}")
    else:
        print("no samples")

    cprint("audio devices:", "red")
    try:
        device_info = sd.query_devices(device=config.audio_input, kind="input")
        print("IN:")
        print(device_info)

        device_info = sd.query_devices(
            device=config.audio_output, kind="output")
        print("OUT:")
        print(device_info)
    except ValueError:
        print("failed to get audio information")


def satellite_verify(config: argparse.Namespace) -> bool:
    if not os.path.exists(config.alice.model_path):
        cprint("no model found, exiting", "red")
        return False

    if not hasattr(config, 'mqtt'):
        cprint("no mqtt server configured", "yellow", attrs=['bold'])

    return True


async def satellite(config: argparse.Namespace, **kwargs) -> None:
    import alice_satellite.model.kwslite as kws
    if config.whisper:
        import whisper
        wm = whisper.load_model(config.whisper_model)
    if not satellite_verify(config):
        cprint("uh-oh", "magenta")
        return

    tasks = []
    model_handler = kws.AliceKWS(config)
    audio_handler = AudioHandler(config, verbose=True)
    audio_handler.connect()
    mqtt_handler = None
    if hasattr(config, 'mqtt'):
        mqtt_handler = MessageHandler(config)
    kw_map = dict()
    if config.index:
        _log.info("monitoring the following: {}".format(config.index))
        for k in config.index:
            kw_map[int(k)] = 0
    detected = False
    listening = False
    play_topic = f"hermes/audioServer/{config.site_id}/playBytes/#"
    start_topic = "hermes/asr/startListening"
    stop_topic = "hermes/asr/stopListening"
    frame_topic = f"hermes/audioServer/{config.site_id}/audioFrame"
    if mqtt_handler:
        mqtt_handler.subscribe(play_topic)

    async def msg_callback(message: aiomqtt.Message):
        nonlocal listening
        _log.debug("handle message topic: %s listen: %i",
                   message.topic, listening)
        if message.topic.matches(f"hermes/audioServer/{config.site_id}/playBytes/#"):
            if config.verbose:
                cprint("P", "white", attrs=['bold'], flush=True, end='')
            audio_handler.stop()
            audio_handler.play_bytes(message.payload)
            session_id = os.path.basename(str(message.topic))
            await mqtt_handler.play_finished(config.site_id, session_id)
            audio_handler.start()
        elif message.topic.matches(start_topic):
            payload = MessageAsrStart.from_json(message.payload)
            if payload.site_id == config.site_id:
                if config.verbose:
                    cprint("S", "red", "on_white",
                           attrs=['bold'], flush=True, end='')
                listening = True
        elif message.topic.matches(stop_topic):
            payload = MessageAsrStop.from_json(message.payload)
            if payload.site_id == config.site_id:
                if config.verbose:
                    cprint("S", "red", "on_white",
                           attrs=['bold'], flush=True, end='')
                listening = False

    async def audio_task():
        nonlocal detected, listening
        window = np.zeros(0)
        window_offset = 0
        mode_deadline = 0
        async with audio_handler.audio() as chunks:
            async for audio, noise in chunks:
                if detected:
                    if config.verbose:
                        cprint("+", "blue",
                               attrs=['bold'], flush=True, end='')
                    window = np.concatenate(
                        (window, audio[-audio_handler.window_size:]))
                    # buffer 2 seconds of audio packets
                    if window.size > (audio_handler.params.desired_sample_rate*2):
                        if listening:
                            send_chunk = window[window_offset:]
                            window_offset = window.size
                            send_chunk = nr.reduce_noise(
                                y=send_chunk, sr=audio_handler.params.desired_sample_rate)
                            for chunk in AudioHandler.chunk_audio(send_chunk):
                                await mqtt_handler.mqtt_client.publish(frame_topic, payload=chunk)
                        if mqtt_handler and not listening:
                            detected = False
                        # work without a mqtt handler
                        if not mqtt_handler and not noise:
                            detected = False
                    mode_now = time.time()
                    if mode_now > mode_deadline:
                        _log.error(
                            "timeout waiting for StopListen; detected %i; listening: %i", detected, listening)
                        detected = False
                        listening = False
                    if not detected:
                        if mode_now > mode_deadline and config.verbose:
                            cprint("timeout", "magenta", flush=True, end='')
                        if config.whisper:
                            result = wm.transcribe(window.astype(
                                dtype=np.float32), fp16=False)
                            print('{}'.format(result['text']))
                        if mqtt_handler:
                            await mqtt_handler.mqtt_client.unsubscribe(start_topic)
                            await mqtt_handler.mqtt_client.unsubscribe(stop_topic)
                elif noise:
                    now = time.time_ns() - (1e9 * 3)  # allow 3 second window
                    p_color = "white"
                    class_id, label, prob, _ = model_handler.predict(audio)
                    _log.debug("id: %i, label: %s, prob: %4.1f",
                               class_id, label, prob)
                    if class_id == config.match:
                        p_color = "yellow"
                        if prob > config.threshold:
                            p_color = "green"
                            detected = True
                    if class_id in config.index:
                        p_color = "yellow"
                        if prob > config.threshold:
                            p_color = "green"
                            kw_map[class_id] = time.time_ns()
                        for v in kw_map.values():
                            if v > now:
                                detected = True
                            else:
                                detected = False
                                break
                    if config.verbose:
                        cprint("*", p_color, flush=True, end='')

                    if detected:
                        window = np.zeros(0)
                        window_offset = 0
                        mode_deadline = time.time() + 10
                        if mqtt_handler:
                            await mqtt_handler.mqtt_client.subscribe(start_topic)
                            await mqtt_handler.mqtt_client.subscribe(stop_topic)
                            await mqtt_handler.hotword_detected(site_id=config.site_id)
                        if config.verbose:
                            cprint("detected", "magenta",
                                   flush=True, end='')
                        else:
                            _log.info("wake word detected")
                else:
                    if config.verbose:
                        print(".", flush=True, end='')

    if mqtt_handler:
        tasks.append(asyncio.ensure_future(
            mqtt_handler.task(callback=msg_callback)))
    tasks.append(asyncio.create_task(audio_task()))

    _log.debug("waiting for tasks")
    while tasks:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        _log.debug("%i remaining tasks", len(pending))
        x: asyncio.Task
        for x in done:
            result = x.exception()
            if result:
                _log.error("Finished task produced", exc_info=result)
        tasks = pending

    _log.debug("completed")


async def mqtt(config: argparse.Namespace, **kwargs) -> None:
    if hasattr(config, 'mqtt'):
        _log.debug(config.mqtt)
        tasks = []
        config.debug = True
        mqtt_handler = MessageHandler(config)
        mqtt_handler.subscribe(config.topic)

        async def msg_callback(message: aiomqtt.Message):
            try:
                json.dumps(message.payload.decode('utf-8'))
                print(message.payload)
            except UnicodeDecodeError:
                cprint('binary', 'red')

        tasks.append(asyncio.create_task(
            mqtt_handler.task(callback=msg_callback)))

        while tasks:
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            _log.debug("%i remaining tasks", len(pending))
            tasks = pending

        _log.debug("completed")
    else:
        _log.error("mqtt not configured")


async def tts(config: argparse.Namespace, **kwargs) -> None:
    if not hasattr(config, 'mqtt'):
        print("this requires rhasspy mqtt configured")
        sys.exit(0)
    mqtt_handler = MessageHandler(config)
    audio_handler = AudioHandler(config)

    asyncio.ensure_future(mqtt_handler.task())
    result = await mqtt_handler.tts(config.text, config.site_id)
    print(f"completed {len(result)}")
    if config.play:
        audio_handler.play_bytes(result, True)
    if config.file:
        audio_handler.save_bytes(result, config.file)


async def transcribe(config: argparse.Namespace, **kwargs) -> None:
    if not hasattr(config, 'mqtt'):
        print("this requires rhasspy mqtt configured")
        sys.exit(0)
    mqtt_handler = MessageHandler(config)
    audio_handler = AudioHandler(config)

    audio = audio_handler.load_audio(config.file)
    asyncio.ensure_future(mqtt_handler.task())
    result = await mqtt_handler.transcribe_audio(audio, config.site_id)
    print(f"result: {result['text']}")


async def do_something(config: argparse.Namespace, **kwargs) -> None:
    print("try python -m alice_satellite satellite")


async def main(**kwargs) -> None:
    """main alice entry point"""
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    _log.addHandler(handler)
    _log.setLevel(logging.INFO)

    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser(prog="alice")
    parser.set_defaults(func=do_something)
    subparsers = parser.add_subparsers(help='supported sub commands')

    parser_list = subparsers.add_parser('list', help='list audio devices')
    parser_list.set_defaults(func=list_audio_devices)

    parser_config = subparsers.add_parser('config', help='save a config file')
    parser_config.set_defaults(func=save_config)

    parser_analyze = subparsers.add_parser('analyze', help='morph samples')
    parser_analyze.set_defaults(func=analyze)
    parser_analyze.add_argument(
        "--all", action="store_true", help="process all files (includes morphed)"
    )
    parser_analyze.add_argument(
        "keywords", type=str, nargs='+', help="a list of keywords to operate on"
    )
    parser_analyze.add_argument(
        "--prompt", action="store_true", help="prompt what to do on mis-match"
    )
    parser_analyze.add_argument(
        "--purge", action="store_true", help="auto delete mis-matched audio"
    )
    parser_analyze.add_argument(
        "-o", "--offset", type=str, help="start file iteration at offset",
    )
    parser_analyze.add_argument(
        "-w", "--whisper", action="store_true", help="enable whisper transcription",
        default=True
    )
    parser_analyze.add_argument(
        "-m", "--match", type=str, nargs='+', help="alternate text to match on"
    )

    parser_listen = subparsers.add_parser(
        'listen', help='requires a trained model')
    parser_listen.set_defaults(func=listen)
    parser_listen.add_argument(
        "--save", action="store_true", help="save samples in <data>/samples/listen"
    )
    parser_listen.add_argument(
        "-i", "--ignore", nargs="+", help="don't save ignored ids",
        default=()
    )
    parser_listen.add_argument(
        "--db", type=int, help="db adjuster for word splitting"
    )

    parser_detect = subparsers.add_parser(
        'detect', help='requires a trained model')
    parser_detect.set_defaults(func=detect)
    parser_detect.add_argument(
        "-i", "--index", type=int, nargs="+", help="index of words to required for wake word detection",
        default=()
    )
    parser_detect.add_argument(
        "-m", "--match", type=int, help="single index to match for wake word detection"
    )
    parser_detect.add_argument(
        "--db", type=int, help="db adjuster for word splitting"
    )
    parser_detect.add_argument(
        "-w", "--whisper", action="store_true", help="enable whisper transcription",
        default=False
    )

    parser_satellite = subparsers.add_parser(
        'satellite', help='main mode, listens for wake word communicates to rhasspy')
    parser_satellite.set_defaults(func=satellite)
    parser_satellite.add_argument(
        "-i", "--index", type=int, nargs="+", help="index of words to required for wake word detection",
        default=()
    )
    parser_satellite.add_argument(
        "-m", "--match", type=int, help="single index to match for wake word detection"
    )
    parser_satellite.add_argument(
        "-t", "--threshold", type=int, help="threshold for keyword match",
        default=95
    )
    parser_satellite.add_argument(
        "-w", "--whisper", action="store_true", help="enable whisper transcription",
        default=False
    )

    parser_train = subparsers.add_parser('train', help='train the model')
    parser_train.set_defaults(func=train_model)
    parser_train.add_argument(
        "--create", action="store_true", help="create a new model"
    )

    parser_predict = subparsers.add_parser('predict', help='classify wav file')
    parser_predict.set_defaults(func=predict)
    parser_predict.add_argument(
        "file", help="wav file to run against the model"
    )

    parser_generate = subparsers.add_parser(
        'generate', help='generate samples')
    parser_generate.set_defaults(func=generate)
    parser_generate.add_argument(
        "keywords", type=str, nargs='+', help="a list of keywords to generate samples for"
    )
    parser_generate.add_argument(
        "-c", "--count", type=int, help="how many samples for each class is required",
        default=100
    )
    parser_generate.add_argument(
        "--db", type=int, help="db adjuster for word splitting"
    )

    parser_morph = subparsers.add_parser('morph', help='morph samples')
    parser_morph.set_defaults(func=morph)
    parser_morph.add_argument(
        "keywords", type=str, nargs='+', help="a list of keywords to morph samples for"
    )
    parser_morph.add_argument(
        "-c", "--count", type=int, help="generate 20 samples by shifting padding",
        default=10
    )
    parser_morph.add_argument(
        "--db", type=int, help="db adjuster for word splitting"
    )

    parser_verify = subparsers.add_parser('verify', help='verify samples')
    parser_verify.set_defaults(func=verify)
    parser_verify.add_argument(
        "--all", action="store_true", help="process all files (includes morphed)"
    )
    parser_verify.add_argument(
        "keywords", type=str, nargs='*', help="a list of keywords to operate on"
    )
    parser_verify.add_argument(
        "-a", "--auto", action="store_true", help="auto play audio",
        default=False
    )
    parser_verify.add_argument(
        "-s", "--skip", action="store_true", help="skip audio",
        default=False
    )
    parser_verify.add_argument(
        "-o", "--offset", type=str, help="start file iteration at offset"
    )
    parser_verify.add_argument(
        "--db", type=int, help="db adjuster for word splitting"
    )
    parser_verify.add_argument(
        "-m", "--move", type=str, help="set default move keyword"
    )

    parser_info = subparsers.add_parser('info', help='dump data folder')
    parser_info.set_defaults(func=info)

    parser_mqtt = subparsers.add_parser('mqtt', help='mqtt util')
    parser_mqtt.set_defaults(func=mqtt)
    parser_mqtt.add_argument(
        "topic", type=str, help="a list of keywords to operate on",
        default="#"
    )

    parser_stitch = subparsers.add_parser('stitch', help='concat 2 words')
    parser_stitch.set_defaults(func=stitch)
    parser_stitch.add_argument(
        "keywords", type=str, nargs='+', help="a list of keywords to generate samples for"
    )
    parser_stitch.add_argument(
        "-o", "--offset", type=str, help="start file iteration at offset",
    )
    parser_stitch.add_argument(
        "--output", type=str, help="output to specific keyword",
        required=True
    )
    parser_stitch.add_argument(
        "--db", type=int, help="db adjuster for word splitting"
    )

    parser_tts = subparsers.add_parser('tts', help='text to speech')
    parser_tts.set_defaults(func=tts)
    parser_tts.add_argument(
        "text", type=str, help="text to send to be transcoded"
    )
    parser_tts.add_argument(
        "--file", help="save the wav"
    )
    parser_tts.add_argument(
        "-p", "--play", action="store_true", help="play audio"
    )

    parser_transcribe = subparsers.add_parser(
        'transcribe', help='transcribe audio file')
    parser_transcribe.set_defaults(func=transcribe)
    parser_transcribe.add_argument(
        "file", type=str, help="wav file to transcribe"
    )

    parser_model = subparsers.add_parser(
        'model', help='validate/update model')
    parser_model.set_defaults(func=alice_model)
    parser_model.add_argument(
        "-u", "--update", action="store_true", help="download the latest model from gitbub"
    )

    parser.add_argument(
        "--data", help="set the default data location",
        default=os.path.join(os.path.expanduser('~'), "alice_data")
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="maybe print something helpful"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="log more"
    )
    parser.add_argument(
        "--audio_input", type=str, help="specify in device from {list}"
    )
    parser.add_argument(
        "--audio_output", type=str, help="specify out device from {list}"
    )
    args = parser.parse_args()
    config = load_config(args)
    if config.verbose:
        _log.setLevel(level=logging.INFO)
    if config.debug:
        config.verbose = True
        cprint("debug enabled", "red")
        _log.setLevel(level=logging.DEBUG)

    if config.whisper:
        try:
            import whisper
        except ImportError:
            print("you will need to install whisper")
            print("pip install git+https://github.com/openai/whisper.git")
            sys.exit(1)

    if not os.path.exists(os.path.join(config.alice.model_path, "stream.tflite")):
        download_model(config)

    _log.info("data path: %s", config.data)

    await args.func(config, **kwargs)

    tasks = [task for task in asyncio.all_tasks() if not task.done]
    _log.debug("pending tasks %i", len(tasks))
    for task in tasks:
        await task

    _log.debug("shutting down")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
