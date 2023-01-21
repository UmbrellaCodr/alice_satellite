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
import re
import asyncio
import random
from typing import (
    Tuple,
    Generator,
)
import time
import datetime
from enum import Enum
import json
import logging
import hashlib
import requests
from six.moves import urllib
from importlib_resources import files
import numpy as np
from termcolor import colored, cprint
import sounddevice as sd
import readchar
import yaml
import asyncio_mqtt as aiomqtt
import noisereduce as nr

from . import ALICE_MODULE_PATH, AliceDefaults
from .audiohandler import AudioHandler
from .mqtthandler import MessageHandler, MessageAsrStart, MessageAsrStop

_log = logging.getLogger("alice")


def keyword_normalize(keyword: str) -> str:
    pattern = re.compile('[\W_]+')
    return pattern.sub('_', keyword).lower()


def default_config() -> dict:
    config = {}
    config['alice_url'] = "https://raw.githubusercontent.com/UmbrellaCodr/alice_satellite/main"
    config['audio_input'] = sd.default.device[0]
    config['audio_output'] = sd.default.device[1]
    config['db'] = int(80)
    config['desired_sample_rate'] = int(16000)
    config['debug'] = False
    config['verbose'] = False
    config['site_id'] = "default"
    config['whisper'] = False
    config['whisper_model'] = "base"
    return config


def load_config(args: argparse.Namespace) -> dict:
    config_path = os.path.join(args.data, "config.yml")
    config = dict()
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            if not config:
                config = dict()
        _log.info("config loaded")
    except (FileNotFoundError, yaml.composer.ComposerError, yaml.constructor.ConstructorError):
        pass
    return (default_config() | config)


def merge_config(config: dict, args: dict) -> argparse.Namespace:
    for k in config.keys():
        if k in args and args[k] is None:
            del args[k]
    merged = config | args
    new_config = argparse.Namespace(**merged)
    return new_config


def get_resources(config: argparse.Namespace):
    config.alice = AliceDefaults(config)
    if not hasattr(config, 'wav'):
        config.wav = argparse.Namespace()
        config.wav.err = files('alice_satellite.wav').joinpath('alice_err.wav')
        config.wav.wake = files(
            'alice_satellite.wav').joinpath('alice_wake.wav')
        config.wav.recorded = files(
            'alice_satellite.wav').joinpath('alice_recorded.wav')
    if hasattr(config, 'mqtt'):
        config.mqtt = argparse.Namespace(**config.mqtt)


async def save_config(config: argparse.Namespace, **kwargs) -> None:
    import alice_satellite.model.kwslite as kws

    config_path = os.path.join(config.data, "config.yml")
    KWS_DATA = 'kws'

    init = argparse.Namespace(data=config.data)
    o = kws.AliceKWS(init)
    new_config = load_config(config)
    if hasattr(config, 'kws'):
        new_config[KWS_DATA] = (vars(o.flags) | config.kws)
    else:
        new_config[KWS_DATA] = (vars(o.flags))
    del new_config[KWS_DATA]['data_stride']
    del new_config[KWS_DATA]['data_shape']
    del new_config[KWS_DATA]['data_dir']
    del new_config[KWS_DATA]['train_dir']
    del new_config[KWS_DATA]['summaries_dir']
    _log.debug(new_config)
    try:
        os.makedirs(config.data)
    except FileExistsError:
        pass
    with open(config_path, 'w', encoding='utf-8') as file:
        yaml.dump(new_config, file)


def get_kw(config: argparse.Namespace) -> list:
    """get a list of all keywords"""
    return sorted(next(os.walk(config.alice.samples_path))[1])


def get_kw_samples(config: argparse.Namespace, keyword: str, samples_path: str = None, get_all: bool = True, offset: str = None, get_audio: bool = False) -> Generator[Tuple[str, str, np.ndarray], None, None]:
    """get kw samples"""
    _use_path = samples_path
    if _use_path is None:
        _use_path = config.alice.samples_path
    kw_path = os.path.join(_use_path, keyword_normalize(keyword))

    found = True
    if offset:
        found = False
    try:
        for audio_name in sorted(os.listdir(kw_path)):
            if not found:
                if offset.upper() in audio_name.upper():
                    found = True
                else:
                    continue
            if audio_name.startswith("."):
                continue
            if not audio_name.endswith("wav"):
                continue
            if not get_all and audio_name.startswith("m"):
                continue
            audio = None
            audio_path = os.path.join(kw_path, audio_name)
            if get_audio:
                audio_handler = AudioHandler(config)
                audio = audio_handler.load_audio(audio_path)
            yield audio_name, audio_path, audio
    except FileNotFoundError:
        pass


def get_kw_samples_count(config: argparse.Namespace, keyword: str, samples_path: str = None, get_all: bool = True, offset: str = None) -> int:
    count = 0
    for _ in get_kw_samples(config, keyword=keyword, samples_path=samples_path, get_all=get_all, offset=offset):
        count += 1
    return count


async def list_audio_devices(config: argparse.Namespace, **kwargs) -> None:
    device_info = sd.query_devices(device=None)
    print(device_info)


async def analyze(config: argparse.Namespace, **kwargs) -> None:
    """analyze samples"""
    import whisper
    wm = whisper.load_model(config.whisper_model)

    audio_handler = AudioHandler(config, training=True, **kwargs)
    kw_found = get_kw(config)
    kw_found.remove("noise")

    print("current keywords: {}".format(kw_found))

    async def handle_response():
        """handle response"""

        # if args.auto:
        #     await a.play_audio(audio, blocking=False)
        result = False
        options = "[1] skip [2] play [3] delete"
        print(options)
        loop = True
        while loop:
            loop = False
            char = readchar.readkey()
            if char == "1":
                pass
            elif char == "2":
                audio_handler.play_audio(a_wavform, blocking=False)
                loop = True
            elif char == "3":
                result = True
            else:
                loop = True
        return result

    removed = 0
    for kw in config.keywords:
        count = 0
        total = get_kw_samples_count(
            config, keyword=kw, get_all=config.all, offset=config.offset)
        for a_name, a_path, a_wavform in get_kw_samples(config, keyword=kw, get_all=config.all, offset=config.offset, get_audio=config.prompt):
            count += 1
            purge = False
            result = wm.transcribe(a_path, fp16=False)
            if kw == "noise":
                if any(element.upper() in result['text'].upper() for element in kw_found):
                    purge = True
                    col = "red"
                else:
                    col = "green"
            else:
                if config.match:
                    if any(element.strip().upper() in result['text'].strip().upper() for element in config.match):
                        col = "green"
                    else:
                        purge = True
                        col = "red"
                elif kw.strip().upper() in result['text'].strip().upper():
                    col = "green"
                else:
                    purge = True
                    col = "red"
            if purge:
                remove = False
                print("({}/{}) {} -> {} ".format(count, total,
                      a_name, colored(result['text'], col)))
                if config.prompt:
                    remove = await handle_response()
                elif config.purge:
                    remove = True
                if remove:
                    os.remove(a_path)
                    print(
                        "-> {}".format(colored("removed", "red", attrs=['bold'])))
                    removed += 1
            else:
                print("({}/{}) {} -> {}".format(count, total,
                      a_name, colored(result['text'], col)))
    print("removed {} samples".format(removed))


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


async def generate(config: argparse.Namespace, **kwargs) -> None:
    audio_handler = AudioHandler(config, training=True, **kwargs)
    window = np.zeros(0)
    audio_handler.connect()

    async with audio_handler.audio() as chunks:
        kw = "noise"
        count = get_kw_samples_count(config, kw, get_all=False)
        if count < config.count:
            print("generating background samples... {} -> {}".format(count, config.count))
            async for audio, noise in chunks:
                if noise:
                    cprint("+", "cyan", attrs=["bold"], flush=True, end="")
                else:
                    print(".", flush=True, end="")
                count += 1
                file_name = "{}_{}".format(int(time.time()), count)
                file_path = os.path.join(
                    config.alice.samples_path, keyword_normalize(kw), file_name)
                audio_handler.save_audio(audio, file_path)
                if count >= config.count:
                    print("")
                    break

        for kw in config.keywords:
            samples = get_kw_samples_count(config, kw, get_all=False)
            target = config.count
            print("keyword: {} -> {}".format(kw, samples))
            if samples >= target:
                cprint("done", "red")
                continue

            print("generating sampels for " +
                  colored("{}".format(kw), "blue", attrs=['bold']))

            async for audio, noise in chunks:
                if noise:
                    cprint("+", "cyan", attrs=["bold"], flush=True, end="")
                    window = np.concatenate(
                        (window, audio[-audio_handler.window_size:]))
                else:
                    print(".", flush=True, end="")
                    if window.size <= audio_handler.window_size:
                        continue
                    samples_n = 0
                    if not window.size % audio_handler.params.desired_sample_rate == 0:
                        window = np.concatenate(
                            (window, np.zeros(audio_handler.window_size)))
                    for seg in audio_handler.detect_words(window, verbose=True):
                        file_name = "{}_{}".format(int(time.time()), samples_n)
                        file_path = os.path.join(
                            config.alice.samples_path, keyword_normalize(kw), file_name)
                        audio_handler.save_audio(seg, file_path)
                        samples_n += 1
                    if samples_n > 0:
                        samples += samples_n
                        if samples % 10 == 0:
                            cprint("{}".format(samples),
                                   "red", flush=True, end="")
                    window = np.zeros(0)
                    if samples >= target:
                        break


def get_random_kw_audio(config: argparse.Namespace, keyword: str = "noise") -> np.ndarray:
    """get a random noise sample"""
    audio_handler = AudioHandler(config)
    index = get_kw_samples_count(config, keyword, get_all=False)
    index = random.randint(0, index-1)
    count = 0
    for _, fp, _ in get_kw_samples(config, keyword, get_all=False):
        if count == index:
            if config.debug:
                print("using noise file: {}".format(fp))
            audio = audio_handler.load_audio(fp)
            break
        count += 1

    return audio


async def stitch(config: argparse.Namespace, **kwargs) -> None:
    audio_handler = AudioHandler(config, training=True, **kwargs)
    output = keyword_normalize(config.output)
    print("processing: {} -> {}".format(config.keywords, output))

    if len(config.keywords) != 2:
        raise ValueError(
            f"need 2 got {len(config.keywords)} keywords to stitch together")

    b_files = []
    for _, fp, _ in get_kw_samples(config, keyword=config.keywords[1], get_all=False):
        b_files.append(fp)

    count = 0
    for _, _, a_wavform in get_kw_samples(config, keyword=config.keywords[0], get_all=False, offset=config.offset, get_audio=True):
        index = random.randint(0, len(b_files)-1)
        b_wavform = audio_handler.load_audio(b_files[index])
        for seg in audio_handler.detect_words(a_wavform, True, callback=audio_handler.concat, pad=b_wavform):
            file_name = "s_{}_{}".format(int(time.time()), count)
            file_path = os.path.join(
                config.alice.samples_path, output, file_name)
            audio_handler.save_audio(seg, file_path)
            count += 1


def morph_file(config: argparse.Namespace, audio_handler: AudioHandler, wav_file: str, keyword: str, morph_count: int, noise: np.ndarray) -> None:
    """morph wavform"""
    audio = audio_handler.load_audio(wav_file)
    name, _ = os.path.splitext(os.path.basename(wav_file))
    seg_c = 0
    for seg in audio_handler.detect_words(audio, verbose=True, morph_count=morph_count, pad=noise):
        file_name = "m_{}_{}".format(name, seg_c)
        file_path = os.path.join(
            config.alice.samples_path, keyword_normalize(keyword), file_name)
        audio_handler.save_audio(seg, file_path)
        seg_c += 1
    if seg_c == 0:
        cprint("No words detected {}".format(
            name), 'magenta', flush=True, end="")


async def morph(config: argparse.Namespace, **kwargs) -> None:
    """morph samples"""
    audio_handler = AudioHandler(config, training=True, **kwargs)
    noise = get_random_kw_audio(config)
    samples = 0
    for k in config.keywords:
        for _, fp, _ in get_kw_samples(config, k, get_all=False):
            samples += 1
            morph_file(config, audio_handler, fp, k, config.count, noise)
        print("  proceessed: {} -> {} samples".format(k, samples))


async def listen(config: argparse.Namespace, **kwargs) -> None:
    import alice_satellite.model.kwslite as kws

    model_handler = kws.AliceKWS(config)
    audio_handler = AudioHandler(config, training=False, **kwargs)
    count = int(0)
    audio_handler.connect()

    async with audio_handler.audio() as chunks:
        async for audio, noise in chunks:
            if noise:
                for seg in audio_handler.detect_words(audio, verbose=False):
                    class_id, label, prob, perf = model_handler.predict(seg)
                    if config.save and not str(class_id) in config.ignore:
                        file_name = "{}_{}_{}".format(
                            int(time.time()), count, int(prob))
                        file_path = os.path.join(
                            config.data, "listen", keyword_normalize(label), file_name)
                        audio_handler.save_audio(seg, file_path)
                    prob_c = "red"
                    if prob > 90:
                        prob_c = "green"
                    print("id: {} label: {} {} time: {:.3f}ms count: {}".format(
                        class_id, label, colored("({:4.1f}%)".format(prob), prob_c), perf, count))
                    count += 1


async def detect(config: argparse.Namespace, **kwargs) -> None:
    import alice_satellite.model.kwslite as kws
    if config.whisper:
        import whisper
        wm = whisper.load_model(config.whisper_model)

    tasks = []
    model_handler = kws.AliceKWS(config)
    audio_handler = AudioHandler(config, training=False, **kwargs)
    audio_handler.connect()
    mqtt_handler = None
    if hasattr(config, 'mqtt'):
        mqtt_handler = MessageHandler(config)

    wav_err = audio_handler.load_audio(config.wav.err)
    wav_wake = audio_handler.load_audio(config.wav.wake)
    wav_recorded = audio_handler.load_audio(config.wav.recorded)

    modes = Enum('Modes', ['LISTEN', 'DETECTED', 'RECORD'])
    mode = modes.LISTEN
    mode_deadline = 0
    window = np.zeros(0)
    kw_map = dict()
    if config.index:
        print("monitoring the following: {}".format(config.index))
        for k in config.index:
            kw_map[int(k)] = 0

    def mode_listen(wavform, noise):
        nonlocal mode, mode_deadline
        now = time.time_ns() - (1e9 * 3)
        detected_class_id = False
        if noise:
            for seg in audio_handler.detect_words(wavform, verbose=False):
                detected = False
                class_id, label, prob, perf = model_handler.predict(seg)
                if class_id == config.match:
                    detected_class_id = True
                    if prob > 90:
                        detected = True
                if class_id in config.index:
                    detected_class_id = True
                    if prob > 90:
                        kw_map[class_id] = time.time_ns()
                    for v in kw_map.values():
                        if v > now:
                            detected = True
                        else:
                            detected = False
                            break
                if detected_class_id and config.verbose:
                    prob_c = "red"
                    if prob > 90:
                        prob_c = "green"
                    print("debug: wake word label: {} {} time: {:.3f}ms now: {}, map: ".format(
                        label, colored("({:4.1f}%)".format(prob), prob_c), perf, now), end="")
                    for k, v in kw_map.items():
                        v_c = "red"
                        if v > now:
                            v_c = "green"
                        print("{}:{} ".format(k, colored(v, v_c)), end="")
                    print("")

                if detected:
                    current = datetime.datetime.now()
                    audio_handler.play_audio(wav_wake, blocking=False)
                    cprint("wake word detected {}".format(
                        current.strftime("%Y-%m-%d %H:%M:%S")), "magenta")
                    for k, _ in kw_map.items():
                        kw_map[k] = 0
                    mode = modes.DETECTED
                    mode_deadline = time.time() + 5

    async def audio_task():
        async with audio_handler.audio() as chunks:
            async for audio, noise in chunks:
                nonlocal window, mode, mode_deadline
                mode_now = time.time()
                if mode == modes.LISTEN:
                    mode_listen(audio, noise)
                elif mode == modes.DETECTED:
                    if mode_now > mode_deadline:
                        cprint("DT", "red", attrs=["bold"], flush=True, end="")
                        mode = modes.LISTEN
                        audio_handler.play_audio(wav_err, blocking=False)
                    elif noise:
                        mode = mode.RECORD
                        mode_deadline = time.time() + 5
                    else:
                        print(".", flush=True, end="")
                if mode == modes.RECORD:
                    transcribe = False
                    if mode_now > mode_deadline:
                        cprint("T", "red", attrs=["bold"], flush=True, end="")
                        if window.size > 0:
                            transcribe = True
                        else:
                            mode = modes.LISTEN
                            audio_handler.play_audio(wav_err, blocking=False)
                            window = np.zeros(0)
                    elif noise:
                        cprint("+", "cyan", attrs=["bold"], flush=True, end="")
                        window = np.concatenate(
                            (window, audio[-audio_handler.window_size:]))
                    else:
                        print(".", flush=True, end="")
                        if not window.size % audio_handler.params.desired_sample_rate == 0:
                            window = np.concatenate(
                                (window, np.zeros(audio_handler.window_size)))
                        transcribe = True

                    if transcribe:
                        mode = mode.LISTEN
                        audio_handler.play_audio(wav_recorded, blocking=False)
                        result = {'text': 'disabled'}
                        if config.whisper:
                            result = wm.transcribe(window.astype(
                                dtype=np.float32), fp16=False)
                        rhasspy_result = {'text': 'disabled'}
                        if mqtt_handler:
                            rhasspy_result = await mqtt_handler.transcribe_audio(window, config.site_id)
                        print(" whisper: {}, rhasspy: {}".format(
                            result['text'], rhasspy_result['text']))
                        window = np.zeros(0)

    if config.debug:
        cprint("starting", "magenta")

    if mqtt_handler:
        tasks.append(asyncio.create_task(mqtt_handler.task()))
    tasks.append(asyncio.create_task(audio_task()))

    while tasks:
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        _log.debug("%i remaining tasks", len(pending))
        tasks = pending


async def verify(config: argparse.Namespace, **kwargs) -> None:
    """verify samples"""
    audio_handler = AudioHandler(config, training=True, **kwargs)
    kw_map = dict()
    count = 1
    if os.path.exists(config.alice.samples_path):
        for i in sorted(os.listdir(config.alice.samples_path)):
            kw_path = os.path.join(config.alice.samples_path, i)
            if not i.startswith(".") and os.path.isdir(kw_path):
                kw_map[count] = dict()
                kw_map[count]['name'] = i
                kw_map[count]['path'] = kw_path
                count += 1

    # todo fix kw > 9
    # def handle_move(audio_file, morph=False):
    #     """move sample"""
    #     result = False
    #     if not config.move:
    #         print(colored("move to:", "magenta"), end="")
    #         for k, v in kw_map.items():
    #             print(" [{}] {}".format(k, v['name']), end="")
    #         print("")
    #     loop = True
    #     while loop:
    #         loop = False
    #         i = 0
    #         if not config.move:
    #             char = readchar.readkey()
    #             try:
    #                 i = int(char)
    #             except ValueError:
    #                 i = 0
    #         else:
    #             for k, v in kw_map.items():
    #                 if v['name'] == config.move:
    #                     i = k
    #                     break
    #             if i == 0:
    #                 raise AssertionError("option not found in map")

    #         if i > 0 and i < count:
    #             name = os.path.basename(audio_file)
    #             dst_file = os.path.join(kw_map[i]['path'], name)
    #             os.rename(audio_file, dst_file)
    #             if morph:
    #                 noise = get_noise(config)
    #                 morph_file(config, audio_handler, dst_file,
    #                            kw_map[i]['name'], 10, noise)
    #             print("{} {} -> {}".format(colored("moved",
    #                   "green"), name, kw_map[i]['name']))
    #             result = True
    #         else:
    #             if not char == "a":
    #                 print("invalid: press 'a' to abort")
    #                 loop = True
    #     return result

    def handle_response(name, audio_name, audio, audio_file):
        """handle response"""

        if config.auto:
            audio_handler.play_audio(audio, blocking=False)
        options = "[1] skip [2] play [3] move [4] delete [5] move/morph"
        if not processing_listen:
            options += " [6] morph"
        print("{}".format(audio_name))
        print(options)
        loop = True
        while loop:
            loop = False
            char = readchar.readkey()
            if char == "1":
                pass
            elif char == "2":
                audio_handler.play_audio(audio, blocking=False)
                loop = True
            elif char == "3":
                print("disabled")
                loop = True
                # if not handle_move(audio_file):
                #     print(colored("aborted", "red", attrs=['bold']))
                #     print(options)
                #     loop = True
            elif char == "4":
                os.remove(audio_file)
                print("{} {}".format(colored("deteled", "red"), audio_name))
            elif char == "5":
                print("disabled")
                loop = True
                # if not handle_move(audio_file, morph=True):
                #     print(colored("aborted", "red", attrs=['bold']))
                #     print(options)
                #     loop = True
            elif char == "6":
                noise = get_random_kw_audio(config)
                morph_file(config, audio_handler,
                           audio_file, name, 10, noise)
            else:
                loop = True

    def handle_path(kw, path):
        """process path"""
        for a_name, a_path, a_wavform in get_kw_samples(config, keyword=kw, samples_path=path, get_all=config.all, offset=config.offset, get_audio=True):
            print(a_name)
            audio_name = "{}/{}".format(colored(kw,
                                        "yellow", attrs=['bold']), a_name)
            handle_response(kw, audio_name, a_wavform, a_path)

    root_path = config.alice.samples_path
    processing_listen = False
    if not len(config.keywords) > 0:
        root_path = os.path.join(config.data, "listen")
        processing_listen = True
        print(colored("processing listen", attrs=['bold']))
        try:
            for parent in os.listdir(root_path):
                parent_path = os.path.join(root_path, parent)
                if not parent_path.startswith(".") and os.path.isdir(parent_path):
                    handle_path(parent, root_path)
        except FileNotFoundError:
            pass
    else:
        for k in config.keywords:
            handle_path(k, root_path)


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


def download_model(config: argparse.Namespace, **kwargs) -> None:
    try:
        os.makedirs(config.alice.model_path)
    except FileExistsError:
        pass
    model_files = ["labels.txt", "model_summary.txt",
                   "stream.tflite", "digest"]
    try:
        for filename in model_files:
            data_url = config.alice_url+"/tflite/"+filename
            out_path = os.path.join(config.alice.model_path, filename)

            def _progress(count, block_size, total_size):
                if total_size > block_size:
                    if (count * block_size) < total_size:
                        percentage = (float(count * block_size) /
                                      float(total_size)) * 100.0
                    else:
                        percentage = 100
                else:
                    percentage = 100
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' %
                    (filename, percentage))
                sys.stdout.flush()

            try:
                filepath, _ = urllib.request.urlretrieve(
                    data_url, out_path, _progress)
                print(" {} {}".format(colored("downloaded", "red"), filepath))
            except:
                _log.error(
                    'Failed to download URL: %s to folder: %s\n'
                    'Please make sure you have enough free space and'
                    ' an internet connection', data_url, config.alice.model_path)
                raise
    except urllib.error.HTTPError:
        _log.warning("unable to download model")


async def alice_model(config: argparse.Namespace, **kwargs) -> None:
    digest_file = os.path.join(config.alice.model_path, "digest")
    model_file = os.path.join(config.alice.model_path, "stream.tflite")
    digest = None
    remote_digest = None
    model_digest = None
    model_c = "red"
    digest_c = "red"

    remote_url = config.alice_url + "/tflite/digest"
    remote = requests.get(remote_url, timeout=10)
    remote_digest = remote.text.strip()
    if os.path.exists(model_file):
        model_digest = hashlib.sha256(
            open(model_file, 'rb').read()).hexdigest()
    if os.path.exists(digest_file):
        digest = open(digest_file, "r", encoding='utf-8').read()
    if digest == model_digest:
        model_c = "green"
    if digest == remote_digest:
        digest_c = "green"

    print(f"remote {remote_digest}")
    print(f"local {colored(digest, digest_c)}")

    print(f"model {colored(model_digest, model_c)}")
    if config.update:
        download_model(config)


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
    config = merge_config(config, vars(args))
    get_resources(config)
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
