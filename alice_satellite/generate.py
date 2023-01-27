"""Alice generate"""
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

import os
import argparse
import logging
import re
from typing import (
    Tuple,
    Generator,
)
import time
import random
import datetime
from enum import Enum
import asyncio
import numpy as np
import readchar
from termcolor import colored, cprint

from .audiohandler import AudioHandler
from .mqtthandler import MessageHandler

_log = logging.getLogger("alice.generate")


def keyword_normalize(keyword: str) -> str:
    pattern = re.compile('[\W_]+')
    return pattern.sub('_', keyword).lower()


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
