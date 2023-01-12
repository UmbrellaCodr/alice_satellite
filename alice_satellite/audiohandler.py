"""Alice audio handler"""
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
from argparse import Namespace
from typing import (
    AsyncGenerator,
    Generator,
    Tuple,
)
import random
import asyncio
import logging
from contextlib import asynccontextmanager
import io
from termcolor import cprint
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import noisereduce as nr

_log = logging.getLogger("alice.audio")


class AudioParams:
    """hold audio settings"""

    def __init__(self, config: Namespace, rec_duration: float = 0.5) -> None:
        self.audio_in = config.audio_input
        self.audio_out = config.audio_output
        self.desired_sample_rate = config.desired_sample_rate
        self.rec_duration = rec_duration

        device_info = sd.query_devices()
        if self.audio_in < len(device_info):
            _log.info(device_info[self.audio_in])
            self.sample_rate_in = int(
                device_info[self.audio_in]['default_samplerate'])
            self.channels_in = int(
                device_info[self.audio_in]['max_input_channels'])
        else:
            _log.error("audio_in device error")
            self.sample_rate_in = 16000
            self.channels_in = 1
        if self.audio_out < len(device_info):
            _log.info(device_info[self.audio_out])
            self.sample_rate_out = int(
                device_info[self.audio_out]['default_samplerate'])
            self.channels_out = int(
                device_info[self.audio_out]['max_output_channels'])
        else:
            _log.error("audio_out device error")
            self.sample_rate_out = 16000
            self.channels_out = 1


class AudioHandler:
    def __init__(self,
                 config: Namespace,
                 training=False,
                 **kwargs) -> None:
        self.db = config.db
        self.training = training
        self.queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()
        self.stream_in = None

        self.params = AudioParams(config)

        self.window_size = int(self.params.desired_sample_rate *
                               self.params.rec_duration)

    def connect(self, **kwargs):
        def callback(indata: np.ndarray, frame_count: int, time_info, status: sd.CallbackFlags):
            try:
                self.loop.call_soon_threadsafe(
                    self.queue.put_nowait, (indata.copy(), status))
            except RuntimeError as error:
                _log.debug(error)

        blocksize = int(self.params.sample_rate_in * self.params.rec_duration)
        self.stream_in = sd.InputStream(callback=callback, channels=self.params.channels_in,
                                        samplerate=self.params.sample_rate_in,
                                        blocksize=blocksize,
                                        device=self.params.audio_in,
                                        dtype='float32',
                                        **kwargs)

    def disconnect(self, **kwargs):
        if self.stream_in:
            if not self.stream_in.stopped:
                self.stream_in.stop()
            self.stream_in.close()
        self.stream_in = None

    async def audio_generator(self, preprocess: bool = True) -> AsyncGenerator[np.ndarray, bool]:
        window = np.zeros(0)
        if not self.stream_in:
            raise AssertionError("stream not initialized")
        if not self.stream_in.active:
            raise AssertionError("stream not started")
        while True:
            indata, _ = await self.queue.get()
            if preprocess:
                window = np.concatenate((window, np.squeeze(indata[:, 0])))
                if window.size < self.params.sample_rate_in:
                    window = np.concatenate(
                        (window, np.zeros(self.window_size)))
                if window.size > self.params.sample_rate_in:
                    window = window[-self.params.sample_rate_in:]
                noise, valid = self.detect_noise(window)
                if valid:
                    if self.params.sample_rate_in != self.params.desired_sample_rate:
                        audio = librosa.resample(
                            window, orig_sr=self.params.sample_rate_in, target_sr=self.params.desired_sample_rate)
                    else:
                        audio = window[-self.params.sample_rate_in:]
                    yield audio, noise
            else:
                yield indata, True

    @asynccontextmanager
    async def audio(self, preprocess: bool = True) -> AsyncGenerator[AsyncGenerator[np.ndarray, bool], None]:
        generator = self.audio_generator(preprocess)

        try:
            self.stream_in.start()
            yield generator
        finally:
            self.stream_in.stop()

    def start(self):
        if self.stream_in:
            self.stream_in.start()

    def stop(self):
        if self.stream_in:
            self.stream_in.stop()

    async def __aenter__(self) -> "AudioHandler":
        """Connect to audio device"""
        self.connect()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        """Disconnect from audio device"""
        self.disconnect()
        _log.debug("handler exit")

    def detect_noise(self, wavform: np.ndarray) -> Tuple[bool, bool]:
        """detect noise"""
        rms = librosa.feature.rms(y=wavform)[0]
        if np.std(rms) == 0:
            return False, False
        r_normalized = (rms - 0.02) / np.std(rms)
        p = np.exp(r_normalized) / (1 + np.exp(r_normalized))

        transition = librosa.sequence.transition_loop(2, [0.5, 0.6])
        full_p = np.vstack([1 - p, p])
        states = librosa.sequence.viterbi_discriminative(full_p, transition)

        return any(states), True

    def pad_trunc(self, wavform: np.ndarray, verbose=False, pad: np.ndarray = None) -> None:
        if wavform.size >= self.params.desired_sample_rate:
            # Truncate the signal to the given length
            if verbose:
                print("T", flush=True, end="")
            wavform = wavform[:self.params.desired_sample_rate]
        else:
            if verbose:
                print("P", flush=True, end="")
            pad_size = (self.params.desired_sample_rate - wavform.size)
            if self.training:
                pad_begin_len = random.randint(0, pad_size)
            else:
                pad_begin_len = int(pad_size/2)
            pad_end_len = (self.params.desired_sample_rate -
                           wavform.size) - pad_begin_len

            # Pad with 0s
            if isinstance(pad, np.ndarray):
                if len(pad) != self.params.desired_sample_rate:
                    raise AssertionError
                pad_begin = pad[:pad_begin_len]
                pad_end = pad[pad_begin_len:pad_begin_len+pad_end_len]
            else:
                pad_begin = np.zeros(pad_begin_len)
                pad_end = np.zeros(pad_end_len)
                _log.debug("** pad: %i -> %i<, %i>", pad_size,
                           pad_begin_len, pad_end_len)

            wavform = np.concatenate([pad_begin, wavform, pad_end])

        return wavform

    def detect_words(self, wavform: np.ndarray, verbose: bool = False, morph_count: int = 0, pad: np.ndarray = None):
        """gather some audio"""
        x = nr.reduce_noise(y=wavform, sr=self.params.desired_sample_rate)
        xs = librosa.effects.split(x, top_db=self.db)
        for i in xs:
            start = i[0]
            stop = i[1]
            seg = x[start:stop]
            noise, valid = self.detect_noise(seg)
            if valid and noise:
                total = 1
                if morph_count > 0 and seg.size < self.params.desired_sample_rate:
                    total = morph_count
                for i in range(total):
                    if verbose:
                        cprint("o", 'green', end="")
                    seg_p = self.pad_trunc(seg, verbose=verbose, pad=pad)
                    yield seg_p

    def save_audio(self, wavform: np.ndarray, file: str) -> None:
        """save audio"""
        path = os.path.dirname(file)
        if not os.path.exists(path):
            os.makedirs(path)
        if not file.endswith(".wav"):
            file += ".wav"
        sf.write(file, wavform,
                 samplerate=self.params.desired_sample_rate, closefd=True)

    def save_bytes(self, wav_bytes: bytes, file: str) -> None:
        wavform, samplerate = sf.read(io.BytesIO(wav_bytes), dtype='float32')
        if samplerate != self.params.desired_sample_rate:
            wavform = librosa.resample(
                wavform, orig_sr=samplerate, target_sr=self.params.desired_sample_rate)
        self.save_audio(wavform, file)

    def play_audio(self, wavform: np.ndarray, blocking=True) -> None:
        """play audio"""
        # reshape audio to match audio out device
        if wavform.ndim > 1 and wavform.shape[1] > self.params.channels_out:
            wavform = wavform[:, 0:self.params.channels_out]
        sd.play(wavform, samplerate=self.params.desired_sample_rate,
                blocking=blocking, device=self.params.audio_out)

    def play_bytes(self, wav_bytes: bytes, blocking=True):
        """play audio bytes"""
        wavform, samplerate = sf.read(io.BytesIO(wav_bytes), dtype='float32')
        if samplerate != self.params.sample_rate_out:
            wavform = librosa.resample(
                wavform, orig_sr=samplerate, target_sr=self.params.sample_rate_out)
        # reshape audio to match audio out device
        if wavform.ndim > 1 and wavform.shape[1] > self.params.channels_out:
            wavform = wavform[:, 0:self.params.channels_out]

        sd.play(wavform, samplerate=self.params.sample_rate_out,
                blocking=blocking, device=self.params.audio_out)

    def load_audio(self, file: str) -> np.ndarray:
        """load wav file from disk"""
        wavform, sr = sf.read(file=file)
        if sr != self.params.desired_sample_rate:
            audio = librosa.resample(
                wavform.T, orig_sr=sr, target_sr=self.params.desired_sample_rate)
        else:
            audio = wavform
        return audio

    @staticmethod
    def chunk_audio(wavform: np.ndarray, chuck_size: int = 8000, samplerate=16000, add_empty_frame=False) -> Generator[bytes, None, None]:
        if not wavform.size > 0:
            return

        if add_empty_frame:
            empty = np.zeros(samplerate)
            wavform = np.concatenate([wavform, empty])
        offset_start = 0
        if wavform.size > chuck_size:
            offset_end = chuck_size
        else:
            offset_end = chuck_size - wavform.size
        while offset_start < wavform.size:
            _log.debug("chunk_audio chunk %i -> %i : %i",
                       offset_start, offset_end, wavform.size)
            with io.BytesIO() as io_chunk:
                sf.write(
                    io_chunk, wavform[offset_start:offset_end], samplerate=samplerate, format="WAV")
                yield io_chunk.getvalue()
            offset_start = offset_end
            if offset_end + chuck_size < wavform.size:
                offset_end += chuck_size
            else:
                offset_end += wavform.size - offset_end
