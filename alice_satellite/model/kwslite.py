"""wrapper for handling the kws module"""
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
import time
import logging
from typing import (
    Tuple
)
import ast
import numpy as np
from scipy.special import softmax
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

_log = logging.getLogger("alice.kws")

MS_PER_SECOND = 1000  # milliseconds in 1 second
SILENCE_LABEL = '_silence_'
UNKNOWN_WORD_LABEL = '_unknown_'


def get_default_flags() -> argparse.Namespace:
    flags = argparse.Namespace()
    # audio processor / training
    flags.restore_checkpoint = 1
    flags.wanted_words = 'visual,wow,learn,backward,dog,two,left,happy,nine,go,up,bed,stop,one,zero,tree,seven,on,four,bird,right,eight,no,six,forward,house,marvin,sheila,five,off,three,down,cat,follow,yes'
    flags.wav = 1
    flags.data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    flags.data_dir = '/tmp/speech_dataset'
    flags.split_data = 1
    flags.silence_percentage = 10.0
    flags.unknown_percentage = 10.0
    flags.validation_percentage = 10
    flags.testing_percentage = 10
    flags.how_many_training_steps = '20000,20000,20000,20000,20000,20000'
    flags.learning_rate = '0.01,0.005,0.002,0.001,0.0005,0.0002'
    flags.model_name = 'ds_tc_resnet'
    flags.train_dir = ''
    flags.optimizer_epsilon = 1e-08
    flags.optimizer = 'adam'
    flags.summaries_dir = ''
    flags.pick_deterministically = 1
    flags.background_frequency = 0.8
    flags.background_volume = 0.1
    flags.volume_resample = 0.0
    flags.lr_schedule = 'linear'
    flags.eval_step_interval = 662
    flags.window_stride_samples = 160
    flags.clip_duration_ms = 1000
    flags.causal_data_frame_padding = 0

    # params
    flags.label_count = 2
    flags.return_softmax = 0
    flags.verbosity = 0

    # speech params
    flags.sample_rate = 16000
    flags.window_size_ms = 30.0
    flags.window_stride_ms = 10.0
    flags.feature_type = 'mfcc_tf'
    flags.preemph = 0.0
    flags.mel_lower_edge_hertz = 20.0
    flags.mel_upper_edge_hertz = 7600.0
    flags.log_epsilon = 1e-12
    flags.dct_num_features = 40
    flags.mel_non_zero_only = 1
    flags.fft_magnitude_squared = False
    flags.mel_num_bins = 80
    flags.window_type = 'hann'
    flags.use_spec_augment = 1
    flags.time_masks_number = 2
    flags.time_mask_max_size = 25
    flags.frequency_masks_number = 2
    flags.frequency_mask_max_size = 7
    flags.use_tf_fft = 0
    flags.use_spec_cutout = 0
    flags.spec_cutout_masks_number = 3
    flags.spec_cutout_time_mask_size = 10
    flags.spec_cutout_frequency_mask_size = 5
    flags.time_shift_ms = 100.0
    flags.sp_time_shift_ms = 0.0
    flags.resample = 0.15
    flags.sp_resample = 0.0
    flags.data_frame_padding = None
    flags.use_quantize_nbit = 0

    # General
    flags.preprocess = 'raw'
    flags.desired_samples = 16000
    flags.batch_size = 128
    flags.quantize = 0
    flags.data_stride = 1

    # Model params
    flags.activation = 'relu'
    flags.dropout = 0.0
    flags.ds_max_pool = 0
    flags.ds_scale = 1
    flags.nbit_8bit_until_block = 1
    flags.nbit_8bit_last = 1
    flags.ds_filters = '128, 64, 64, 64, 128, 128'
    flags.ds_repeat = '1, 1, 1, 1, 1, 1'
    flags.ds_kernel_size = '11, 13, 15, 17, 29, 1'
    flags.ds_stride = '1, 1, 1, 1, 1, 1'
    flags.ds_dilation = '1, 1, 1, 1, 2, 1'
    flags.ds_residual = '0, 1, 1, 1, 0, 0'
    flags.ds_pool = '1, 1, 1, 1, 1, 1'
    # flags.ds_paddig',
    #         "'same', 'same', 'same', 'same', 'same', 'same'"
    flags.ds_padding = "'causal','causal','causal','causal','causal','causal'"
    flags.ds_filter_separable = '1, 1, 1, 1, 1, 1'

    return flags


def parse(text):
    """Parse model parameters.

    Args:
      text: string with layer parameters: '128,128' or "'relu','relu'".

    Returns:
      list of parsed parameters
    """
    if not text:
        return []
    res = ast.literal_eval(text)
    if isinstance(res, tuple):
        return res
    else:
        return [res]


def run_stream_inference_classification_tflite(flags, interpreter, inp_audio,
                                               input_states):
    """Runs streaming inference classification with tflite (external state).

    It is useful for testing streaming classification
    Args:
      flags: model and data settings
      interpreter: tf lite interpreter in streaming mode
      inp_audio: input audio data
      input_states: input states
    Returns:
      last output
    """

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != len(output_details):
        raise ValueError('Number of inputs should be equal to the number of outputs'
                         'for the case of streaming model with external state')

    stream_step_size = flags.data_shape[0]
    start = 0
    end = stream_step_size
    while end <= inp_audio.shape[1]:
        stream_update = inp_audio[:, start:end]
        stream_update = stream_update.astype(np.float32)

        # update indexes of streamed updates
        start = end
        end += stream_step_size

        # set input audio data (by default input data at index 0)
        interpreter.set_tensor(input_details[0]['index'], stream_update)

        # set input states (index 1...)
        for s in range(1, len(input_details)):
            interpreter.set_tensor(input_details[s]['index'], input_states[s])

        # run inference
        interpreter.invoke()

        # get output: classification
        out_tflite = interpreter.get_tensor(output_details[0]['index'])

        # get output states and set it back to input states
        # which will be fed in the next inference cycle
        for s in range(1, len(input_details)):
            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            input_states[s] = interpreter.get_tensor(
                output_details[s]['index'])

    return out_tflite


def prepare_words_list(wanted_words, split_data):
    """Prepends common tokens to the custom word list.

    Args:
      wanted_words: List of strings containing the custom words.
      split_data: True - split data automatically; False - user splits the data

    Returns:
      List with the standard silence and unknown tokens added.
    """
    if split_data:
        # with automatic data split we append two more labels
        return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words
    else:
        # data already split by user, no need to add other labels
        return wanted_words


def update_flags(flags):
    """Update flags with new parameters.

    Args:
      flags: All model and data parameters

    Returns:
      Updated flags

    Raises:
      ValueError: If the preprocessing mode isn't recognized.
    """

    label_count = len(
        prepare_words_list(flags.wanted_words.split(','), flags.split_data))
    desired_samples = int(flags.sample_rate * flags.clip_duration_ms /
                          MS_PER_SECOND)
    window_size_samples = int(flags.sample_rate * flags.window_size_ms /
                              MS_PER_SECOND)
    window_stride_samples = int(flags.sample_rate * flags.window_stride_ms /
                                MS_PER_SECOND)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + \
            int(length_minus_window / window_stride_samples)

    upd_flags = flags
    upd_flags.label_count = label_count
    upd_flags.desired_samples = desired_samples
    upd_flags.window_size_samples = window_size_samples
    upd_flags.window_stride_samples = window_stride_samples
    upd_flags.spectrogram_length = spectrogram_length
    if upd_flags.fft_magnitude_squared in (0, 1):
        upd_flags.fft_magnitude_squared = bool(upd_flags.fft_magnitude_squared)
    else:
        raise ValueError('Non boolean value %d' %
                         upd_flags.fft_magnitude_squared)

    # by default data_frame does not do use causal padding
    # it can cause small numerical difference in streaming mode
    if flags.causal_data_frame_padding:
        upd_flags.data_frame_padding = 'causal'
    else:
        upd_flags.data_frame_padding = None

    # summary logs for TensorBoard
    upd_flags.summaries_dir = os.path.join(flags.train_dir, 'logs/')
    return upd_flags


class AliceKWS():
    def __init__(self, config: argparse.Namespace, training: bool = False) -> None:
        _log.debug("initializing")

        self.interpreter = None
        self.index_to_label = None

        if hasattr(config, 'kws'):
            _log.info("loading kws state")
            d = vars(get_default_flags())
            m = d | config.kws
            self.flags = argparse.Namespace(**m)
        else:
            self.flags = get_default_flags()
        self.flags = update_flags(self.flags)
        self.flags.data_dir = os.path.join(config.data, 'samples')
        self.flags.train_dir = os.path.join(config.data, 'kws')
        self.flags.summaries_dir = os.path.join(self.flags.train_dir, 'logs')
        self.training = training
        self.data = config.data

        pools = parse(self.flags.ds_pool)
        strides = parse(self.flags.ds_stride)
        time_stride = [1]
        for pool in pools:
            if pool > 1:
                time_stride.append(pool)
        for stride in strides:
            if stride > 1:
                time_stride.append(stride)
        total_stride = np.prod(time_stride)
        self.flags.data_stride = total_stride
        self.flags.data_shape = (
            total_stride * self.flags.window_stride_samples,)
        _log.debug("object initialized")

    def load(self) -> None:
        if os.path.exists(self.data):
            _log.info("loading existing model")
            model_path = os.path.join(
                self.data, "tflite", "stream.tflite")
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            if self.index_to_label is None:
                labels_file = os.path.join(self.data, "tflite", "labels.txt")
                with open(labels_file, encoding="utf-8") as fd:
                    self.index_to_label = fd.read().split('\n')

            _log.info(self.index_to_label)
        else:
            raise FileNotFoundError(f"You need to install a model to {self.data}/tflite")

    def predict(self, wavform) -> Tuple[int, str, float, float]:
        start_time = time.time()
        if not self.interpreter:
            self.load()

        input_details = self.interpreter.get_input_details()

        input_data = wavform[np.newaxis, ...]
        input_states = []
        input_len = len(input_details)
        for shape in range(input_len):
            input_states.append(
                np.zeros(input_details[shape]['shape'], dtype=np.float32))

        predictions = run_stream_inference_classification_tflite(
            self.flags, self.interpreter, input_data, input_states)

        class_idx = np.argmax(predictions, axis=1)[0]
        label = self.index_to_label[class_idx]
        p = softmax(predictions[0])[class_idx]
        stop_time = time.time()

        return class_idx, label, (100*p), ((stop_time - start_time) * 1000)
