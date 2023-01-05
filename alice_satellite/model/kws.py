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

import sys
import os
import argparse
import time
import typing
import logging
import tensorflow.compat.v1 as tf
from scipy.special import softmax
import numpy as np
import hashlib
import shutil
from .kwslite import AliceKWS
from .. import ALICE_MODULE_PATH
sys.path.append(ALICE_MODULE_PATH)
from ..kws_streaming.layers import modes
from ..kws_streaming.models import utils
from ..kws_streaming.models import models
from ..kws_streaming.train import inference
from ..kws_streaming.train import train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_log = logging.getLogger("alice.kws")

class AliceKWSTrain(AliceKWS):
    def __init__(self, config: argparse.Namespace, training: bool = False) -> None:
        super().__init__(config, training)

    def convert_model_tflite(self, folder, fname, mode=modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE, optimizations=None, weights_name='best_weights') -> None:
        _log.info("converting model to tflite")
        tf.reset_default_graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        tf.keras.backend.set_session(sess)
        tf.keras.backend.set_learning_phase(0)
        self.flags.batch_size = 1  # set batch size for inference
        model = models.MODELS[self.flags.model_name](self.flags)
        model.load_weights(os.path.join(self.flags.train_dir,
                                        weights_name)).expect_partial()
        path_model = os.path.join(self.data, folder)
        if not os.path.exists(path_model):
            os.makedirs(path_model)
        src_labels_file = os.path.join(self.flags.train_dir, "labels.txt")
        dst_labels_file = os.path.join(self.data, "tflite", "labels.txt")
        shutil.copy(src_labels_file, dst_labels_file)
        try:
            with open(os.path.join(path_model, fname), 'wb') as fd:
                fd.write(
                    utils.model_to_tflite(sess, model, self.flags, mode, path_model,
                                          optimizations))
        except IOError as err:
            _log.warning('FAILED to write file: %s', err)
        except (ValueError, AttributeError, RuntimeError, TypeError) as err:
            _log.warning(
                'FAILED to convert to mode %s, tflite: %s', mode, err)
        digest_file = os.path.join(self.data, "tflite", "digest")
        model_digest = hashlib.sha256(open(os.path.join(path_model, fname), "rb").read()).hexdigest()
        with open(digest_file, "w", encoding="utf-8") as f:
            f.write(model_digest)

    def train(self) -> None:
        try:
            os.makedirs(self.flags.summaries_dir)
        except FileExistsError:
            pass

        train.train(self.flags)
        self.convert_model_tflite("tflite", "stream.tflite")

    def predict_with_model(self, wavform) -> typing.Tuple[int, str, float, float]:
        start_time = time.time()
        model = None
        weights_name = 'best_weights'
        if os.path.exists(self.flags.train_dir):
            _log.info("loading existing model")
            model = models.MODELS[self.flags.model_name](self.flags)
            model.load_weights(os.path.join(
                self.flags.train_dir, weights_name)).expect_partial()

        input_data = wavform[np.newaxis, ...]
        predictions = inference.run_stream_inference_classification(
            self.flags, model, input_data)

        class_idx = np.argmax(predictions, axis=1)[0]
        label = self.index_to_label[class_idx]
        p = softmax(predictions[0])[class_idx]
        stop_time = time.time()

        return class_idx, label, (100*p), ((stop_time - start_time) * 1000)

