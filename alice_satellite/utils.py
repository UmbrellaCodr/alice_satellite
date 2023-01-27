"""Alice utils"""
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
import hashlib
import requests
from six.moves import urllib
from termcolor import colored

_log = logging.getLogger("alice.utils")


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
