"""Alice config"""
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
import yaml
import sounddevice as sd
from importlib_resources import files

from . import AliceDefaults

_log = logging.getLogger("alice.config")


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


def load_config(args: argparse.Namespace, merge: bool = True) -> argparse.Namespace:
    config_path = os.path.join(args.data, "config.yml")
    config = dict()
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            if not config:
                config = dict()
        _log.info("config loaded merged: %r", merge)
    except (FileNotFoundError, yaml.composer.ComposerError, yaml.constructor.ConstructorError):
        pass

    if merge:
        config = merge_config((default_config() | config), vars(args))
        get_resources(config)
    else:
        config = merge_config(default_config(), config)

    return config


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

    init = argparse.Namespace(data=config.data)
    o = kws.AliceKWS(init)
    new_config = load_config(config, merge=False)
    if hasattr(config, 'kws'):
        new_config.kws = (vars(o.flags) | config.kws)
    else:
        new_config.kws = (vars(o.flags))
    del new_config.kws['data_stride']
    del new_config.kws['data_shape']
    del new_config.kws['data_dir']
    del new_config.kws['train_dir']
    del new_config.kws['summaries_dir']
    _log.debug(new_config)
    try:
        os.makedirs(config.data)
    except FileExistsError:
        pass
    with open(config_path, 'w', encoding='utf-8') as file:
        yaml.dump(vars(new_config), file)
