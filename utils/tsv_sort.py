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

'''
Used to sort data sets download from mozilla, used for training different languages
https://commonvoice.mozilla.org/en/datasets

usage:
    cd dataset/lang
    python ~/umbrellacodr/alice_satellite/utils/tsv_sort.py -s clips -d out -p pl -t dev.tsv test.tsv train.tsv

    validated.tsv should contain all validated samples
'''

import os
import sys
import argparse
import unicodedata
import pandas as pd
from pydub import AudioSegment


def normalize(string: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', string)
                   if unicodedata.category(c) != 'Mn').lower()


def do_something(args, **kwargs):
    if not os.path.exists(args.src):
        print(f"{args.src} not found")
        sys.exit(0)

    for file in args.tsv:
        if os.path.exists(file):
            print(f"processing {file}")
            data = pd.read_csv(file, header=0, usecols=[
                               "path", "sentence"], sep='\t')

            for _, sound_info in data.iterrows():
                keyword = normalize(sound_info['sentence'])
                if args.prefix:
                    keyword_path = os.path.join(
                        args.dst, args.prefix+'_'+keyword)
                else:
                    keyword_path = os.path.join(args.dst, keyword)
                sound_file = sound_info['path']
                sound_path = os.path.join(args.src, sound_file)
                print(f"{sound_file} -> {keyword}")
                if not os.path.exists(keyword_path):
                    try:
                        os.makedirs(keyword_path)
                    except FileExistsError:
                        pass
                sound_name, _ = os.path.splitext(
                    os.path.basename(sound_file))
                out_file = os.path.join(keyword_path, sound_name+".wav")
                sound: AudioSegment
                sound = AudioSegment.from_mp3(sound_path)
                sound = sound.set_frame_rate(16000)
                sound = sound.set_channels(1)
                sound.export(out_file, format="wav")
        else:
            print(f"{file} not found")


def main(**kwargs):
    parser = argparse.ArgumentParser(prog="tsv sort")
    parser.set_defaults(func=do_something)
    parser.add_argument(
        "-s", "--src", help="location of audio files", required=True
    )
    parser.add_argument(
        "-d", "--dst", help="where to copy the files too", required=True
    )
    parser.add_argument(
        "-t", "--tsv", type=str, nargs='*', help="list of tsv files to parse", required=True
    )
    parser.add_argument(
        "-p", "--prefix", help="prefix keyword folders"
    )

    args = parser.parse_args()
    args.func(args, **kwargs)


if __name__ == "__main__":
    main()
