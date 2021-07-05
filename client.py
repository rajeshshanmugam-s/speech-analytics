#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import shlex
import subprocess
import sys
import wave

from deepspeech import Model, printVersions
from timeit import default_timer as timer

from model_downloader import ds_model_downloader

from pre_processor import processor, words_corrector, boundary_definer

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

# Define the sample rate for audio

SAMPLE_RATE = 16000
# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_ALPHA = 0.75

# The beta hyperparameter of the CTC decoder. Word insertion bonus.
LM_BETA = 1.85


# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


lm="models/lm.binary"
trie="models/trie"
model1="models/output_graph.pbmm"
alph="models/alphabet.txt"
print('Loading model from file {}'.format(model1), file=sys.stderr)
model_load_start = timer()
ds = Model(model1, N_FEATURES, N_CONTEXT, alph, BEAM_WIDTH)
model_load_end = timer() - model_load_start
print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)


def convert_samplerate(audio_path):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), SAMPLE_RATE)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(SAMPLE_RATE, e.strerror))

    return SAMPLE_RATE, np.frombuffer(output, np.int16)


def words_from_metadata(metadata):
    #Todo: Simplify this
    word = ""
    word_list = []
    word_start_time = 0
    # Loop through each character
    for i in range(0, metadata.num_items):
        item = metadata.items[i]
        # Append character to word if it's not a space
        if item.character != " ":
            word = word + item.character
        # Word boundary is either a space or the last character in the array
        if item.character == " " or i == metadata.num_items - 1:
            word_duration = item.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["word"] = word
            each_word["start_time"] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0
        else:
            if len(word) == 1:
                # Log the start time of the new word
                word_start_time = item.start_time

    return word_list

def metadata_json_output(metadata):
    json_result = dict()
    json_result["words"] = words_from_metadata(metadata)
    # json_result["start_time"] = metadata.start_time
    return json_result


class VersionAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(VersionAction, self).__init__(nargs=0, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        printVersions()
        exit(0)


ds = Model(model1, N_FEATURES, N_CONTEXT, alph, BEAM_WIDTH)

if lm and trie:
    lm_load_start = timer()
    ds.enableDecoderWithLM(alph, lm, trie, LM_ALPHA, LM_BETA)
    lm_load_end = timer() - lm_load_start


def transcriber(audio, word_tagging):
    ds_model_downloader()
    fin = wave.open(audio, 'rb')
    fs = fin.getframerate()
    if fs != SAMPLE_RATE:
        fs, audio = convert_samplerate(audio)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    # audio_length = fin.getnframes() * (1/fs)
    fin.close()

    # inference_start = timer()
    if word_tagging:
        output = ds.sttWithMetadata(audio, fs)
        print(output)
        return metadata_json_output(output)
    else:
        transcript = ds.stt(audio, fs)
        return transcript


def governor(audio,word_tagging):
    if not word_tagging:
        transcript = transcriber(audio, word_tagging)
        return boundary_definer(transcript)
    elif word_tagging:
        transcript = transcriber(audio, word_tagging)
        return transcript

# print(governor('new_home_stars.wav', False))