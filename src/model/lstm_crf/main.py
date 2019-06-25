# reimplementation of https://github.com/guillaumegenthial/tf_ner/blob/master/models/lstm_crf/main.py

import functools
import json
import logging
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()
from tf_metrics import precision, recall, f1

DATADIR = "../../../data/toy"

# Setup Logging
Path('results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [ logging.FileHandler('results/main.log'), logging.StreamHandler(sys.stdout)]
logging.getLogger('tensorflow').handlers = handlers

# Data Pipeline
def parse_fn(line_words, line_tags):
    """Encodes words into bytes for tensor

    :param line_words: one line with words (aka sentences) with space between each word/token
    :param line_tags: one line of tags (one tag per word in line_words)
    :return: (list of encoded words, len(words)), list of encoded tags
    """

    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Number of words {} and Number of tags must be the same {}".format(len(words), len(tags))
    return (words, len(words)), tags

def generator_fn(words_file, tags_file):
    """Enumerator to enumerate through words_file and associated tags_file one line at a time

    :param words_file: file path of the words file (one sentence per line)
    :param tags_file: file path of tags file (tags corresponding to words file)
    :return enumerator that enumerates over the format (words, len(words)), tags one line at a time from input files.
    """

    with Path(words_file).open('r') as f_words, Path(tags_file).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words_file, tags_file, params = None, shuffle_and_repeat = False):
    """Creates tensorflow dataset using the generator_fn

    :param words_file: file path of the words file (one sentence per line)
    :param tags_file: file path of tags file (tags corresponding to words file)
    :param params: if not None then model hyperparameters expected - 'buffer' (as in buffer size) and 'epochs'
    :param shuffle_and_repeat: if the input is to be shuffled and repeat-delivered (say per epoch)
    :return: instance of tf.data.Dataset
    """

    params = params if params is not None else {}

    # shapes are analogous to (list of encoded words, len(words)), list of encoded tags
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)

    defaults = (('<pad>', 0), 'O')

    generator = functools.partial(generator_fn, words_file, tags_file)
    dataset = tf.data.Dataset.from_generator(generator, output_shapes = shapes, output_types = types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).reduce(params['epochs'])

    dataset = dataset.padded_batch(params.get('batch_size', 20), shapes, defaults).prefetch(1)\

    return dataset

if __name__ == '__main__':
    dataset = input_fn(Path(DATADIR + '/words.txt'), Path(DATADIR + '/tags.txt'))
    iterator = dataset.make_one_shot_iterator()
    node = iterator.get_next()
    with tf.Session() as sess:
        print(sess.run(node))