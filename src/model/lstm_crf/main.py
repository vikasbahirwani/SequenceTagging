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

DATADIR = "../../../data/toy/"

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

def model_fn(features, labels, mode, params):
    """

    :param features: words from sentence and number of words per sentence
    :param labels: One tag per word
    :param mode:  tf.estimator.ModeKeys.TRAIN or  tf.estimator.ModeKeys.PREDICT or  tf.estimator.ModeKeys.EVAL
    :param params: dictionary of hyper parameters for the model
    :return:
    """

    # For serving, features are a bit different
    if isinstance(features, dict):
        features = features['words'], features['nwords']

    # Read vocab_words_file, vocab_tags_file, features
    words, nwords = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(params['vocab_words_file'], num_oov_buckets = params['num_oov_buckets'])

    '''
    If the file contains the following: 
    B-LOC
    B-PER
    O
    I-LOC
    
    then indices = [0, 1, 3] and num_tags = 4
    
    Open Question: The special treatment of tag indices is probably needed for microavg metrics. Why though?
    '''

    with Path(params['vocab_tags_file']).open('r') as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1

    # Word Embeddings
    # remember - as per the parse function "words" is a python list of
    word_ids = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']
    glove = np.vstack([glove, [[0.]*params['dim']]])
    variable = tf.Variable(glove, dtype=tf.float32, trainable=False)
    embeddings = tf.nn.embedding_lookup(variable, word_ids)
    dropout = params['dropout']
    embeddings = tf.layers.dropout(embeddings, rate = dropout, training = training)

    # LSTM CRF
    time_major = tf.transpose(embeddings, perm = [1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)

    """
    Any LSTM Cell returns two things: Cell Output (h) and Cell State (c)

    Following this, lstm_fw or lstm_bw each return a pair containing:

    Cell Output: A 3-D tensor of shape [time_len, batch_size, output_size]
    Final state: a tuple (cell_state, output) produced by the last LSTM Cell in the sequence.

    """
    output_fw,_ = lstm_cell_fw(time_major, dtype = tf.float32, sequence_length = nwords)
    output_bw,_ = lstm_cell_bw(time_major, dtype = tf.float32, sequence_length = nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)



if __name__ == '__main__':
    vocab_words = tf.contrib.lookup.index_table_from_file(str(Path(DATADIR, 'vocab.words.txt')), num_oov_buckets = 1)
    my_text = tf.constant([['San', 'Paris', 'Vikas'], ['I', 'live', 'in']])
    ids = vocab_words.lookup(my_text)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        print(ids.eval())