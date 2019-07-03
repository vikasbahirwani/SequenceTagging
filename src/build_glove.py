# taken from https://github.com/guillaumegenthial/tf_ner/blob/master/data/example/build_glove.py
# edited to fit my project structure
# original__author__ = "Guillaume Genthial"

"""Build an np.array from some glove file and some vocab file
You need to download `glove.840B.300d.txt` from
https://nlp.stanford.edu/projects/glove/ and you need to have built
your vocabulary first (Maybe using `build_vocab.py`)
"""

__author__ = "Vikas Bahirwani"

from pathlib import Path

import numpy as np

DATADIR = '../data/conll'
GLOVEDIR = '../data'

if __name__ == '__main__':
    # Load vocab
    with Path(DATADIR, 'vocab.words.txt').open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Array of zeros
    embeddings = np.zeros((size_vocab, 300))

    # Get relevant glove vectors
    found = 0
    print('Reading GloVe file (may take a while)')
    with Path(GLOVEDIR, 'glove.840B.300d.txt').open('r', encoding="utf8") as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    # Save np.array to file
    glove_npz_path = str(Path(DATADIR, 'glove.npz'))
    np.savez_compressed(glove_npz_path, embeddings=embeddings)