# taken from https://github.com/guillaumegenthial/tf_ner/blob/master/data/example/build_vocab.py
# edited to fit my project structure
# original__author__ = "Guillaume Genthial"

"""Script to build words, chars and tags vocab"""

__author__ = "Vikas Bahirwani"

from collections import Counter
from pathlib import Path

# TODO: modify this depending on your needs (1 will work just fine)
# You might also want to be more clever about your vocab and intersect
# the GloVe vocab with your dataset vocab, etc. You figure it out ;)
MINCOUNT = 1

DATADIR = "../data/conll"

if __name__ == '__main__':
    # 1. Words
    # Get Counter of words on all the data, filter by min count, save
    def words(name):
        return Path(DATADIR, '{}.words.txt'.format(name))

    print('Build vocab words (may take a while)')
    counter_words = Counter()

    # TODO: Here vocab is created using train and test data. Ideally, only use train
    for n in ['train', 'testa', 'testb']:
        with Path(words(n)).open() as f:
            for line in f:
                counter_words.update(line.strip().split())

    vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}

    with Path(DATADIR, 'vocab.words.txt').open('w') as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(len(vocab_words), len(counter_words)))

    # 2. Chars
    # Get all the characters from the vocab words
    print('Build vocab chars')
    vocab_chars = set()
    for w in vocab_words:
        vocab_chars.update(w)

    with Path(DATADIR, 'vocab.chars.txt').open('w') as f:
        for c in sorted(list(vocab_chars)):
            f.write('{}\n'.format(c))
    print('- done. Found {} chars'.format(len(vocab_chars)))

    # 3. Tags
    # Get all tags from the training set

    def tags(name):
        return Path(DATADIR, '{}.tags.txt'.format(name))

    print('Build vocab tags (may take a while)')
    vocab_tags = set()
    with Path(tags('train')).open() as f:
        for line in f:
            vocab_tags.update(line.strip().split())

    with Path(DATADIR, 'vocab.tags.txt').open('w') as f:
        for t in sorted(list(vocab_tags)):
            f.write('{}\n'.format(t))
    print('- done. Found {} tags.'.format(len(vocab_tags)))