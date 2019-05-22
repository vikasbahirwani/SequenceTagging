import numpy as np
import os

# shared global variables, to be imported by the model as well
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

# special error message
class MyIOError(Exception):
    # Defines a custome file not found error message

    def __init__(self, filename):
        message = """ ERROR: Unable to locate file {:}.
FIX: Have you tried running python build_data.py first?

This will build vocab file from your train, test and dev sets and
trim your word vectors.
""".format(filename)

        super(MyIOError, self).__init__(message)

class CoNLLDataset(object):
    """Class iterates over the CoNLL dataset.

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None, optional pre-processing is applied.

    Example:
        ```python
        data = CoNLLDataset(filename)

        for words, tags in data:
            print("{:} {:}".format(words, tags))
    """

    def __init__(self, filename, processing_word = None, processing_tag = None, max_iter = None):
        """
        :param filename: path to dataset file
        :param processing_word: (optional) function that takes in a word as input and returns a "processed" word
        :param processing_tag: (optional) function that takes in a tag as input and returns a "processed" tag
        :param max_iter: (optional) max number of sentences to yield
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None

    def __iter__(self):
        niter = 0

        with open(self.filename) as f:
            words, tags = []

            for line in f:
                line = line.strip()
                if(len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is None or niter <= self.max_iter:
                            yield words, tags
                        else:
                            break
                else:
                    ls = line.split(' ') # split on space
                    word, tag = ls[0], ls[1]

                    if self.processing_word is not None:
                        word = self.processing_word(word)

                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)

                    words += [word]
                    tags += [tag]

    def __len__(self):
        """Iterates over the entire dataset to count number of sentences"""
        if self.length is not None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

def get_vocabs(datasets):
    """Builds vocabulary of words and tags from given list of dataset objects

    :param datasets: list of datasets from which vocab should be derived
    :return: a tuple of the format (vocab_words, vocab_tags)
    """

    print("Building words and tags vocab.")
    vocab_words = set()
    vocab_tags = set()

    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)

    print(" - finished building words and tags vocab. # of words = {:}, # of tags = {:}".format(len(vocab_words), len(vocab_tags)))

    return vocab_words, vocab_tags

def get_char_vocab(dataset):
    """Builds vocabulary of characters from a given dataset object

    :param dataset: dataset from which vocab should be derived
    :return: vocabulary (set) of characters
    """

    print("Building char vocab.")
    vocab_chars = set()
    for words,_ in dataset:
        for word in words:
            vocab_chars.update(word)

    print(" - finished building chars vocab. # of characters = {:}".format(len(vocab_chars)))

    return vocab_chars

def get_glove_vocab(filename):
    """Build vocab from glove file

    :param filename: path to glove file
    :return: vocabulary (set) of words included in the glove file provided
    """

    print("Building glove vocab.")
    vocab_glove = set()

    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab_glove.add(word)

    print(" - finished building glove vocab. # of glove words = {:}".format(len(vocab_glove)))
    return vocab_glove

def write_vocab(vocab, filename):
    """Writes vocab to given file

    :param vocab: the vocabulary to save
    :param filename: full file path including it's name
    :return writes one word per line
    """

    print("Writing vocab to file")

    word_count = len(vocab)
    with open(filename, "w") as f:
        for i,word in enumerate(vocab):
            if i != word_count:
                f.write("{:}\n".format(word))
            else:
                f.write(word)

    print(" - done. {:} word types written".format(word_count))

def load_vocab(filename):
    """ Loads the vocab from a given file

    :param filename: full file path including it's name
    :return: dictionary (say d) of the format d[word] = index
    """
    try:
        vocab = dict()
        with open(filename, 'r') as f:
            for index, line in enumerate(f):
                word = line.strip()
                vocab[word] = index

        return vocab

    except IOError:
        raise MyIOError(filename)

def export_trimmed_glove_embeddings(vocab, glove_filename, trimmed_filename, dim):
    """For words in the vocab, extracts the glove embeddings from the glove file,
    creates a numpy array of the format [vocab[word] = idx, glove vector],
    and saves this numpy array (a.k.a. trimmed glove embeddings) to trimmed_filename

    :param vocab: dictionary of the format vocab[word] = idx
    :param glove_filename: file containing glove vectors of dim dimension
    :param trimmed_filename: file to save the trimmed glove vectors
    :param dim: dimension of glove vectors
    """

    embeddings = np.zeros((len(vocab), dim))

    with open(glove_filename, 'r') as f:
        for line in f:
            splits = line.split(' ')
            word = splits[0]
            embedding = [float(x) for x in splits[1:]]

            if word in vocab:
                # np.asarray() does not make a new copy of embedding while np.array() does
                # though here we may be fine with either given that embedding is a local variable
                # and embeddings are saved to a file
                embeddings[vocab[word]] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings = embeddings) 







