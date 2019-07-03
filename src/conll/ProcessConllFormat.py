__author__ = "Vikas Bahirwani"

DATADIR = "../../data/conll"

from pathlib import Path

def convertToSentences(conll_filename):
    with Path(conll_filename).open('r') as f:
        words = []
        pos_tags = []
        chunk_tags = []
        ner_tags = []

        for i, line in enumerate(f):
            if line != '\n':
                splits = line.strip().split()
                words.append(splits[0])
                pos_tags.append(splits[1])
                chunk_tags.append(splits[2])
                ner_tags.append(splits[3])
            else:
                yield ' '.join(words), ' '.join(pos_tags), ' '.join(chunk_tags), ' '.join(ner_tags)
                words = []
                pos_tags = []
                chunk_tags = []
                ner_tags = []

if __name__ == '__main__':
    file_prefixes = ["train", "testa", "testb"]

    for prefix in file_prefixes:
        words_filename = str(Path(DATADIR, "{}.words.txt".format(prefix)))
        tags_filename = str(Path(DATADIR, "{}.tags.txt".format(prefix)))

        with Path(words_filename).open('w') as fwords, Path(tags_filename).open('w') as ftags:
            for i, processed in enumerate(convertToSentences(str(Path(DATADIR, '{}.txt'.format(prefix))))):
                sentence, _, _, ner_tags = processed
                fwords.write("{}\n".format(sentence))
                ftags.write("{}\n".format(ner_tags))

                if i % 100 == 0:
                    print("{} {} lines processed".format(prefix, i))

            print("{} A total of {} lines processed".format(prefix, i))
