# SequenceTagging
- Tutorial reference: https://guillaumegenthial.github.io/introduction-tensorflow-estimator.html#define-a-bi-lstm--crf-model_fn

## Steps taken:
- Get conll data (train, testa, testb) from https://github.com/Franck-Dernoncourt/NeuroNER/tree/master/neuroner/data/conll2003/en
- Run src/conll/ProcessConllFormat.py
- glove.bat
- Run build_vocab.py
- Run build_glove.py
- Run lstm_crf/main.py
- Run conlleval

Metrics thus far:
- Train f1: 98.1
- Testa f1: 93.54
- Testb f1: 89.62 (Paper benchmark: 90.1)
