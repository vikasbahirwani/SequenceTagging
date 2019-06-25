# taken from https://github.com/guillaumegenthial/tf_ner/blob/master/data/example/Makefile
# use this file if running in linux

download-glove:
	wget -P ../data/ "http://nlp.stanford.edu/data/glove.840B.300d.zip"
	unzip ../data/glove.840B.300d.zip -d ../data/glove.840B.300d.txt
	rm ../data/glove.840B.300d.zip

build:
	python build_vocab.py
	python build_glove.py