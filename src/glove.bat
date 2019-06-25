@echo OFF
REM run this file if using windows OS
REM this batch file downloads and unzips glove in the data folder

mkdir data
wget -P ../data/ "http://nlp.stanford.edu/data/glove.840B.300d.zip" --no-check-certificate
unzip ../data/glove.840B.300d.zip -d ../data/
rm ../data/glove.840B.300d.zip