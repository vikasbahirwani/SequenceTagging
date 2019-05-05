@echo OFF
REM run this file if using windows OS
REM this batch file downloads and unzips glove in the data folder

mkdir data
wget -P ./data/ "http://nlp.stanford.edu/data/glove.6B.zip" --no-check-certificate
unzip ./data/glove.6B.zip -d data/glove.6B/
rm ./data/glove.6B.zip