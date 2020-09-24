echo 'Download Dataset' && \
mkdir examples && \
wget -P ./examples https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip && \
wget -P ./examples https://data.broadinstitute.org/bbbc/BBBC038/stage1_train_labels.csv && \
unzip examples/stage1_train.zip -d ./examples/raw_data && \

echo 'Preprocess Dataset (takes > 30 min...)' && \
python preprocess.py && \
rm -r ./examples/stage1_train.zip ./examples/raw_data ./examples/stage1_train_labels.csv