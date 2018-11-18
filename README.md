# Deep-attention-model-for-relation-extraction

reference: http://www.aclweb.org/anthology/P16-2034

please download GoogleNews-vectors-negative300.bin by yourself when you want to run train.py or train_add_tensorboard.py

new_dataset: a dataset which contains same split with Jiaxin.

att_bilstm.py: Model

data_helpers.py: Data preprocessing

data_preprocess.py: Functions might be used when we load data
                   (open folder and file/ load data)

dataset_function.py: Functions might be used when we load data
                     (generate a dataset:[[id1,mention1,relation1],[id2,mention2,relation2]...])

train.py: train

train_add_tensorboard.py: train + tensorboard

train_test.py
