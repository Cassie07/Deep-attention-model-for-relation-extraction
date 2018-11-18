# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import nltk
import re
import json
import os, sys

# ===========================================================================================
# clean_str: extract words we want or delete punctuation we don't want
# Regular expressionn
# re.sub('[A-Za-z0-9]') means only save letters and numbers. Without [] means save every thing
# re.sub(',',' , ',string) means change a,b to a , b
# ============================================================================================
def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9()<>/,!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()


def load_data_and_labels(dataset):
	#data = []
	# read by line and add all elements into a list
	#lines = [line.strip() for line in open(path)]
	#for idx in range(0, len(lines), 4): # start: 0. end: lines.length. selected_interval:4(0,4,8,12,...)
	#	id = lines[idx].split("\t")[0]
	#	relation = lines[idx + 1] # extract relation(string)
	#	sentence = lines[idx].split("\t")[1][1:-1]# delete "(index=0) and "(index=the last word in sentence) in sentence
	#	sentence = sentence.replace("<e1>", "<e1> ").replace("</e1>", " </e11>") # replace the front by the back
	#	sentence = sentence.replace("<e2>", "<e2> ").replace("</e2>", " </e22>")
	#	sentence = clean_str(sentence) # delete
		# data.append([id, sentence, e1, e2, relation])
	#	data.append([id, sentence, relation])

	# df = pd.DataFrame(data=data, columns=["id", "sentence", "e1_pos", "e2_pos", "relation"])
    # a structure like dictionary
	df = pd.DataFrame(data=dataset, columns=["id", "sentence", "relation"])
	labelsMapping = {'+(e2, e1)': 1,'+(e1, e2)':1,'-(e2, e1)':0,'-(e1, e2)':0}
    # transfer string label to int lable
	df['label'] = [labelsMapping[r] for r in df['relation']]

	x_text = df['sentence'].tolist()

	# pos1, pos2 = get_relative_position(df)

	# Label Data
	y = df['label']
	labels_flat = y.values.ravel() #a contiguous flattened array.
	# count the total numbers of labels
	labels_count = np.unique(labels_flat).shape[0]

	# convert class labels from scalars to one-hot vectors
	# 0  => [1 0 0 0 0 ... 0 0 0 0 0]
	# 1  => [0 1 0 0 0 ... 0 0 0 0 0]
	# ...
	# 18 => [0 0 0 0 0 ... 0 0 0 0 1]
	def dense_to_one_hot(labels_dense, num_classes):
		num_labels = labels_dense.shape[0]
		index_offset = np.arange(num_labels) * num_classes
		labels_one_hot = np.zeros((num_labels, num_classes))
		labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
		return labels_one_hot

	labels = dense_to_one_hot(labels_flat, labels_count)
	labels = labels.astype(np.uint8)

	# return x_text, pos1, pos2, labels
	return x_text, labels


def get_relative_position(df, max_sentence_length=100):
	# Position data
	pos1 = []
	pos2 = []
	for df_idx in range(len(df)):
		sentence = df.iloc[df_idx]['sentence']
		tokens = nltk.word_tokenize(sentence)
		e1 = df.iloc[df_idx]['e1_pos']
		e2 = df.iloc[df_idx]['e2_pos']

		d1 = ""
		d2 = ""
		for word_idx in range(len(tokens)):
			d1 += str((max_sentence_length - 1) + word_idx - e1) + " "
			d2 += str((max_sentence_length - 1) + word_idx - e2) + " "
		for _ in range(max_sentence_length - len(tokens)):
			d1 += "999 "
			d2 += "999 "
		pos1.append(d1)
		pos2.append(d2)

	return pos1, pos2


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		# if shuffle:
		# 	shuffle_indices = np.random.permutation(np.arange(data_size))
		# 	shuffled_data = data[shuffle_indices]
		# else:
		# 	shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			# yield shuffled_data[start_index:end_index]
			yield data[start_index:end_index]
