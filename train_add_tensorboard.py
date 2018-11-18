import pandas as pd
import nltk
import re
import json
import os, sys
import tensorflow as tf
import numpy as np
import datetime
import time
from att_bilstm import BiLSTMAttention
from sklearn.metrics import f1_score
import data_helpers
import dataset_function
import train_test
import warnings
import sklearn.exceptions
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
# train.py
tf.flags.DEFINE_string("pair_mid_dir", "/home/ked317/NN/Jiaxin_Liu/data/mention_id/pair_mid", "Path of json file:{'pair':['mention_ids']}")
tf.flags.DEFINE_string("id_mention_dir", "/home/ked317/NN/Jiaxin_Liu/data/mention_id/mid_sentence", "Path of json file: {'id':'mention'}")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("max_sentence_length", 100, "Max sentence length in train(98)/test(70) data (Default: 100)")
tf.flags.DEFINE_string("log_dir", "tensorboard", "Path of tensorboard")
# train_test.py
tf.flags.DEFINE_string("train_dir", "/home/ked317/NN/new_dataset/train_dataset", "Path of txt file which contains training data")
tf.flags.DEFINE_string("test_dir", "/home/ked317/NN/new_dataset/test_dataset", "Path of txt file which contains testing data")
tf.flags.DEFINE_string("dir", "/home/ked317/NN/new_dataset", "path of new dataset(same as Jiaxin)")
tf.flags.DEFINE_string("pair_id_dir", "/home/ked317/NN/Jiaxin_Liu/data/mention_id/pair_mid", "Path of json file:{'pair':['mention_ids']}")
# tensorboard
tf.flags.DEFINE_string("summaries_dir", "/home/ked317/NN/logs/1", "Path of save checkpoint")
# Model Hyperparameters
tf.flags.DEFINE_string("word2vec", "GoogleNews-vectors-negative300.bin", "Word2vec file with pre-trained embeddings")  # ../GoogleNews-vectors-negative300.bin
tf.flags.DEFINE_integer("text_embedding_dim", 300, "Dimensionality of character embedding (Default: 300)")
tf.flags.DEFINE_string("layers", "500", "Size of rnn output (Default: 500")
tf.flags.DEFINE_float("dropout_keep_prob1", 0.3, "Dropout keep probability for embedding, LSTM layer(Default: 0.3)")
tf.flags.DEFINE_float("dropout_keep_prob2", 0.5, "Dropout keep probability attention layer (Default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1e-5, "L2 regularization lambda (Default: 1e-5)")
tf.flags.DEFINE_boolean("use_ranking_loss", True, "Use ranking loss (Default : True)")
tf.flags.DEFINE_float("gamma", 2.0, "scaling parameter (Default: 2.0)")
tf.flags.DEFINE_float("mp", 2.5, "m value for positive class (Default: 2.5)")
tf.flags.DEFINE_float("mn", 0.5, "m value for negative class (Default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (Default: 100)")  # 100 epochs - 11290 steps
tf.flags.DEFINE_integer("display_every", 10, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-10, "Which learning rate to start with. (Default: 1e-3)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# ===============================================================
# FLAGS.__flags is a dictionary:
# {'max_sentence_length': 100, 'log_dir': 'tensorboard', 'dev_sample_percentage': 0.1,
# 'train_dir': 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'}
# tf.flags.DEFINE_XXX(flag name,value,some helpful descriotion)
# ===============================================================
FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
FLAGS(sys.argv）
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # sorted here is to sort the elements by their first letter
	print("{} = {}".format(attr.upper(), value))
print("")

# ===============================================================
# Define which device to use.
# "/cpu:0": The CPU of your machine.
# "/device:GPU:0": The GPU of your machine, if you have one.
# "/device:GPU:1": The second GPU of your machine, etc.
# ===============================================================
def train():
	# Data preprocessinng
	with tf.device('/cpu:0'): # Define which device to use.
    # ===============================================================
    # Load testing data and its ground truth
    # x_test is a list which contains all sentences in testinng data
    # y is an array which containns ground truth for each sentence
    # ===============================================================
		#dataset=dataset_function.generate_dataset(FLAGS.id_mention_dir,FLAGS.pair_mid_dir)
		train_d,test_d=train_test.load_pair_name(FLAGS.dir,FLAGS.pair_id_dir)
		id_mention_dir=FLAGS.id_mention_dir
		train,test=train_test.generate_dataset(train_d,test_d,id_mention_dir)
		x_train, y_train = data_helpers.load_data_and_labels(train)
		print(y_train)
		x_test, y_test = data_helpers.load_data_and_labels(test)

	# Build vocabulary
	# Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
	# ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
	# =>
	# [27 39 40 41 42  1 43  0  0 ... 0]
	# dimension = FLAGS.max_sentence_length
	text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length) # build a vocabulary dictionary model
    # learn the vocabulary dictionary annd return inndexies of words, each sentences has been change to a list of index(in dictionary). and sentences be collected in a outer list--> change to array.
	train_vec = np.array(list(text_vocab_processor.fit_transform(x_train)))
	test_vec = np.array(list(text_vocab_processor.fit_transform(x_test)))
	print("Train Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_))) #
	print("Test Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_))) #

	x_train = np.array([list(i) for i in train_vec])
	x_test = np.array([list(i) for i in test_vec])
    
	print("x_train = {0}".format(x_train.shape))
	print("y_train = {0}".format(y_train.shape))
	print("x_test = {0}".format(x_test.shape))
	print("y_test = {0}".format(y_test.shape))

	# Randomly shuffle data
	np.random.seed(10)
	shuffle_indices_train = np.random.permutation(np.arange(len(y_train)))
	x_train = x_train[shuffle_indices_train]
	y_train = y_train[shuffle_indices_train]
	np.random.seed(10)
	shuffle_indices_test = np.random.permutation(np.arange(len(y_test)))
	x_test = x_test[shuffle_indices_test]
	y_test = y_test[shuffle_indices_test]

	# Split train/test set
	# TODO: This is very crude, should use cross-validation
	#dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
	#x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
	#y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
	#print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))
	#print(x_train)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
	# Save every N iterations
	save_every_n = 200
	# Model
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		# Tensoboard
#		train_writer = tf.summary.FileWriter('./logs/1/train', sess.graph)
		model = BiLSTMAttention(layers=FLAGS.layers, max_length=FLAGS.max_sentence_length, n_classes=y_train.shape[1],
					vocab_size=len(text_vocab_processor.vocabulary_), embedding_size=FLAGS.text_embedding_dim, batch_size=FLAGS.batch_size,
					l2_reg_lambda=FLAGS.l2_reg_lambda, gamma=FLAGS.gamma, mp=FLAGS.mp, mn=FLAGS.mn, use_ranking_loss=FLAGS.use_ranking_loss)
		# writer = tf.summary.FileWriter("C:/Users/Kwon/workspace/Python/relation-extraction/Bidirectional-LSTM-with-attention-for-relation-classification/" + FLAGS.log_dir, sess.graph)
		# writer.add_graph(sess.graph)
		# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train')
		test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
		tf.global_variables_initializer().run()
		# Output directory for models and summaries
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
		print("Writing to {}\n".format(out_dir))

		# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

		# Write vocabulary: Saves vocabulary processor into given file.
		text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))

		sess.run(tf.global_variables_initializer())

		# Pre-trained word2vec
		if FLAGS.word2vec:
			# initial matrix with random uniform
			initW = np.random.uniform(-0.25, 0.25, (len(text_vocab_processor.vocabulary_), FLAGS.text_embedding_dim))
			# load any vectors from the word2vec
			print("Load word2vec file {0}".format(FLAGS.word2vec))
			with open(FLAGS.word2vec, "rb") as f:
				header = f.readline()
				print(header)
				vocab_size, layer1_size = map(int, header.split())
				binary_len = np.dtype('float32').itemsize * layer1_size
				for line in range(vocab_size):
					word = []
					while True:
						ch = f.read(1).decode('latin-1')
						if ch == ' ':
							word = ''.join(word)
							break
						if ch != '\n':
							word.append(ch)
					idx = text_vocab_processor.vocabulary_.get(word)
					if idx != 0:
						initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
					else:
						f.read(binary_len)
			sess.run(model.W_emb.assign(initW))
			print("Success to load pre-trained word2vec model!\n")
        # split data into different batch and feed it into nerual network by epochs
		batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
		# batches = data_helpers.batch_iter(list(zip(x_shuffled, y_shuffled)), FLAGS.batch_size, FLAGS.num_epochs)

		max_f1 = -1
        
		for step, batch in enumerate(batches):
#			counter+=1
			x_batch, y_batch = zip(*batch)

			feed_dict = {model.input_text: x_batch, model.dropout_keep_prob1: FLAGS.dropout_keep_prob1, model.dropout_keep_prob2: FLAGS.dropout_keep_prob2, model.labels: y_batch}
			# top2i0, cond, output, pos_out, pos_index, neg_out, neg_index, max_index, _, loss, accuracy = \
			# 	sess.run([model.top2i, model.cond, model.output, model.pos_out, model.pos_index, model.neg_out, model.neg_index, model.max_index, model.train, model.cost, model.accuracy], feed_dict=feed_dict)
            # sess.run(y,feed_dict): feed data in feed_dict into y
			summary, _, loss, accuracy = sess.run([merged, model.train, model.cost, model.accuracy], feed_dict=feed_dict)
			# writer.add_summary(sum, global_step=step)
			train_writer.add_summary(summary, step)
			# Training log display
			if step % FLAGS.display_every == 0:
#				saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, layers))
				print("step {}:, loss {}, acc {}".format(step, loss, accuracy))

			# Evaluation
			if step % FLAGS.evaluate_every == 0:
				print("\nEvaluation:")
				feed_dict = {
					model.input_text: x_test,
					model.labels: y_test,
					model.dropout_keep_prob1: 1.0,
					model.dropout_keep_prob2: 1.0
				}
				summary,loss, accuracy, predictions = sess.run(
					[merged,model.cost, model.accuracy, model.predictions], feed_dict)
				test_writer.add_summary(summary, step)                               
                # calculate F1 score
				f1 = f1_score(np.argmax(y_test, axis=1), predictions, average="macro")
				print("step {}:, loss {}, acc {}, f1 {}\n".format(step, loss, accuracy, f1))
				# print(predictions[:5])

				# Model checkpoint
				# Decide if we should save model checkpoint in this epoch
				if f1 > max_f1 * 0.99:
					path = saver.save(sess, checkpoint_prefix, global_step=step)
					print("Saved model checkpoint to {}\n".format(path))
					max_f1 = f1
#		saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, layers))
#		fpr_keras, tpr_keras, thresholds_keras = roc_curve(model.labels, model.predictions)
#		auc_keras = auc(fpr_keras, tpr_keras)
#		plt.figure(1)
#		plt.plot([0, 1], [0, 1], 'k--')
#		plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#		plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
#		plt.xlabel('False positive rate')
#		plt.ylabel('True positive rate')
#		plt.title('ROC curve')
#		plt.legend(loc='best')
#		plt.show()
        
        #tf.metrics.auc(model.labels,model.predictions,weights=None,num_thresholds=200,metrics_collections=None,updates_collections=None,curve='ROC',name=None,summation_method='trapezoidal')
def main(_):
	train()
