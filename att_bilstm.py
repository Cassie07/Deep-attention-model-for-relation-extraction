# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# ===========================================================================================
# tf.reduce_max: get the maximum
# tf.sign: sign function(-1:x<0;0:x=0;1:x>0)
# Regular expressionn
# re.sub('[A-Za-z0-9]') means only save letters and numbers. Without [] means save every thing
# re.sub(',',' , ',string) means change a,b to a , b
# ============================================================================================
def get_length(sequence):
		used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
		length = tf.reduce_sum(used, 1)
		length = tf.cast(length, tf.int32)
		return length

# Useless
def get_by_label(tensor, label, n_classes):
	labels = tf.argmax(label, 1, output_type=tf.int32)  # (?, )
	label_index = tf.range(0, tf.shape(tensor)[0]) * n_classes + labels
	flat_tensor = tf.reshape(tensor, [-1])
	return tf.gather(flat_tensor, label_index), label_index  # (?, 19) -> (?,)

# Useless
def get_neg_out(tensor, label, n_classes):
	print("input tensor shape : ", np.shape(tensor))
    # Returns the index with the largest value across axes of a tensor. (deprecated arguments)
	labels = tf.argmax(label, 1, output_type=tf.int32)  # (?, )
	# label_index = tf.range(0, tf.shape(tensor)[0]) * n_classes + labels

	top2v, top2i = tf.nn.top_k(tensor, 2)
	# max_index = top2i[:, 0]
	max_index = tf.range(0, tf.shape(tensor)[0]) * n_classes + top2i[:, 0]

	lab_max_comp = tf.equal(labels, top2i[:, 0])  # 라벨이랑 같으면 true true.. 다르면 false => false를 찾고
	# lab_2max_comp = tf.equal(label_index, top2i[1])  #true인 애들을 찾고
	index = tf.where(lab_max_comp, x=top2i[:, 1], y=top2i[:, 0])
	neg_index = tf.range(0, tf.shape(tensor)[0]) * n_classes + index
	flat_tensor = tf.reshape(tensor, [-1])
	return tf.gather(flat_tensor, neg_index), neg_index, max_index, lab_max_comp, top2i[:, 0]


class BiLSTMAttention(object):
	def __init__(self, layers, max_length, n_classes, vocab_size, embedding_size=300, batch_size = 64, l2_reg_lambda=1e-5, gamma=2.0, mp=2.5, mn=0.5, use_ranking_loss=False):
# =============================================================================================================
# placeholder:  A placeholder is simply a variable that we will assign data to at a later date.
#               It allows us to create our operations and build our computation graph, without needing the data.
# ==============================================================================================================
        # initialization
		self.input_text = tf.placeholder(tf.int32, shape=[None, max_length], name="input_text")
		self.labels = tf.placeholder(tf.int32, shape=[None, n_classes])
		self.dropout_keep_prob1 = tf.placeholder(tf.float32, name='dropout_keep_prob1') # Dropout keep probability for embedding, LSTM layer
		self.dropout_keep_prob2 = tf.placeholder(tf.float32, name='dropout_keep_prob2') # Dropout keep probability attention layer
        # tf.get_variable(name，[shape])
		# tf.nn: neural network Operations
		self.W_emb = tf.get_variable("word_embeddings", [vocab_size, embedding_size])
		embedding_dropout = tf.nn.dropout(self.W_emb, keep_prob=self.dropout_keep_prob1)
		rnn_input = tf.nn.embedding_lookup(embedding_dropout, self.input_text)
        # timestep
		seq_len = get_length(rnn_input)


		# lstm start here!
		layers = list(map(int, layers.split('-')))
        # tf.nn.rnn_cell.DropoutWrapper: Create a cell with added input, state, and/or output dropout.
        # tf.nn.rnn_cell.LSTMCell: Initialize the parameters for an LSTM cell.
		cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(h, activation=tf.nn.tanh, state_is_tuple=True), output_keep_prob=self.dropout_keep_prob2) for _, h in enumerate(layers)]
        # tf.nn.rnn_cell.MultiRNNCell: Create a RNN cell composed sequentially of a number of RNNCells.
        # cells: list of RNNCells that will be composed in this order.
		multi_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        # tf.nn.bidirectional_dynamic_rnn: Creates a dynamic version of bidirectional recurrent neural network.
        # cell_fw: An instance of RNNCell, to be used for forward direction.
        # cell_bw: An instance of RNNCell, to be used for backward direction.
        # inputs: The RNN inputs. If time_major == False (default), this must be a tensor of shape: [batch_size, max_time, input_size]. If time_major == True, this must be a tensor of shape: [max_time, batch_size, input_size]. [batch_size, input_size].
        # sequence_length: An int32/int64 vector, size [batch_size], containing the actual lengths for each of the sequences.
        # return: A tuple (outputs, output_states) where: outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output Tensor.
		rnn_out, _state = tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_cells, cell_bw=multi_cells, inputs=rnn_input, sequence_length=seq_len, dtype=tf.float32)
        # Concatenates tensors along one dimension.
		rnn_out = tf.concat(rnn_out, 2)
		# rnn_out = tf.reduce_sum(rnn_out, axis=0) #concat producces better results. maybe due to more parameters?


        # attention layer start here!
		att_vec_param = tf.get_variable("att_vec_param", [layers[-1] * 2])
		# att_vec_param = tf.get_variable("att_vec_param", layers[-1])
        # M=tan(H) (9)
		tan_rnn_out = tf.tanh(rnn_out)
		# att_vec = tf.tensordot(sim_mat_tan, tf.transpose(att_vec_param), axes=1) # (B,T,D) * (D, 1)=> (B, T)
        # alpha=softmax(transpose（w)*M）(10)
		att_vec = tf.einsum('aij,j->ai', tan_rnn_out, tf.transpose(att_vec_param))
		att_sm_vec = tf.nn.softmax(att_vec)
		# r=H*transpose(alpha)
		rnn_attention = tf.einsum('aij,ai->aj', rnn_out, att_sm_vec)  # (B, T, D) * (B, T) => (B, D)
        # dropout
		rnn_attention = tf.nn.dropout(rnn_attention, keep_prob=self.dropout_keep_prob2)

		max_val = np.power(6/(n_classes + layers[-1]), 1/2)
		C_emb = tf.get_variable("c_emb", [layers[-1] * 2, n_classes], initializer=tf.random_uniform_initializer(minval=-max_val, maxval=max_val))
		# C_emb = tf.get_variable("c_emb", [layers[-1], n_classes], initializer=tf.random_uniform_initializer(minval=-max_val, maxval=max_val))
        # tf.matmul: Multiplies matrix a by matrix b, producing a * b.
		output = tf.matmul(rnn_attention, C_emb)  # (B, D) * (D, C) => (B, C)
		# output = tf.nn.dropout(output, keep_prob=self.dropout_keep_prob2)
		self.output = output

		# evaluationn
		if use_ranking_loss:
			# pos_out, pos_index = get_by_label(output, self.labels, n_classes)
			# self.pos_out = pos_out
			# self.pos_index = pos_index

			# neg_out, neg_index, max_index, cond, top2i = get_neg_out(output, self.labels, n_classes)
			# self.neg_out = neg_out
			# self.neg_index = neg_index
			# self.max_index = max_index
			# self.cond = cond
			# self.top2i = top2i

			L = tf.constant(0.0)
			i = tf.constant(0)
			loop_cond = lambda i, L: tf.less(i, tf.shape(self.labels)[0])

			def loop_body(i, L):
				pos_label = tf.argmax(self.labels, 1, output_type=tf.int32)[i]
				# tf.nn.top_k: Finds values and indices of the k largest entries for the last dimension.
                # For matrices (resp. higher rank input), computes the top k entries in each row (resp. vector along the last dimension).
                # Return:
				# values: The k largest elements along each last dimensional slice.
                # indices: The indices of values within the last dimension of input.
				_, neg_indices = tf.nn.top_k(output[i, :], k=2)
                # Returns the truth value of (x == y) element-wise.
				cond = tf.equal(pos_label, neg_indices[0])
                # result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
                # If x < y, the tf.add operation will be executed and tf.square operation will not be executed.
				max_neg_index = tf.cond(cond, lambda: neg_indices[1], lambda: neg_indices[0])
				pos_out = output[i, pos_label]
				neg_out = output[i, max_neg_index]
				l = tf.log(1.0 + tf.exp(gamma*(mp-pos_out))) + tf.log(1.0+tf.exp(gamma*(mn+neg_out)))
				return [tf.add(i, 1), tf.add(L, l)]

			_, L = tf.while_loop(loop_cond, loop_body, loop_vars=[i, L])
			batch_size_f = tf.to_float(tf.shape(self.labels)[0])

			# self.cost = tf.reduce_sum(tf.log(1 + tf.exp(gamma * (mp - pos_out))) + tf.log(1 + tf.exp(gamma * (mn - neg_out))))
			# self.cost = tf.reduce_sum(tf.log(1.0 + tf.exp(gamma * (mp - pos_out))))
			self.cost = L/batch_size_f
		else:
			# tf.reduce_mean: mean
			# tf.nn.softmax_cross_entropy_with_logits
			# Measures the probability error in discrete classification tasks in which the classes are mutually exclusive (each entry is in exactly one class).
			# For example, each CIFAR-10 image is labeled with one and only one label: an image can be a dog or a truck, but not both.
			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.labels))
		self.cost += l2_reg_lambda * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

		optimizer = tf.train.AdamOptimizer()
		self.train = optimizer.minimize(self.cost)
		# tf.summary.scalar("cost", self.cost)
		# self.summary = tf.summary.merge_all()

		# classification
		self.predictions = tf.argmax(output, 1, name="predictions")
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(self.labels, 1)), tf.float32))
