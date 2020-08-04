
from __future__ import absolute_import, print_function, division
import os
import sys
import re
import datetime
import numpy as np
import tensorflow as tf
from prep_data import generate_online

class C_RNN(object):

	def __init__(self, params):

		self.params = params
		self.graph = tf.Graph()

		with self.graph.as_default():
			self.build_graph()

	def build_graph(self):

		self.pattern_X = tf.placeholder(shape=[None, None], dtype=tf.int32, name="pattern_input")
		self.intention_X = tf.placeholder(shape=[None, None], dtype=tf.int32,  name="intention_input")
		self.Y = tf.placeholder(shape=[None, None], dtype=tf.int32, name="pattern_target")

		# embedding for pattern vocabulary
		pattern_emb = tf.Variable(
			tf.random_uniform([self.params.vocab_size, self.params.vocab_emb], -1.0, 1.0),
			name="embeddings")
		pattern_emb_lookup = tf.nn.embedding_lookup(pattern_emb, self.pattern_X)
		# or just use one-hot vector for tag #

		# ver1. concatenate above two embeddings #
		# embedding for intention tag #
		intention_emb = tf.Variable(
				tf.random_uniform([self.params.tag_size, self.params.intent_emb], -1.0, 1.0),
				name="embeddings")
		intention_emb_lookup = tf.nn.embedding_lookup(intention_emb, self.intention_X)
		#print(pattern_emb_lookup)
		#print(intention_emb_lookup)

		self.all_X = tf.concat([pattern_emb_lookup, intention_emb_lookup], axis=-1)
		#print(self.all_X)

		# lstm unravel #
		def RNN(X, init_state):
			with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
				cells = tf.contrib.rnn.MultiRNNCell(
					[tf.contrib.rnn.GRUCell(self.params.num_hidden) for _ in range(self.params.num_cells)])
				outputs, states = tf.nn.dynamic_rnn(cell=cells, inputs=X, initial_state=init_state, dtype=tf.float32)
				logits = tf.layers.dense(inputs=outputs, units=self.params.vocab_size)
			return tf.nn.softmax(logits), logits, states

		self.initial_state = tf.placeholder(dtype=tf.float32, shape=[self.params.num_cells, 1, self.params.num_hidden])
		state_tuple = tuple([t for t in tf.unstack(self.initial_state)])
		#print(type(state_tuple))


		# projection #
		self.pred, logits, self.state = RNN(self.all_X, state_tuple)

		self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.Y)
		self.loss = tf.reduce_mean(self.loss)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate).minimize(self.loss)

		self.saver = tf.train.Saver(save_relative_paths=True)

		self.init = tf.global_variables_initializer()


	def _load_ckpt(self):
		ckpt = tf.train.get_checkpoint_state(self.params.logdir)
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			self.latest_global_step = int(re.search(                               # Retrieving lastest_global_step
				'\d+$', ckpt.model_checkpoint_path).group(0))
			self.latest_global_epoch = int(                                        # Retrieving lastest_global_epoch
				self.latest_global_step / self.params.total_steps)  # Retrieving lastest_global_epoch
			print("Restoring Variables...")
		else:
			self.sess.run(self.init)
			self.latest_global_epoch = 0
			self.latest_global_step = 0
			print("Initialize Variables...")

	def train(self):

		self.sess = tf.Session(graph=self.graph)
		with self.sess.as_default():

			self._load_ckpt()

			### Train
			for epoch in range(self.params.num_epoch - self.latest_global_epoch):

				total_loss = 0
				shuffle_list = np.random.permutation(self.params.total_steps)

				for each_gradient_step, current_index in enumerate(shuffle_list):
					train_x, train_y, train_i, train_x_length = generate_online(current_index)
					prev_state = np.zeros(shape=[self.params.num_cells, 1, self.params.num_hidden])
					train_x = np.array(train_x).reshape([1, train_x_length])
					train_y = np.array(train_y).reshape([1, train_x_length])
					train_i = np.array(train_i).reshape([1, train_x_length])

					oneline_loss, _, curr_state = self.sess.run(                                       # For error per epoch
						[self.loss, self.optimizer, self.state],                                     # For error per epoch
						feed_dict={self.pattern_X: train_x,
								   self.intention_X: train_i,                   # For error per epoch
								   self.Y: train_y,                   # For error per epoch
								   self.initial_state: prev_state})
					total_loss += oneline_loss                                        # For error per epoch
				average_loss = total_loss / self.params.total_steps       # For error per epoch
				print('Epoch {}: average_loss is {}'                                     # For error per epoch
					  .format(self.latest_global_epoch + epoch+1, average_loss))         # For error per epoch
				# Saving the model every 5 epochs
				if (epoch+1) % 5 == 0:
					savepath = os.path.join(self.params.logdir, "model_{}.ckpt".format(self.latest_global_step))
					saved_file_path = self.saver.save(self.sess, savepath,
													  global_step=
													  self.latest_global_step +
													  ((epoch+1) * self.params.total_steps))
					print('Epoch {}: model saved in {}'.format(self.latest_global_epoch + epoch+1, saved_file_path))
		return


	def infer(self, intention):
		self.sess = tf.Session(graph=self.graph)
		with self.sess.as_default():
			ckpt = tf.train.get_checkpoint_state(self.params.logdir)
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			result = []
			num_sents = 0
			max_gen =0

			while num_sents < self.params.num_pattern:
				sent = []
				inf_x = np.array([0])
				prev_x = 0
				inf_int_x = np.array([intention]).reshape([1, 1])
				prev_state = np.zeros(shape=[self.params.num_cells, 1, self.params.num_hidden], dtype=np.float32)
				for i in range(self.params.max_inference):
					inf_x = inf_x.reshape([1, 1])
					infer_feed_dict = {self.pattern_X: inf_x, self.intention_X: inf_int_x,
									   self.initial_state: prev_state}
					current_preds, current_state = self.sess.run([self.pred, self.state], feed_dict=infer_feed_dict)
					#print(current_preds)
					inf_x_list = np.argwhere(current_preds > self.params.threshold)
					inf_x_list = inf_x_list[:,-1]

					if len(inf_x_list) == 0:
						inf_x = np.argmax(current_preds)

					else:
						inf_x = np.random.choice(inf_x_list, 1)
						inf_x = inf_x[0]

					if inf_x in range(1):
						inf_x = 0

					elif prev_x != inf_x:
						sent.append(inf_x)

					prev_x = inf_x
					#print(prev_state)
					prev_state = tuple_to_array(current_state)

					if inf_x == 1 and i != 0:
						break
					inf_x = np.array(inf_x)
					#print(sent)

				if sent not in result:
					result.append(sent)
					#print(result)
					num_sents += 1
				max_gen += 1
				if max_gen > 1000:
					break
				#print(result)

		return result

def tuple_to_array(state_tuple):
	return np.array([t for t in state_tuple])




