from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange# pylint: disable=redefined-builtin
from six.moves import zip	 # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange# pylint: disable=redefined-builtin
import tensorflow as tf
import PersonalizedEmbedding
import LSE


class ProductSearchEmbedding_model(object):
	def __init__(self, vocab_size, review_size, user_size, product_size, query_max_length,
				 vocab_distribute, review_distribute, product_distribute, window_size,
				 embed_size, max_gradient_norm, batch_size, learning_rate, L2_lambda, query_weight,
				 net_struct, similarity_func, forward_only=False, negative_sample = 5):
		"""Create the model.
	
		Args:
			vocab_size: the number of words in the corpus.
			dm_feature_len: the length of document model features (query based).
			review_size: the number of reviews in the corpus.
			user_size: the number of users in the corpus.
			product_size: the number of products in the corpus.
			embed_size: the size of each embedding
			window_size: the size of half context window
			vocab_distribute: the distribution for words, used for negative sampling
			review_distribute: the distribution for reviews, used for negative sampling
			product_distribute: the distribution for products, used for negative sampling
			max_gradient_norm: gradients will be clipped to maximally this norm.
			batch_size: the size of the batches used during training;
			the model construction is not independent of batch_size, so it cannot be
			changed after initialization.
			learning_rate: learning rate to start with.
			learning_rate_decay_factor: decay learning rate by this much when needed.
			forward_only: if set, we do not construct the backward pass in the model.
			negative_sample: the number of negative_samples for training
		"""
		self.vocab_size = vocab_size
		self.review_size = review_size
		self.user_size = user_size
		self.product_size = product_size
		self.query_max_length = query_max_length
		self.negative_sample = negative_sample
		self.embed_size = embed_size
		self.window_size = window_size
		self.vocab_distribute = vocab_distribute
		self.review_distribute = review_distribute
		self.product_distribute = product_distribute
		self.max_gradient_norm = max_gradient_norm
		self.batch_size = batch_size * (self.negative_sample + 1)
		self.init_learning_rate = learning_rate
		self.L2_lambda = L2_lambda
		self.net_struct = net_struct
		self.similarity_func = similarity_func
		self.global_step = tf.Variable(0, trainable=False)
		if query_weight >= 0:
			self.Wu = tf.Variable(query_weight, name="user_weight", dtype=tf.float32, trainable=False)
		else:
			self.Wu = tf.sigmoid(tf.Variable(0, name="user_weight", dtype=tf.float32))
		# Feeds for inputs.
		self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
		self.review_idxs = tf.placeholder(tf.int64, shape=[None], name="review_idxs")
		self.user_idxs = tf.placeholder(tf.int64, shape=[None], name="user_idxs")
		self.product_idxs = tf.placeholder(tf.int64, shape=[None], name="product_idxs")
		self.word_idxs = tf.placeholder(tf.int64, shape=[None], name="word_idxs")
		self.query_word_idxs = tf.placeholder(tf.int64, shape=[None, self.query_max_length], name="query_word_idxs")
		self.PAD_embed = tf.get_variable("PAD_embed", [1,self.embed_size],dtype=tf.float32)

		# setup model
		print("Model Name " + self.net_struct)
		self.need_review = True
		if 'simplified' in self.net_struct:
			print('Simplified model')
			self.need_review = False
			
		self.need_context = False
		if 'hdc' in self.net_struct:
			print('Use context words')
			self.need_context = True

		if 'LSE' == self.net_struct:
			self.need_review = False
			self.need_context = True

		if self.need_context:
			self.context_word_idxs = []
			for i in xrange(2 * self.window_size):
				self.context_word_idxs.append(tf.placeholder(tf.int64, shape=[None], name="context_idx{0}".format(i)))

		print('L2 lambda ' + str(self.L2_lambda))

		# Training losses.
		self.loss = None
		if 'LSE' == self.net_struct:
			self.loss = LSE.build_embedding_graph_and_loss(self)
		else:
			self.loss = PersonalizedEmbedding.build_embedding_graph_and_loss(self)

		# Gradients and SGD update operation for training the model.
		params = tf.trainable_variables()
		if not forward_only:
			opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			self.gradients = tf.gradients(self.loss, params)
			
			self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
																	 self.max_gradient_norm)
			self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
											 global_step=self.global_step)
			
			#self.updates = opt.apply_gradients(zip(self.gradients, params),
			#								 global_step=self.global_step)
		else:
			if 'LSE' == self.net_struct:
				self.product_scores = LSE.get_product_scores(self, self.query_word_idxs)
			else:
				self.product_scores = PersonalizedEmbedding.get_product_scores(self, self.user_idxs, self.query_word_idxs)
	
		self.saver = tf.train.Saver(tf.global_variables())
	
	def step(self, session, learning_rate, user_idxs, product_idxs, query_word_idxs, review_idxs, 
					word_idxs, context_idxs, forward_only, test_mode = 'product_scores'):
		"""Run a step of the model feeding the given inputs.
	
		Args:
			session: tensorflow session to use.
			learning_rate: the learning rate of current step
			user_idxs: A numpy [1] float vector.
			product_idxs: A numpy [1] float vector.
			review_idxs: A numpy [1] float vector.
			word_idxs: A numpy [None] float vector.
			context_idxs: list of numpy [None] float vectors.
			forward_only: whether to do the update step or only forward.
	
		Returns:
			A triple consisting of gradient norm (or None if we did not do backward),
			average perplexity, and the outputs.
	
		Raises:
			ValueError: if length of encoder_inputs, decoder_inputs, or
			target_weights disagrees with bucket size for the specified bucket_id.
		"""
	
		# Input feed: encoder inputs, decoder inputs, target_weights, as provided.
		input_feed = {}
		input_feed[self.learning_rate.name] = learning_rate
		input_feed[self.user_idxs.name] = user_idxs
		input_feed[self.product_idxs.name] = product_idxs
		input_feed[self.query_word_idxs.name] = query_word_idxs
		input_feed[self.review_idxs.name] = review_idxs
		input_feed[self.word_idxs.name] = word_idxs
		if context_idxs != None:
			for i in xrange(2 * self.window_size):
				input_feed[self.context_word_idxs[i].name] = context_idxs[i]
	
		# Output feed: depends on whether we do a backward step or not.
		if not forward_only:
			output_feed = [self.updates,	# Update Op that does SGD.
						 #self.norm,	# Gradient norm.
						 self.loss]	# Loss for this batch.
		else:
			if test_mode == 'output_embedding':
				output_feed = [self.user_emb, self.product_emb, self.Wu, self.word_emb, self.word_bias]
				if self.need_review:
					output_feed += [self.review_emb, self.review_bias]
				
				if self.need_context and 'LSE' != self.net_struct:
					output_feed += [self.context_emb, self.context_bias]
				
			else:
				output_feed = [self.product_scores] #negative instance output
	
		outputs = session.run(output_feed, input_feed)
		if not forward_only:
			return outputs[1], None	# loss, no outputs, Gradient norm.
		else:
			if test_mode == 'output_embedding':
				return outputs[:4], outputs[4:]
			else:
				return outputs[0], None	# product scores to input user

	def setup_data_set(self, data_set, words_to_train):
		self.data_set = data_set
		self.words_to_train = words_to_train
		self.finished_word_num = 0
		if self.net_struct == 'hdc':
			self.need_context = True

	def intialize_epoch(self, training_seq):
		self.train_seq = training_seq
		self.review_size = len(self.train_seq)
		self.cur_review_i = 0
		self.cur_word_i = 0

	def get_train_batch(self):
		user_idxs, product_idxs, review_idxs, word_idxs, context_word_idxs = [],[],[],[],[]
		query_word_idxs = []
		learning_rate = self.init_learning_rate * max(0.0001, 
									1.0 - self.finished_word_num / self.words_to_train)
		review_idx = self.train_seq[self.cur_review_i]
		user_idx = self.data_set.review_info[review_idx][0]
		product_idx = self.data_set.review_info[review_idx][1]
		query_idx = random.choice(self.data_set.product_query_idx[product_idx])
		text_list = self.data_set.review_text[review_idx]
		text_length = len(text_list)
		while len(word_idxs) < self.batch_size:
			#print('review %d word %d word_idx %d' % (review_idx, self.cur_word_i, text_list[self.cur_word_i]))
			#if sample this word
			if self.data_set.sub_sampling_rate == None or random.random() < self.data_set.sub_sampling_rate[text_list[self.cur_word_i]]:
				user_idxs.append(user_idx)
				product_idxs.append(product_idx)
				query_word_idxs.append(self.data_set.query_words[query_idx])
				review_idxs.append(review_idx)
				word_idxs.append(text_list[self.cur_word_i])
				if self.need_context:
					i = self.cur_word_i
					start_index = i - self.window_size if i - self.window_size > 0 else 0
					context_word_list = text_list[start_index:i] + text_list[i+1:i+self.window_size+1]
					while len(context_word_list) < 2 * self.window_size:
						context_word_list += text_list[start_index:start_index+2*self.window_size-len(context_word_list)]
					context_word_idxs.append(context_word_list)

			#move to the next
			self.cur_word_i += 1
			self.finished_word_num += 1
			if self.cur_word_i == text_length:
				self.cur_review_i += 1
				if self.cur_review_i == self.review_size:
					break
				self.cur_word_i = 0
				review_idx = self.train_seq[self.cur_review_i]
				user_idx = self.data_set.review_info[review_idx][0]
				product_idx = self.data_set.review_info[review_idx][1]
				query_idx = random.choice(self.data_set.product_query_idx[product_idx])
				text_list = self.data_set.review_text[review_idx]
				text_length = len(text_list)

		batch_context_word_idxs = None
		length = len(word_idxs)
		if self.need_context:
			batch_context_word_idxs = []
			for length_idx in xrange(2 * self.window_size):
				batch_context_word_idxs.append(np.array([context_word_idxs[batch_idx][length_idx]
						for batch_idx in xrange(length)], dtype=np.int64))

		has_next = False if self.cur_review_i == self.review_size else True
		return user_idxs, product_idxs, query_word_idxs, review_idxs, word_idxs, batch_context_word_idxs, learning_rate, has_next

	def prepare_test_epoch(self):
		self.test_user_query_set = set()
		self.test_seq = []
		for review_idx in xrange(len(self.data_set.review_info)):
			user_idx = self.data_set.review_info[review_idx][0]
			product_idx = self.data_set.review_info[review_idx][1]	
			for query_idx in self.data_set.product_query_idx[product_idx]:
				if (user_idx, query_idx) not in self.test_user_query_set:
					self.test_user_query_set.add((user_idx, query_idx))
					self.test_seq.append((user_idx, product_idx, query_idx, review_idx))
		self.cur_uqr_i = 0

	def get_test_batch(self):
		user_idxs, product_idxs, review_idxs, word_idxs, context_word_idxs = [],[],[],[],[]
		query_word_idxs = []
		learning_rate = self.init_learning_rate * max(0.0001, 
									1.0 - self.finished_word_num / self.words_to_train)
		start_i = self.cur_uqr_i
		user_idx, product_idx, query_idx, review_idx = self.test_seq[self.cur_uqr_i]

		while len(user_idxs) < self.batch_size:
			text_list = self.data_set.review_text[review_idx]
			user_idxs.append(user_idx)
			product_idxs.append(product_idx)
			query_word_idxs.append(self.data_set.query_words[query_idx])
			review_idxs.append(review_idx)
			word_idxs.append(text_list[0])
			if self.need_context:
				i = 0
				start_index = i - self.window_size if i - self.window_size > 0 else 0
				context_word_list = text_list[start_index:i] + text_list[i+1:i+self.window_size+1]
				while len(context_word_list) < 2 * self.window_size:
					context_word_list += text_list[start_index:start_index+2*self.window_size-len(context_word_list)]
				context_word_idxs.append(context_word_list)
			
			#move to the next review
			self.cur_uqr_i += 1
			if self.cur_uqr_i == len(self.test_seq):
				break
			user_idx, product_idx, query_idx, review_idx = self.test_seq[self.cur_uqr_i]

		batch_context_word_idxs = None
		length = len(word_idxs)
		if self.need_context:
			batch_context_word_idxs = []
			for length_idx in xrange(2 * self.window_size):
				batch_context_word_idxs.append(np.array([context_word_idxs[batch_idx][length_idx]
						for batch_idx in xrange(length)], dtype=np.int64))

		has_next = False if self.cur_uqr_i == len(self.test_seq) else True
		return user_idxs, product_idxs, query_word_idxs, review_idxs, word_idxs, batch_context_word_idxs, learning_rate, has_next, self.test_seq[start_i:self.cur_uqr_i]
