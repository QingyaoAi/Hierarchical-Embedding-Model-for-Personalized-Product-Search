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


def get_product_scores(model, query_word_idx, product_idxs = None, scope = None):
	with variable_scope.variable_scope(scope or "LSE_graph"):
		# get query vector
		query_vec, word_vecs = get_fs_from_words(model, query_word_idx, True)
		# match with product
		product_vec = None
		product_bias = None
		if product_idxs != None:										
			product_vec = tf.nn.embedding_lookup(model.product_emb, product_idxs)									
			product_bias = tf.nn.embedding_lookup(model.product_bias, product_idxs)
		else:										
			product_vec = model.product_emb
			product_bias = model.product_bias									
												
		print('Similarity Function : ' + model.similarity_func)										
												
		if model.similarity_func == 'product':										
			return tf.matmul(query_vec, product_vec, transpose_b=True)
		elif model.similarity_func == 'bias_product':
			return tf.matmul(query_vec, product_vec, transpose_b=True) + product_bias								
		else:										
			query_norm = tf.sqrt(tf.reduce_sum(tf.square(query_vec), 1, keep_dims=True))									
			product_norm = tf.sqrt(tf.reduce_sum(tf.square(product_vec), 1, keep_dims=True))									
			return tf.matmul(query_vec/query_norm, product_vec/product_norm, transpose_b=True)


def build_embedding_graph_and_loss(model, scope = None):	
	with variable_scope.variable_scope(scope or "LSE_graph"):										
		# Word embeddings.									
		init_width = 0.5 / model.embed_size									
		model.word_emb = tf.Variable( tf.random_uniform(									
							[model.vocab_size, model.embed_size], -init_width, init_width),				
							name="word_emb")
		model.word_emb = tf.concat(axis=0, values=[model.word_emb, tf.zeros([1, model.embed_size])])
		model.word_bias = tf.Variable(tf.zeros([model.vocab_size]), name="word_b")
		model.word_bias = tf.concat(axis=0, values=[model.word_bias, tf.zeros([1])])									
											
		# user/product embeddings.									
		model.user_emb =	tf.Variable( tf.zeros([model.user_size, model.embed_size]),								
							name="user_emb")				
		model.user_bias =	tf.Variable( tf.zeros([model.user_size]), name="user_b")								
		model.product_emb =	tf.Variable( tf.zeros([model.product_size, model.embed_size]),								
							name="product_emb")				
		model.product_bias =	tf.Variable( tf.zeros([model.product_size]), name="product_b")															
																		
		#model.context_emb = tf.Variable( tf.zeros([model.vocab_size, model.embed_size]),								
		#						name="context_emb")			
		#model.context_bias = tf.Variable(tf.zeros([model.vocab_size]), name="context_b")								
		return LSE_nce_loss(model, model.user_idxs, model.product_idxs, model.word_idxs, 
						model.context_word_idxs)	
											
											
											
def LSE_nce_loss(model, user_idxs, product_idxs, word_idxs, context_word_idxs):				
	batch_size = array_ops.shape(word_idxs)[0]#get batch_size										
	loss = None											

	# get f(s)
	word_idx_list = tf.stack([word_idxs] + context_word_idxs, 1)
	f_s, word_vecs = get_fs_from_words(model, word_idx_list, None)
	
	# Negative sampling
	loss, true_w, sample_w = LSE_single_nce_loss(model, f_s, product_idxs, model.product_emb,
					model.product_bias, model.product_size, model.product_distribute)

	# L2 regularization
	if model.L2_lambda > 0:
		loss += model.L2_lambda * (tf.nn.l2_loss(true_w) + tf.nn.l2_loss(sample_w) +
								tf.nn.l2_loss(model.f_W) + tf.nn.l2_loss(word_vecs))

	return loss / math_ops.cast(batch_size, dtypes.float32)										

def get_fs_from_words(model, word_idxs, reuse, scope=None):
	with variable_scope.variable_scope(scope or 'f_s_abstraction',
										 reuse=reuse):
		# get mean word vectors
		word_vecs = tf.nn.embedding_lookup(model.word_emb, word_idxs)
		mean_word_vec = tf.reduce_mean(word_vecs, 1)
		# get f(s)
		model.f_W = variable_scope.get_variable("f_W", [model.embed_size, model.embed_size])
		model.f_b = variable_scope.get_variable("f_b", [model.embed_size])
		f_s = tf.tanh(tf.nn.bias_add(tf.matmul(mean_word_vec, model.f_W), model.f_b))
		return f_s, word_vecs
						
def LSE_single_nce_loss(model, example_vec, label_idxs, label_emb, 											
					label_bias, label_size, label_distribution):						
	batch_size = array_ops.shape(label_idxs)[0]#get batch_size										
	# Nodes to compute the nce loss w/ candidate sampling.										
	labels_matrix = tf.reshape(tf.cast(label_idxs,dtype=tf.int64),[batch_size, 1])										
											
	# Negative sampling.										
	sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(										
			true_classes=labels_matrix,								
			num_true=1,								
			num_sampled=model.negative_sample,								
			unique=False,								
			range_max=label_size,								
			distortion=0.75,								
			unigrams=label_distribution))								
											
	#get label embeddings and bias [batch_size, embed_size], [batch_size, 1]										
	true_w = tf.nn.embedding_lookup(label_emb, label_idxs)										
	true_b = tf.nn.embedding_lookup(label_bias, label_idxs)										
											
	#get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]										
	sampled_w = tf.nn.embedding_lookup(label_emb, sampled_ids)										
	sampled_b = tf.nn.embedding_lookup(label_bias, sampled_ids)										
											
	# True logits: [batch_size, 1]										
	true_logits = tf.reduce_sum(tf.multiply(example_vec, true_w), 1) + true_b										
											
	# Sampled logits: [batch_size, num_sampled]										
	# We replicate sampled noise lables for all examples in the batch										
	# using the matmul.										
	sampled_b_vec = tf.reshape(sampled_b, [model.negative_sample])										
	sampled_logits = tf.matmul(example_vec, sampled_w, transpose_b=True) + sampled_b_vec										
											
	return nce_loss(model, true_logits, sampled_logits), true_w, sampled_w										
	#return model.nce_loss(true_logits, true_logits)										
											
											
def nce_loss(model, true_logits, sampled_logits):											
	"Build the graph for the NCE loss."										
											
	# cross-entropy(logits, labels)										
	true_xent = tf.nn.sigmoid_cross_entropy_with_logits(										
			logits=true_logits, labels=tf.ones_like(true_logits))								
	sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(										
			logits=sampled_logits, labels=tf.zeros_like(sampled_logits))								
											
	# NCE-loss is the sum of the true and noise (sampled words)										
	# contributions, averaged over the batch.										
	nce_loss_tensor = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent)) 										
	return nce_loss_tensor										



	