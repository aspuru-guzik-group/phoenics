#!/usr/bin/env python 

import numpy as np 
import tensorflow as tf 
import edward as ed 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#======================================================================================

class ManagerDummy(object):

	def __init__(self):
		self.mean_features = None 
		self.std_features  = None 
		self.mean_targets  = None 
		self.std_targets   = None

		self.train_features = None
		self.train_targets  = None

#======================================================================================

class BayesianNeuralNetwork(object):

	BATCH_SIZE    = 100
	IN_SHAPE      = 6
	OUT_SHAPE     = 2
	HIDDEN_SHAPE  = 192
	NUM_LAYERS    = 3
	ACT_FUNC      = 'leaky_relu'
	LEARNING_RATE = 10**(-2.5) 


	def generator(self, arrays, batch_size):
		starts = [0] * len(arrays)
		while True:
			batches = []
			for i, array in enumerate(arrays):
				start = starts[i]
				stop  = start + batch_size
				diff  = stop - array.shape[0]
				if diff <= 0:
					batch = array[start: stop]
					starts[i] += batch_size
				else:
					batch = np.concatenate((array[start:], array[:diff]))
					starts[i] = diff
				batches.append(batch)
			yield batches



	def __init__(self):
		self.manager = ManagerDummy()
		self.train_features = None
		self.train_targets  = None
		self.valid_features = None 
		self.valid_targets  = None



	def _prepare_network(self):

		# get batched training set
		self.batched_train_data = self.generator([self.train_features, self.train_targets], self.BATCH_SIZE)

		# get weights and biases
		self.weight_shapes = [(self.IN_SHAPE, self.HIDDEN_SHAPE)]
		self.bias_shapes   = [(self.HIDDEN_SHAPE)]
		for layer_index in range(1, self.NUM_LAYERS - 1):
			self.weight_shapes.append((self.HIDDEN_SHAPE, self.HIDDEN_SHAPE))
			self.bias_shapes.append((self.HIDDEN_SHAPE))
		self.weight_shapes.append((self.HIDDEN_SHAPE, self.OUT_SHAPE))
		self.bias_shapes.append((self.OUT_SHAPE))


		self.tf_activation_functions = {'softsign': tf.nn.softsign, 'softplus': tf.nn.softplus, 'relu': tf.nn.relu6, 'tanh': tf.nn.tanh, 'sigmoid': tf.nn.sigmoid,
								        'leaky_relu': lambda x: tf.nn.leaky_relu(x, 0.2)}
		self.act_tf = [tf_activation_functions[self.ACT_FUNC] for i in range(self.NUM_LAYERS)]


	def construct_networks(self):
		print('... constructing network')
		self._prepare_network()
		self.x = tf.placeholder(tf.float32, shape = (None, self.IN_SHAPE))
		self.y = tf.placeholder(tf.float32, shape = (None, self.OUT_SHAPE))

		# initialize weights and biases
		p_dists = []
		for layer_index in range(self.NUM_LAYERS):
			weight = ed.models.Laplace(loc = tf.zeros(self.weight_shapes[layer_index]), scale = tf.nn.softplus(tf.zeros(self.weight_shapes[layer_index])))
			bias   = ed.models.Laplace(loc = tf.zeros(self.bias_shapes[layer_index]),   scale = tf.nn.softplus(tf.zeros(self.bias_shapes[layer_index])))
			setattr(self, 'w_%d' % layer_index, weight)
			setattr(self, 'b_%d' % layer_index, bias)
			p_dists.extend([weight, bias])

		# construct network graph
		self.fc_0 = self.act_tf[0](tf.matmul(self.x, self.w_0) + self.b_0)
		for layer_index in range(1, self.NUM_LAYERS):
			fc = self.act_tf[layer_index](tf.matmul(getattr(self, 'fc_%d' % (layer_index - 1)), getattr(self, 'w_%d' % layer_index)) + getattr(self, 'b_%d' % layer_index))
			setattr(self, 'fc_%d' % layer_index, fc)

		y = ed.models.Normal(loc = getattr(self, 'fc_%d' % layer_index), scale = 10**-2.5)

		q_dists = []
		for layer_index in range(self.NUM_LAYERS):
			q_weight = ed.models.Laplace(loc = tf.Variable(tf.random_normal(self.weight_shapes[layer_index])), scale = tf.nn.softplus(tf.Variable(tf.random_normal(self.weight_shapes[layer_index]))))
			q_bias   = ed.models.Laplace(loc = tf.Variable(tf.random_normal(self.bias_shapes[layer_index])),   scale = tf.nn.softplus(tf.Variable(tf.random_normal(self.bias_shapes[layer_index]))))
			setattr(self, 'q_w_%d' % layer_index, weight)
			setattr(self, 'q_b_%d' % layer_index, bias)
			q_dists.extend([q_weight, q_bias])

		var_dict = {}
		for index, element in enumerate(p_dists):
			var_dict{element : q_dists[index]}

		self.y_post = ed.copy(y, var_dict)

		self.inference = ed.KLqp(var_dict, data = (y: self.y))
		optimizer      = tf.train.AdamOptimizer(self.LEARNING_RATE)
		self.inference.initialize(optimizer = optimizer, n_iter = 10**6)
		tf.global_variables_initializer().run()


	def train(self, train_iters, epoch = -1):
		for i in range(train_iters):
			x_batch, y_batch = next(self.batched_train_data)
			self.inference.update(feed_dict = {self.x: x_batch, self.y: y_batch})	



	def predict(self, params, n_samples = 100):

		prediction = self.y_post.sample(n_samples).eval(feed_dict = {self.x: params})

		pred_raw      = prediction
		pred_raw_mean = np.mean(pred_raw, axis = 0)
		pred_raw_std  = np.std(pred_raw, axis = 0) 
