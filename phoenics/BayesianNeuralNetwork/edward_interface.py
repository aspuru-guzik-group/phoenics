#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import tensorflow as tf 
import edward as ed 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np 

from Utils.utils import VarDictParser 
from BayesianNeuralNetwork.distributions import DiscreteLaplace

#========================================================================

class EdwardNetwork(VarDictParser):



	def __init__(self, var_dicts, observed_params, observed_losses, batch_size, model_details):
		VarDictParser.__init__(self, var_dicts)

		self.observed_params = observed_params
		self.observed_losses = observed_losses
		self.num_obs         = len(self.observed_losses)
		self.batch_size      = batch_size
		self.model_details   = model_details

		for key, value in self.model_details.items():
			setattr(self, str(key), value)


		self._process_network_inputs()
		self._get_weights_and_bias_shapes()



	def __get_weights(self, index, shape, scale = None):
		return ed.models.Normal(loc = tf.zeros(shape) + self.weight_loc, scale = tf.zeros(shape) + self.weight_scale)

	def __get_biases(self, index, shape, scale = None):
		return ed.models.Normal(loc = tf.zeros(shape) + self.weight_loc, scale = tf.zeros(shape) + self.weight_scale)


	def weight(self, index):
		return getattr(self, 'w%d' % index)
	def q_weight(self, index):
		return getattr(self, 'q_w%d' % index)

	def bias(self, index):
		return getattr(self, 'b%d' % index)
	def q_bias(self, index):
		return getattr(self, 'q_b%d' % index)


	def _get_weights_and_bias_shapes(self):
		self.weight_shapes = [[self.network_input.shape[1], self.hidden_shape]]
		self.bias_shapes   = [[self.hidden_shape]]
		for index in range(1, self.num_layers - 1):
			self.weight_shapes.append([self.hidden_shape, self.hidden_shape])
			self.bias_shapes.append([self.hidden_shape])
		self.weight_shapes.append([self.hidden_shape, self.network_input.shape[1]])
		self.bias_shapes.append([self.network_input.shape[1]])



	def _process_network_inputs(self):
		self.network_input  = np.zeros((self.num_obs, self.complete_size)) #+ 10.**-4
		self.network_output = np.zeros((self.num_obs, self.total_size))
		for obs_index, obs in enumerate(self.observed_params):
			current_index  = 0
			for var_index, value in enumerate(obs):
				if self.var_p_types[var_index] == 'categorical':
					self.network_input[obs_index, int(current_index + value)] += 1. #- 2 * 10.**-4
					self.network_output[obs_index, var_index] = value
					current_index += len(self.var_p_options[var_index])
				else:
					self.network_input[obs_index, current_index]  = value
					self.network_output[obs_index, var_index] = value
					current_index += 1


		for att in ['floats', 'ints', 'cats']:
			setattr(self, att, np.array([False for i in range(self.complete_size)]))

		self.upper_rescalings = np.empty(self.complete_size)
		self.lower_rescalings = np.empty(self.complete_size)
		for var_e_index, var_e_name in enumerate(self.var_e_names):
			high = self.var_e_highs[var_e_index]
			low  = self.var_e_lows[var_e_index]
			if self.var_e_types[var_e_index] == 'float':
				self.upper_rescalings[var_e_index] = high + 0.1 * (high - low)
				self.lower_rescalings[var_e_index] = low - 0.1 * (high - low)
				self.floats[var_e_index] = True
			elif self.var_e_types[var_e_index] == 'integer':
				self.upper_rescalings[var_e_index] = high# + np.ceil(0.1 * (high - low))
				self.lower_rescalings[var_e_index] = low# - np.ceil(0.1 * (high - low))
				self.ints[var_e_index] = True
			elif self.var_e_types[var_e_index] == 'categorical':
				self.upper_rescalings[var_e_index] = 1.
				self.lower_rescalings[var_e_index] = 0.
				self.cats[var_e_index] = True


		self.network_input  = 2. * (self.network_input - self.lower_rescalings) / (self.upper_rescalings - self.lower_rescalings) - 1.



	def _create_model(self):


		self.x = tf.convert_to_tensor(self.network_input,  dtype = tf.float32)
		self.y = tf.convert_to_tensor(self.network_output, dtype = tf.float32)


		# getting the precision / standard deviation / variance
		self.tau_rescaling = np.zeros((self.num_obs, self.network_input.shape[1]))
		for obs_index in range(self.num_obs):
			self.tau_rescaling[obs_index] += self.var_e_ranges
		self.tau_rescaling = self.tau_rescaling**2


		# PRIOR
		for layer_index in range(self.num_layers):
			setattr(self, 'w%d' % layer_index, self.__get_weights(layer_index, self.weight_shapes[layer_index]))
			setattr(self, 'b%d' % layer_index, self.__get_biases(layer_index, self.bias_shapes[layer_index]))

			if layer_index == 0:
				fc = tf.nn.tanh(tf.matmul(self.x, self.weight(layer_index)) + self.bias(layer_index))
				setattr(self, 'fc%d' % layer_index, fc)
			elif 0 < layer_index < self.num_layers - 1:
				fc = tf.nn.tanh(tf.matmul(getattr(self, 'fc%d' % (layer_index - 1)), self.weight(layer_index)) + self.bias(layer_index))
				setattr(self, 'fc%d' % layer_index, fc)
			else:
				self._loc = tf.nn.sigmoid(tf.matmul(getattr(self, 'fc%d' % (layer_index - 1)), self.weight(layer_index)) + self.bias(layer_index))


		tau = ed.models.Gamma(tf.zeros((self.num_obs, self.network_input.shape[1])) + self.num_obs**2, tf.ones((self.num_obs, self.network_input.shape[1])))
		self.tau   = tau / self.tau_rescaling
		self.scale = ed.models.Deterministic(1. / tf.sqrt(self.tau))

		# learn the floats
		self.loc 		= ed.models.Deterministic((self.upper_rescalings - self.lower_rescalings) * self._loc + self.lower_rescalings)
		self.out_floats = ed.models.Normal(self.loc, self.scale)




		# POSTERIOR
		for layer_index in range(self.num_layers):
			setattr(self, 'q_w%d' % layer_index, ed.models.Normal(tf.Variable(tf.zeros(self.weight_shapes[layer_index])), tf.nn.softplus(tf.Variable(tf.zeros(self.weight_shapes[layer_index])))))
			setattr(self, 'q_b%d' % layer_index, ed.models.Normal(tf.Variable(tf.zeros(self.bias_shapes[layer_index])),   tf.nn.softplus(tf.Variable(tf.zeros(self.bias_shapes[layer_index])))))

			if layer_index == 0:
				q_fc = tf.nn.tanh(tf.matmul(self.x, self.q_weight(layer_index)) + self.q_bias(layer_index))
				setattr(self, 'q_fc%d' % layer_index, q_fc)
			elif 0 < layer_index < self.num_layers - 1:
				q_fc = tf.nn.tanh(tf.matmul(getattr(self, 'q_fc%d' % (layer_index - 1)), self.q_weight(layer_index)) + self.q_bias(layer_index))
				setattr(self, 'q_fc%d' % layer_index, q_fc)
			else:
				self.q_loc = tf.nn.sigmoid(tf.matmul(getattr(self, 'q_fc%d' % (layer_index - 1)), self.q_weight(layer_index)) + self.q_bias(layer_index))


		q_tau = ed.models.Gamma(tf.Variable(self.num_obs**2 + tf.zeros((self.num_obs, self.network_input.shape[1]))), tf.nn.softplus(tf.Variable(tf.ones((self.num_obs, self.network_input.shape[1])))))
		self.q_tau   = q_tau / self.tau_rescaling
		self.q_scale = ed.models.Deterministic(1. / tf.sqrt(self.q_tau))

		# learn the floats
		self.q_loc 		  = ed.models.Deterministic((self.upper_rescalings - self.lower_rescalings) * self.q_loc + self.lower_rescalings)
		self.q_out_floats = ed.models.Normal(self.q_loc, self.q_scale)




		# INFERENCE
		var_dict = {}
		for layer_index in range(self.num_layers):
			var_dict[getattr(self, 'w%d' % layer_index)] = getattr(self, 'q_w%d' % layer_index)
			var_dict[getattr(self, 'b%d' % layer_index)] = getattr(self, 'q_b%d' % layer_index)

		self.inference = ed.KLqp(var_dict, data = {self.out_floats: self.y})
		optimizer      = tf.train.AdamOptimizer(self.learning_rate)
		self.inference.initialize(optimizer = optimizer, n_iter = 5 * 10**3)
		tf.global_variables_initializer().run()


	def _sample(self, num_epochs = None, num_draws = None):
		print('... sampling')
		if not num_epochs: num_epochs = self.num_epochs
		if not num_draws:  num_draws  = self.num_draws

		import time 
		start = time.time()
		for i in range(self.inference.n_iter):
			info_dict = self.inference.update()
			self.inference.print_progress(info_dict)
		self.inference.finalize()
		print('... took ', time.time() - start, 's')

		self.trace = {}
		print('... getting posterior')
		self.trace['loc']       = self.q_loc.sample(10**4).eval()
		self.trace['scale']     = self.q_scale.sample(10**4).eval()
		self.trace['int_scale'] = self.q_scale.sample(10**4).eval()

#		print('OUTPUT', self.network_output)
#		print('')
#		print('LOC', np.mean(self.trace['loc'], axis = 0))
#		print('')
#		print(np.mean(self.trace['scale'], axis = 0))
