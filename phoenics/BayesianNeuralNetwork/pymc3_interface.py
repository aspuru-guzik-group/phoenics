#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import theano
import theano.tensor as T

import numpy as np
import pymc3 as pm 

from Utils.utils import VarDictParser
from BayesianNeuralNetwork.distributions import DiscreteLaplace

#========================================================================

class Pymc3Network(VarDictParser):

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
		return pm.Normal('w%d' % index, self.weight_loc, self.weight_scale, shape = shape)

	def __get_biases(self, index, shape, scale = None):
		return pm.Normal('b%d' % index, self.weight_loc, self.weight_scale, shape = shape)

	def weight(self, index):
		return getattr(self, 'w%d' % index)

	def bias(self, index):
		return getattr(self, 'b%d' % index)

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

		with pm.Model() as self.model:

			# getting the location primers
			for layer_index in range(self.num_layers):
				setattr(self, 'w%d' % layer_index, self.__get_weights(layer_index, self.weight_shapes[layer_index]))
				setattr(self, 'b%d' % layer_index, self.__get_biases(layer_index, self.bias_shapes[layer_index]))

				if layer_index == 0:
					fc = pm.Deterministic('fc%d' % layer_index, pm.math.tanh(pm.math.dot(self.network_input, self.weight(layer_index)) + self.bias(layer_index)))
					setattr(self, 'fc%d' % layer_index, fc)
				elif 0 < layer_index < self.num_layers - 1:
					fc = pm.Deterministic('fc%d' % layer_index, pm.math.tanh(pm.math.dot(getattr(self, 'fc%d' % (layer_index - 1)), self.weight(layer_index)) + self.bias(layer_index)))
					setattr(self, 'fc%d' % layer_index, fc)
				else:
					self._loc = pm.Deterministic('bnn_out', pm.math.sigmoid(pm.math.dot(getattr(self, 'fc%d' % (layer_index - 1)), self.weight(layer_index)) + self.bias(layer_index)) )	


			# getting the precision / standard deviation / variance
			self.tau_rescaling = np.zeros((self.num_obs, self.network_input.shape[1]))
			for obs_index in range(self.num_obs):
				self.tau_rescaling[obs_index] += self.var_e_ranges
			self.tau_rescaling = self.tau_rescaling**2

			tau        = pm.Gamma('tau', self.num_obs**2, 1., shape = (self.num_obs, self.network_input.shape[1]))
			self.tau   = tau / self.tau_rescaling
			self.scale = pm.Deterministic('scale', 1. / pm.math.sqrt(self.tau))


			# learn the floats
			self.loc        = pm.Deterministic('loc', (self.upper_rescalings - self.lower_rescalings) * self._loc + self.lower_rescalings)
			self.out_floats = pm.Normal('out_floats', self.loc[:, self.floats], tau = self.tau[:, self.floats], observed = self.network_output[:, self._floats])


			# learn the integers
			self.int_scale = pm.Deterministic('int_scale', 1. * self.scale)
			self.out_ints  = DiscreteLaplace('out_ints', loc = self.loc[:, self.ints], scale = self.int_scale[:, self.ints], observed = self.network_output[:, self._ints])


			# learn the categories
			dist_counter, cat_var_index = 0, 0
			
			self.alpha = pm.Deterministic('alpha', (self.loc + 1.) * self.scale)
			self.num_cats = 0
			for var_e_index, var_e_type in enumerate(self.var_e_types):
				if var_e_type == 'categorical' and self.var_e_begin[var_e_index] == var_e_index:
					begin, end  = self.var_e_begin[var_e_index], self.var_e_end[var_e_index]
					var_e_name  = self.var_e_names[var_e_index]
					param_index = np.argwhere(self.var_p_names == var_e_name)[0, 0]
					self.param_index = param_index

					out_dirichlet = pm.Dirichlet('dirich_%d' % dist_counter, a = self.alpha[:, begin : end], shape = (self.num_obs, int(end - begin)) )
					out_cats      = pm.Categorical('out_cats_%d' % dist_counter, p = out_dirichlet, observed = self.network_output[:, param_index])
					self.num_cats += 1
					dist_counter += 1


	def _sample(self, num_epochs = None, num_draws = None):
		if not num_epochs: num_epochs = self.num_epochs
		if not num_draws:  num_draws  = self.num_draws

		with self.model:
			
#			approx     = pm.fit(method = 'svgd', n = 1000, obj_optimizer = pm.adam(learning_rate = self.learning_rate))
#			self.trace = approx.sample(draws = num_draws)

			approx     = pm.fit(n = num_epochs, obj_optimizer = pm.adam(learning_rate = self.learning_rate))
			self.trace = approx.sample(draws = num_draws)

