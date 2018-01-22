#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import theano
import theano.tensor as T

import numpy as np
import pymc3 as pm 

from Utils.utils import VarDictParser

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
		self._get_weight_and_bias_shapes()


	def _get_weight_and_bias_shapes(self):
		self.weight_shapes = [[self.observed_params.shape[1], self.hidden_shape]]
		self.bias_shapes   = [[self.hidden_shape]]
		for index in range(1, self.num_layers - 1):
			self.weight_shapes.append([self.hidden_shape, self.hidden_shape])
			self.bias_shapes.append([self.hidden_shape])
		self.weight_shapes.append([self.hidden_shape, self.observed_params.shape[1]])
		self.bias_shapes.append([self.observed_params.shape[1]])



	def __get_weights(self, index, shape, scale = None):
		return pm.Normal('w%d' % index, self.weight_loc, self.weight_scale, shape = shape)


	def __get_biases(self, index, shape, scale = None):
		return pm.Normal('b%d' % index, self.weight_loc, self.weight_scale, shape = shape)


	def weight(self, index):
		return getattr(self, 'w%d' % index)

	def bias(self, index):
		return getattr(self, 'b%d' % index)


	def _get_rescalings(self):
		# compute rescaling factors for the different variables in the system
		# these rescaling factors will eventually substitute the 1.2 and 0.1 in the model below
		self.upper_rescalings = np.empty(self.total_size)
		self.lower_rescalings = np.empty(self.total_size)
		for var_p_index, var_p_name in enumerate(self.var_p_names):
			high = self.var_p_highs[var_p_index]
			low  = self.var_p_lows[var_p_index]
			if self.var_p_types[var_p_index] == 'float':
				self.upper_rescalings[var_p_index] = high + 0.1 * (high - low)
				self.lower_rescalings[var_p_index] = low - 0.1 * (high - low)
			else:
				raise NotImplementedError()
		# and don't forget to rescale the network input
		self.network_input = 2. * (self.observed_params - self.lower_rescalings) / (self.upper_rescalings - self.lower_rescalings) - 1.


	def _create_model(self):
		self._get_rescalings()

		with pm.Model() as self.model:

			# getting the location
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
					self._loc = pm.Deterministic('_loc', pm.math.sigmoid(pm.math.dot(getattr(self, 'fc%d' % (layer_index - 1)), self.weight(layer_index)) + self.bias(layer_index)) )


			# getting the precision / standard deviation / variance
			self.tau_rescaling = np.zeros((self.num_obs, self.observed_params.shape[1]))
			for obs_index in range(self.num_obs):
				self.tau_rescaling[obs_index] += self.var_p_ranges
			self.tau_rescaling = self.tau_rescaling**2

			self.tau   = pm.Gamma('tau', self.num_obs**2, 1., shape = (self.num_obs, self.observed_params.shape[1]))
			self.tau   = self.tau / self.tau_rescaling
			self.scale = pm.Deterministic('scale', 1. / pm.math.sqrt(self.tau))

			# learn the floats
			self.loc        = pm.Deterministic('loc', (self.upper_rescalings - self.lower_rescalings) * self._loc + self.lower_rescalings)
			self.out_floats = pm.Normal('out_floats', self.loc[:, self._floats], tau = self.tau[:, self._floats], observed = self.observed_params[:, self._floats])




	def _sample(self, num_epochs = None, num_draws = None):
		if not num_epochs: num_epochs = self.num_epochs
		if not num_draws:  num_draws  = self.num_draws

		with self.model:
			self.trace = pm.sample(draws = num_draws)

#			approx     = pm.fit(n = num_epochs, obj_optimizer = pm.adam(learning_rate = self.learning_rate))
#			self.trace = approx.sample(draws = num_draws)

