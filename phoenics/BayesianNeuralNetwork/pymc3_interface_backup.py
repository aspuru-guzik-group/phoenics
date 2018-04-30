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

		self._get_weights_and_bias_shapes()
		self._process_network_inputs()


	def __get_weights(self, index, shape, scale = None):
		return pm.Normal('w%d' % index, self.weight_loc, self.weight_scale, shape = shape)

	def __get_biases(self, index, shape, scale = None):
		return pm.Normal('b%d' % index, self.weight_loc, self.weight_scale, shape = shape)

	def weight(self, index):
		return getattr(self, 'w%d' % index)

	def bias(self, index):
		return getattr(self, 'b%d' % index)

	def _get_weight_and_bias_shapes(self):
		self.weight_shapes = [[self.observed_params.shape[1], self.hidden_shape]]
		self.bias_shapes   = [[self.hidden_shape]]
		for index in range(1, self.num_layers - 1):
			self.weight_shapes.append([self.hidden_shape, self.hidden_shape])
			self.bias_shapes.append([self.hidden_shape])
		self.weight_shapes.append([self.hidden_shape, self.observed_params.shape[1]])
		self.bias_shapes.append([self.observed_params.shape[1]])

	def _process_network_inputs(self):
		print('OBS', self.observed_params)
		quit()




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
			elif self.var_p_types[var_p_index] == 'integer':
				self.upper_rescalings[var_p_index] = high# + np.ceil(0.1 * (high - low))
				self.lower_rescalings[var_p_index] = low# - np.ceil(0.1 * (high - low))
		# and don't forget to rescale the network input

		self.network_input = 2. * (self.observed_params - self.lower_rescalings) / (self.upper_rescalings - self.lower_rescalings) - 1.
		print('OBSERVED_PARAMS', self.observed_params)
		print('NETWORK_INPUT', self.network_input)
		quit()

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
			elif self.var_p_types[var_p_index] == 'integer':
				self.upper_rescalings[var_p_index] = high# + np.ceil(0.1 * (high - low))
				self.lower_rescalings[var_p_index] = low# - np.ceil(0.1 * (high - low))
		# and don't forget to rescale the network input

		self.network_input = 2. * (self.observed_params - self.lower_rescalings) / (self.upper_rescalings - self.lower_rescalings) - 1.
		print('OBSERVED_PARAMS', self.observed_params)
		print('NETWORK_INPUT', self.network_input)
		quit()


	def _get_categorical_observations(self):
		# note that we might have multiple categorical variables with a different number of categories
		cat_obs     = []
		for var_p_index, var_p_type in enumerate(self.var_p_types):
			if var_p_type == 'categorical':
				new_cat_obs = np.zeros((self.num_obs, len(self.var_p_options[var_p_index])))
				for obs_index, obs in enumerate(self.observed_params[:, var_p_index]):
					new_cat_obs[obs_index, int(obs)] += 1
				cat_obs.append(new_cat_obs.copy())				

		self.cat_obs = cat_obs


	def _create_model(self):
		self._get_rescalings()
		self._get_categorical_observations()

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
#					self.loc = pm.Deterministic('loc', (self.upper_rescalings - self.lower_rescalings) * pm.math.sigmoid(pm.math.dot(getattr(self, 'fc%d' % (layer_index - 1)), self.weight(layer_index)) + self.bias(layer_index)) + self.lower_rescalings)
					self._loc = pm.Deterministic('_loc', pm.math.sigmoid(pm.math.dot(getattr(self, 'fc%d' % (layer_index - 1)), self.weight(layer_index)) + self.bias(layer_index)) )


			# getting the precision / standard deviation / variance
			self.tau_rescaling = np.zeros((self.num_obs, self.observed_params.shape[1]))
			for obs_index in range(self.num_obs):
				self.tau_rescaling[obs_index] += self.var_p_ranges
			self.tau_rescaling = self.tau_rescaling**2

			self.tau   = pm.Gamma('tau', self.num_obs**2, 1., shape = (self.num_obs, self.observed_params.shape[1]))
#			self.tau   = pm.Gamma('tau', self.num_obs**1.5, 1., shape = (self.num_obs, self.observed_params.shape[1]))
			self.tau   = self.tau / self.tau_rescaling
#			self.sd    = pm.Deterministic('sd', 0.05 + 1. / pm.math.sqrt(self.tau))
			self.scale = pm.Deterministic('scale', 1. / pm.math.sqrt(self.tau))


			# learn the floats
			self.loc        = pm.Deterministic('loc', (self.upper_rescalings - self.lower_rescalings) * self._loc + self.lower_rescalings)
			self.out_floats = pm.Normal('out_floats', self.loc[:, self._floats], tau = self.tau[:, self._floats], observed = self.observed_params[:, self._floats])


			# learn the integers
			self.out_ints = DiscreteLaplace('out_ints', loc = self.loc[:, self._ints], scale = self.scale[:, self._ints], observed = self.observed_params[:, self._ints])


			# learn the categories
#			alpha = self.loc * (self.loc * (1 - self.loc) * self.tau - 1)
#			beta  = (1 - self.loc) * (self.loc * (1 - self.loc) * self.tau - 1)
#			self.alpha = pm.Deterministic('alpha', alpha)
#			self.beta  = pm.Deterministic('beta', beta)
#			self.p     = pm.Beta('p', alpha = self.alpha, beta = self.beta)
#			print('ALL_PARAMS', self.observed_params)
#			print('OBSERV', self.observed_params[:, self._cats])

			self.probs    = pm.Deterministic('a_dirich', self._loc * self.tau)

			for cat_obs_index in range(len(self.cat_obs)):
#				print(self._cats)
#				print(self._cats[cat_obs_index])
#				indices = np.array([self._cats[cat_obs_index]])
#				print('INDICES', indices)
#				print(self.probs[:, self._cats])
#				cat_specific_indices = 

				out_cats = pm.Dirichlet('out_cats_%d' % cat_obs_index, a = self.probs[:, cat_specific_indices], observed = self.cat_obs[cat_obs_index])
				setattr(self, 'out_cats_%d' % cat_obs_index, out_cats)

#			self.out_cats = pm.Dirichlet('out_cats', a = self.probs[:, self._cats], observed = self.observed_params[:, self._cats])
#			self.out_cats = pm.Normal('p', loc = self.loc[:, self._cats], tau = self.tau[:, self._cats], observed = self.observed_params[:, self._cats]) 					# perhaps constrain this to only positive numbers!
#			self.out_cats = pm.Categorical('out_cats', p = self.p, observed = self.observed_params[:, self._cats])








	def _create_model_old(self):
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
					self.loc = pm.Deterministic('loc', (self.upper_rescalings - self.lower_rescalings) * pm.math.sigmoid(pm.math.dot(getattr(self, 'fc%d' % (layer_index - 1)), self.weight(layer_index)) + self.bias(layer_index)) + self.lower_rescalings)



			# getting the standard deviation (or rather precision)
			self.tau_rescaling = np.zeros((self.num_obs, self.observed_params.shape[1]))
			for obs_index in range(self.num_obs):
				self.tau_rescaling[obs_index] += self.domain_ranges
			self.tau_rescaling = self.tau_rescaling**2
			self.tau = pm.Gamma('tau', self.num_obs**2, 1., shape = (self.num_obs, self.observed_params.shape[1]))
			self.tau = self.tau / self.tau_rescaling
#			self.sd  = pm.Deterministic('sd', 0.05 + 1. / pm.math.sqrt(self.tau))
			self.scale  = pm.Deterministic('scale', 1. / pm.math.sqrt(self.tau))



			print(self.observed_params.shape)
			print(self._floats)
			print(self._integers)
			quit()

			# now that we got all locations and scales we can start getting the distributions


			# floats are easy, as we can take loc and scale as they are
			self.out = pm.Normal('out', self.loc, tau = self.tau, observed = self.observed_params)

			# integers are a bit more tricky and require the following transformation for the beta binomial
			alpha = ((n - mu) / sigma**2 - 1) / (n / mu - (n - mu) / sigma**2)
			beta  = (n / mu - 1) * alpha
			self.alpha = pm.Deterministic('alpha', alpha)
			self.beta  = pm.Deterministic('beta', beta)




	def _sample(self, num_epochs = None, num_draws = None):
		if not num_epochs: num_epochs = self.num_epochs
		if not num_draws:  num_draws  = self.num_draws

		with self.model:
			approx     = pm.fit(n = num_epochs, obj_optimizer = pm.adam(learning_rate = self.learning_rate))
			self.trace = approx.sample(draws = num_draws)

