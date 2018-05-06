#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import time
import copy
import numpy as np 

from Utils.utils import VarDictParser

import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},
                  reload_support=True)
from BayesianNeuralNetwork.dist_evaluations import DistEvaluator

#========================================================================

class BayesianNeuralNetwork(VarDictParser):

	MODEL_DETAILS = {'burnin': 10**2, 'thinning': 20, 'num_epochs': 5 * 10**4, 'num_draws': 10**4, 'learning_rate': 0.1,
					 'num_layers': 3, 'hidden_shape': 6,
					 'weight_loc': 0., 'weight_scale': 1., 'bias_loc': 0., 'bias_scale': 1.}



	def __init__(self, var_dicts, observed_params, observed_losses, batch_size, backend = 'edward', model_details = None):

		VarDictParser.__init__(self, var_dicts)

		self.observed_params = observed_params
		self.observed_losses = observed_losses
		self.batch_size      = batch_size
		self.backend         = backend

		self.model_details = self.MODEL_DETAILS
		if model_details:
			for key, value in model_details.items():
				self.model_details[key] = value

		# get the volume of the domain
		self.volume = np.prod(self.var_p_ranges)

		if backend == 'pymc3':
			from BayesianNeuralNetwork.pymc3_interface import Pymc3Network
			self.network = Pymc3Network(self.var_dicts, observed_params, observed_losses, batch_size, self.model_details)
		elif backend == 'edward':
			from BayesianNeuralNetwork.edward_interface import EdwardNetwork
			self.network = EdwardNetwork(self.var_dicts, observed_params, observed_losses, batch_size, self.model_details)
		else:
			raise NotImplementedError()

		if self.batch_size == 1:
			self.lambda_values = np.array([0.])
		else:
			self.lambda_values = np.linspace(-0.25, 0.25, self.batch_size)
			self.lambda_values = self.lambda_values[::-1]
		self.lambda_values *= 1. / self.volume
		self.sqrt2pi = np.sqrt(2 * np.pi)




	def create_model(self):
		self.network._create_model()


	def sample(self, num_epochs = None, num_draws = None):
		self.network._sample(num_epochs, num_draws)


	def build_penalties(self):

		trace_mus_float = self.network.trace['loc'][self.model_details['burnin']::self.model_details['thinning']].copy()
		trace_sds_float = self.network.trace['scale'][self.model_details['burnin']::self.model_details['thinning']].copy()
		trace_mus_int   = self.network.trace['loc'][self.model_details['burnin']::self.model_details['thinning']].copy()
		trace_sds_int   = self.network.trace['int_scale'][self.model_details['burnin']::self.model_details['thinning']].copy()
		if hasattr(self.network, 'num_cats'):
			trace_cat_probs = [self.network.trace['dirich_%d' % counter][self.model_details['burnin']::self.model_details['thinning']] for counter in range(self.network.num_cats)].copy()
		else:
			trace_cat_probs = []

		self.network.trace = None

		# we need to contract the traces
		num_samples = trace_mus_float.shape[0]
		num_obs     = trace_mus_float.shape[1]

		mus_float = np.zeros((num_samples, num_obs, self.total_size))
		sds_float = np.zeros((num_samples, num_obs, self.total_size))
		mus_int   = np.zeros((num_samples, num_obs, self.total_size))
		sds_int   = np.zeros((num_samples, num_obs, self.total_size))
		cat_probs = np.array(trace_cat_probs)

		current_index = 0
		for var_index, var_p_type in enumerate(self.var_p_types):
			mus_float[:, :, var_index] = trace_mus_float[:, :, current_index]
			sds_float[:, :, var_index] = trace_sds_float[:, :, current_index]
			mus_int[:, :, var_index]   = trace_mus_int[:, :, current_index]
			sds_int[:, :, var_index]   = trace_sds_int[:, :, current_index]
			if var_p_type == 'categorical':
				current_index += len(self.var_p_options[var_index])
			else:
				current_index += 1

		if len(cat_probs) == 0:
			cat_probs = np.zeros((2, 2, 2, 2))

		self.dist_evaluator = DistEvaluator(mus_float, sds_float, mus_int, sds_int, cat_probs, self.observed_losses, 
										    self.var_p_type_indicators, self.var_p_periodic, self.var_p_ranges)


		def penalty_contributions(x):
#			probs_x = self.dist_evaluator.probs(x)
#			return np.dot(self.observed_losses, probs_x) / float(num_samples), np.mean(probs_x) + 1.
#			return np.dot(self.observed_losses, probs_x), np.sum(probs_x) + 1.
#			start_time = time.time()
			num, den = self.dist_evaluator.get_penalty(x)
			den += 1 / self.volume
#			print('TOOK', time.time() - start_time)
			return num, den

		self.penalty_contributions = penalty_contributions

#		print('TEST', penalty_contributions(np.array([0., 40.])))
#		quit()


