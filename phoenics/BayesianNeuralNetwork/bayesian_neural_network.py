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

	MODEL_DETAILS = {'burnin': 1000, 'thinning': 100, 'num_epochs': 10**5, 'num_draws': 10**4, 'learning_rate': 0.01,
					 'num_layers': 3, 'hidden_shape': 6,
					 'weight_loc': 0., 'weight_scale': 1., 'bias_loc': 0., 'bias_scale': 1.}



	def __init__(self, var_dicts, observed_params, observed_losses, batch_size, backend = 'pymc3', model_details = None):

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
		else:
			raise NotImplementedError()

		if self.batch_size == 1:
			self.lambda_values = [0.]
		else:
			self.lambda_values = np.linspace(-1., 1., self.batch_size)
			self.lambda_values = self.lambda_values[::-1]
		self.lambda_values *= 1 / self.volume



	def create_model(self):
		self.network._create_model()


	def sample(self, num_epochs = None, num_draws = None):
		self.network._sample(num_epochs, num_draws)


	def build_penalties(self):
		mus = self.network.trace['loc'][self.model_details['burnin']::self.model_details['thinning']]
		sds = self.network.trace['scale'][self.model_details['burnin']::self.model_details['thinning']]

		num_samples = len(mus)
		self.dist_evaluator = DistEvaluator(mus, sds, self.observed_losses, self.var_p_type_indicators)

		def penalty_contributions(x):
			num, den = self.dist_evaluator.get_penalty(x)
			den += 1 / self.volume
			return num, den

		self.penalty_contributions = penalty_contributions
