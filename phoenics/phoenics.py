#!/usr/bin/env python

__author__ = 'Florian Hase'

#========================================================================

import numpy as np 
import os, sys, copy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .Acquisitions.sampler                          import AcquisitionFunctionSampler
from .BayesianNeuralNetwork.bayesian_neural_network import BayesianNeuralNetwork
from .ObservationParser.observation_parser          import ObservationParser
from .RandomNumberGenerator.random_number_generator import RandomNumberGenerator
from .SampleSelector.sample_selector                import SampleSelector
from .Utils.utils import ParserJSON, VarDictParser

#========================================================================

class Phoenics(VarDictParser):


	def __init__(self, config_file = None):

		self._parse_config_file(config_file)

		self.observation_parser      = ObservationParser(self.param_dict['variables'])
		self.acq_func_sampler        = AcquisitionFunctionSampler(self.var_infos, self.param_dict['variables'])
		self.random_number_generator = RandomNumberGenerator()
		self.sample_selector         = SampleSelector(self.param_dict['variables'])

		self.obs_params, self.obs_losses = [], []


	def _parse_config_file(self, file_name):
		# first parse the json file
		self.json_parser = ParserJSON(file_name = file_name)
		self.json_parser.parse()
		self.param_dict = self.json_parser.param_dict
		VarDictParser.__init__(self, self.param_dict['variables'])


	def _generate_uniform(self, num_samples):
		samples = []
		for var_index, var_name in enumerate(self.var_names):
			var_dict = self.param_dict['variables'][var_index][var_name]
			sampled_values = self.random_number_generator.generate(var_dict, size = (self.var_sizes[var_index], num_samples))
			samples.extend(sampled_values)
		self.proposed_samples = np.array(samples).transpose()


	def _compute_characteristic_distances(self):
		self.characteristic_distances = np.empty(self.total_size)
		start_index = 0
		for var_index, var_name in enumerate(self.var_names):
			var_dict   = self.param_dict['variables'][var_index][var_name]
			var_range  = var_dict['high'] - var_dict['low']
			var_ranges = np.zeros(var_dict['size']) + var_range
			self.characteristic_distances[start_index : start_index + var_dict['size']] = var_ranges 
			start_index += var_dict['size']
		self.characteristic_distances /= float(len(self.obs_params))


	def _generate_sampled(self, num_samples, observ_dict):
		# clean observations
		obs_params, obs_losses = self.observation_parser.parse(observ_dict)
		lowest_loss   = np.amin(obs_losses)
		lowest_params = obs_params[np.argmin(obs_losses)]

		self.obs_params, self.obs_losses = self.observation_parser._raw_obs_params, self.observation_parser._raw_obs_losses
		self._compute_characteristic_distances()

		# create and sample the model
		print('# running density estimation')
		self.network = BayesianNeuralNetwork(self.var_dicts, obs_params, obs_losses, self.param_dict['general']['batch_size'])
		self.network.create_model()
		self.network.sample()
		self.network.build_penalties()

		# sample the acquisition function
		print('# proposing new samples')
		self.proposed_samples = self.acq_func_sampler.sample(lowest_params, self.network.penalty_contributions, self.network.lambda_values, parallel = self.param_dict['general']['parallel_evaluations'])



	def choose(self, num_samples = None, observations = None, as_array = False):

		if not num_samples:
			num_samples = self.param_dict['general']['num_batches']

		if observations:
			# generating samples
			self._generate_sampled(num_samples, observations)		
			# get the most informative
			print('# selecting informative samples')
			self.imp_samples = self.sample_selector.select(num_samples, self.proposed_samples, self.network.penalty_contributions, self.network.lambda_values, self.characteristic_distances)
		else:
			# generating samples
			self._generate_uniform(num_samples)
			# cleaning samples - not required for uniform samples
			self.imp_samples = self.proposed_samples

		# convert sampled parameters to list of dicts
		self.gen_samples = []
		for sample in self.imp_samples:
			sample_dict = {}
			lower, upper = 0, self.var_sizes[0]
			for var_index, var_name in enumerate(self.var_names):
				sample_dict[var_name] = {'samples': sample[lower:upper]}
				if var_index == len(self.var_names) - 1:
					break
				lower = upper
				upper += self.var_sizes[var_index + 1]
			self.gen_samples.append(copy.deepcopy(sample_dict))

		if as_array:
			return self.imp_samples
		else:
			return self.gen_samples

#========================================================================

if __name__ == '__main__':

	phoenics = Phoenics('config_default.txt')
