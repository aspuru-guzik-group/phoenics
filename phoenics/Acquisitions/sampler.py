#!/usr/bin/env python

__author__ = 'Florian Hase'

#========================================================================

import numpy as np 
from scipy.optimize import minimize
from multiprocessing import Process, Queue

from RandomNumberGenerator.random_number_generator import RandomNumberGenerator
from Utils.utils import VarDictParser

#========================================================================

class AcquisitionFunctionSampler(VarDictParser):

	def __init__(self, var_infos, var_dicts):
		VarDictParser.__init__(self, var_dicts)

		self.var_infos = var_infos
		for key, value in self.var_infos.items():
			setattr(self, str(key), value)
		self.total_size = np.sum(self.var_sizes)

		self.random_number_generator = RandomNumberGenerator()



	def _get_perturbed_samples(self, scale, current_best, num_samples):
		start_index = 0
		perturbed_samples = []
		for var_index, full_var_dict in enumerate(self.var_dicts):
			var_dict = full_var_dict[self.var_names[var_index]]

			# first we generate the perturbations
			var_dict['loc']   = (var_dict['high'] + var_dict['low']) / 2.
			var_dict['scale'] = (var_dict['high'] - var_dict['low']) * scale
			sampled_values = self.random_number_generator.generate(var_dict, size = (self.var_sizes[var_index], self.total_size * num_samples), kind = 'normal')
			
			# then we generate the samples
			relevant_current_best = current_best[start_index : start_index + var_dict['size']]
			sampled_values += relevant_current_best

			# then we make sure that we didn't leave the domain
			domain_sampled_values = []
			for sample in sampled_values:
				sample = np.abs(sample - var_dict['low']) + var_dict['low']
				sample = var_dict['high'] - np.abs(var_dict['high'] - sample)
				if np.all(sample > var_dict['low']) and np.all(sample < var_dict['high']):
					domain_sampled_values.append(sample)

			# and finally store the generated samples
			perturbed_samples.extend(domain_sampled_values)
			start_index += var_dict['size']
		return np.array(perturbed_samples).transpose()




	def _get_random_proposals(self, current_best, num_samples):
		# get uniform samples first
		uniform_samples = []
		for var_index, full_var_dict in enumerate(self.var_dicts):
			var_dict = full_var_dict[self.var_names[var_index]]
			sampled_values = self.random_number_generator.generate(var_dict, size = (self.var_sizes[var_index], self.total_size * num_samples))
			uniform_samples.extend(sampled_values)
		uniform_samples = np.array(uniform_samples).transpose()

		proposals = [sample for sample in uniform_samples]

		return np.array(proposals)



	def _proposal_optimization_thread(self, batch_index, queue):
		print('starting process for ', batch_index)
		# prepare penalty function
		def penalty(x):
			num, den = self.penalty_contributions(x)
			return (num + self.lambda_values[batch_index]) / den

		optimized = []
		for sample in self.proposals:
			if np.random.uniform() < 0.5:
				optimized.append(sample)
				continue
			res = minimize(penalty, sample, method = 'L-BFGS-B', options = {'maxiter': 25})

			# FIXME
			if np.any(res.x < self.var_lows) or np.any(res.x > self.var_highs):
				optimized.append(sample)
			else:
				optimized.append(res.x)
		optimized = np.array(optimized)
		optimized[:, self._ints] = np.around(optimized[:, self._ints])

		queue.put({batch_index: optimized})
		print('finished process for ', batch_index)



	def _optimize_proposals(self, proposals):
		self.proposals = proposals

		q = Queue()
		processes = []
		for batch_index in range(len(self.lambda_values)):
			process = Process(target = self._proposal_optimization_thread, args = (batch_index, q))
			processes.append(process)
			process.start()

		for process_index, process in enumerate(processes):
			process.join()

		result_dict = {}
		while not q.empty():
			results = q.get()
			for key, value in results.items():
				result_dict[key] = value

		samples = [result_dict[batch_index] for batch_index in range(len(self.lambda_values))]
		return np.array(samples)




	def sample(self, current_best, penalty_contributions, lambda_values, num_samples = 200):

		self.penalty_contributions = penalty_contributions
		self.lambda_values         = lambda_values

		# FIXME samples are not yet optimized!
		proposals = self._get_random_proposals(current_best, num_samples)
		print('# ... optimizing')
		proposals = self._optimize_proposals(proposals)
		return proposals

#========================================================================
