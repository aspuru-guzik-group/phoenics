#!/usr/bin/env python

__author__ = 'Florian Hase'

#========================================================================

import numpy as np 
from multiprocessing import Process, Queue

from Utils.utils import VarDictParser

#========================================================================

class SampleSelector(VarDictParser):

	def __init__(self, var_dicts):
		VarDictParser.__init__(self, var_dicts)

		self.total_size = 0
		self.var_sizes  = []
		self.var_names  = []
		for var_dict in self.var_dicts:
			self.total_size += var_dict[list(var_dict)[0]]['size']
			self.var_sizes.append(int(var_dict[list(var_dict)[0]]['size']))
			self.var_names.append(list(var_dict)[0])



	def _compute_rewards_per_batch(self, batch_index, queue):
		proposals = self.proposals[batch_index]
		rewards   = np.empty(len(proposals))
		for sample_index, sample in enumerate(proposals):
			num, den = self.penalty_contribs(sample)
			penalty = (num + self.lambda_values[batch_index]) / den
			rewards[sample_index] = np.exp( - penalty)
		rewards = np.array(rewards)
		queue.put({batch_index: rewards})

	


	def _compute_rewards(self):
		q = Queue()
		processes = []
		for batch_index in range(self.batch_size):
			process = Process(target = self._compute_rewards_per_batch, args = (batch_index, q))
			processes.append(process)
			process.start()

		for process in processes:
			process.join()

		result_dict = {}
		while not q.empty():
			results = q.get()
			for key, value in results.items():
				result_dict[key] = value

		rewards = [result_dict[batch_index] for batch_index in range(self.batch_size)]
		for reward_index, reward in enumerate(rewards):
			setattr(self, 'rewards_%d' % reward_index, np.array(reward))



	def select(self, num_samples, proposals, penalty_contribs, lambda_values, characteristic_distances):
		self.num_samples      = num_samples
		self.proposals        = proposals
		self.penalty_contribs = penalty_contribs
		self.lambda_values    = lambda_values
		self.characteristic_distances = characteristic_distances
		self.batch_size       = len(self.lambda_values)

		self._compute_rewards()

		# now we collect the samples
		all_samples = []
		proposal_copy = np.copy(self.proposals)
		for sample_index in range(num_samples):
			new_samples = []

			for batch_index in range(self.batch_size):
				batch_proposals = proposal_copy[batch_index]

				# compute diversity punishments
				div_crits = np.ones(len(batch_proposals))
				if len(new_samples) > 0:	
					for sample_index, sample in enumerate(batch_proposals):
#						min_distance = np.amin([np.linalg.norm(sample - x) for x in new_samples])
#						min_distance = np.amin([np.linalg.norm(sample - x) for x in new_samples], axis = 0)
						min_distance = np.amin([np.abs(sample - x) for x in new_samples], axis = 0)
						div_crits[sample_index] = np.amin([1., np.amin(np.exp( 2. * (min_distance - self.characteristic_distances) / self.var_p_ranges))])

				# get reweighted rewards
				rewards              = getattr(self, 'rewards_%d' % batch_index)
				reweighted_rewards   = div_crits * rewards
#				reweighted_rewards   = rewards
				largest_reward_index = np.argmax(reweighted_rewards)
				new_sample = batch_proposals[largest_reward_index]
				new_samples.append(new_sample)

				# update reward of picked sample
				rewards[largest_reward_index] = 0.
				setattr(self, 'rewards_%d' % batch_index, rewards)

			all_samples.append(np.array(new_samples))
		all_samples = np.array(all_samples)
		if len(all_samples) == 1:
			all_samples = all_samples[0]

		return all_samples
