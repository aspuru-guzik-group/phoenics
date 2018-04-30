#!/usr/bin/env python

__author__ = 'Florian Hase'

#========================================================================

import numpy as np 

from ObservationParser.hierarchies import HierarchicalLossShaper
from Utils.utils import VarDictParser, ObsDictParser

#========================================================================

def heavyside(value):
	beta = 50.
	arg  = - beta * value
	return 1 / (1. + np.exp(arg))

#========================================================================

class ObservationParser(VarDictParser, ObsDictParser):

	def __init__(self, var_dicts, obs_dicts, softness = 0.01):
		VarDictParser.__init__(self, var_dicts)
		ObsDictParser.__init__(self, obs_dicts)

		self.softness = softness
		self.loss_shaper = HierarchicalLossShaper(self.loss_tolerances, self.softness)

		self.all_lower  = []
		self.all_upper  = []
		for var_index, full_var_dict in enumerate(self.var_dicts):
			var_dict = full_var_dict[self.var_names[var_index]]
			if 'low' in var_dict:
				self.all_lower.extend([var_dict['low'] for i in range(self.var_sizes[var_index])])
				self.all_upper.extend([var_dict['high'] for i in range(self.var_sizes[var_index])])
			else:
				self.all_lower.extend([0. for i in range(self.var_sizes[var_index])])
				self.all_upper.extend([len(var_dict['options']) for i in range(self.var_sizes[var_index])])
		self.all_lower = np.array(self.all_lower)
		self.all_upper = np.array(self.all_upper)

		self.soft_lower = self.all_lower + 0.1 * (self.all_upper - self.all_lower)
		self.soft_upper = self.all_upper - 0.1 * (self.all_upper - self.all_lower)
		self.soft_lower[self._cats] = -10**6
		self.soft_upper[self._cats] = 10**6



	def _get_mirrored_samples(self, sample):
		# first, we get the indices
		lower_indices = np.where(sample < self.soft_lower)[0]
		upper_indices = np.where(sample > self.soft_upper)[0]
		index_dict = {index: 'lower' for index in lower_indices}
		for index in upper_indices:
			index_dict[index] = 'upper'

		# now we start the mirroring procedure
		samples = []
		index_dict_keys   = list(index_dict.keys())
		index_dict_values = list(index_dict.values())
		for index in range(2**len(index_dict)):
			sample_copy = np.copy(sample)
			for jndex in range(len(index_dict)):

				if (index // 2**jndex) % 2 == 1:
					sample_index = index_dict_keys[jndex]
					if index_dict_values[jndex] == 'lower':
						sample_copy[sample_index] = self.all_lower[sample_index] - (sample[sample_index] - self.all_lower[sample_index])
					elif index_dict_values[jndex] == 'upper':
						sample_copy[sample_index] = self.all_upper[sample_index] + (self.all_upper[sample_index] - sample[sample_index])
			samples.append(sample_copy)
		if len(samples) == 0:
			samples.append(np.copy(sample))
		return samples




	def _rescale_losses(self, losses):

		hier_losses = self.loss_shaper.rescale_losses(losses)

		if np.amin(hier_losses) != np.amax(hier_losses):
			hier_losses = (hier_losses - np.amin(hier_losses)) / (np.amax(hier_losses) - np.amin(hier_losses))
			hier_losses = np.sqrt(hier_losses)
		else:
			hier_losses -= np.amin(hier_losses)

		return hier_losses
#		return losses[:, 0]




	def _get_sample_from_categorical(self, var_index, sample):
		options = self.var_options[var_index]
		parsed_sample = [options.index(element) for element in sample]
		return parsed_sample



	def parse(self, observ_dicts):
		raw_samples, raw_losses = [], []
		samples, losses = [], []
		for observ_dict in observ_dicts:

			# first, we get the sample
			sample = []
			for var_index, var_name in enumerate(self.var_names):
				observed_sample = observ_dict[var_name]['samples']
				if self.var_types[var_index] == 'categorical':
					observed_sample = self._get_sample_from_categorical(var_index, observed_sample)
				try:
					sample.extend(observed_sample)
				except TypeError:
					sample.append(observed_sample)
			sample = np.array(sample)
			raw_samples.append(sample)

			# now we need to mirror the sample
			mirrored_samples = self._get_mirrored_samples(sample)

			# get the current losses
			losses = np.array([observ_dict[loss_name] for loss_name in self.loss_names])

			# and now add them to the lists
			for sample in mirrored_samples:
				samples.append(np.array(sample))
				raw_losses.append(losses.copy())

		self._raw_obs_params = np.array(raw_samples)
		self._raw_obs_losses = np.array(raw_losses)

		# we close with rescaling the losses
		samples = np.array(samples)
		losses  = self._rescale_losses(np.array(raw_losses))

		return samples, losses
