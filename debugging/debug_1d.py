#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import sys
sys.path.append('../Phoenics_cutting_edge')
import copy
import numpy as np 

import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import seaborn as sns
sns.set_context('paper', font_scale = 2.0, rc = {'lines.linewidth': 4})
sns.set_style('ticks')


from phoenics import Phoenics
from Utils.utils import pickle_load, pickle_dump

#========================================================================

colors = ['b', 'y', 'r']

#========================================================================

def loss_func(vector):
    x = 20 * vector - 10
    values = np.sin(x / 0.5) * np.exp(-(x - 5)**2 / (2 * 5**2)) - 0.75 * np.sin(x / 0.5) * np.exp(-(x + 5)**2 / (2 * 5**2))
    return np.squeeze(values)

#========================================================================

class Manager(object):

	def __init__(self, config_file, loss_function):
		self.loss_function = loss_function
		self.chooser       = Phoenics(config_file)

		self.submitted_sets = []
		self.evaluated_sets = []
		self.all_losses     = []
		self.running_minima = []

		self.fig = plt.figure(figsize = (18, 4))
		self.ax0 = plt.subplot2grid((1, 2), (0, 0))
		self.ax1 = plt.subplot2grid((1, 2), (0, 1))

		content = open('logfile.dat', 'w')
		content.close()

		plt.ion()



	def _submit(self, samples):
		for sample in samples:
			sample_dict = sample
			
			self.submitted_sets.append(copy.deepcopy(sample_dict))
			pickle_dump(self.submitted_sets, self.chooser.param_dict['general']['submission_log'])

			sample_vector = np.array([sample_dict['param0']['samples'], sample_dict['param1']['samples']])
			sample_vector = np.squeeze(sample_vector)
			
			loss = self.loss_function(sample)
			sample_dict['loss'] = loss
			self.evaluated_sets.append(copy.deepcopy(sample_dict))
			pickle_dump(self.evaluated_sets, self.chooser.param_dict['general']['evaluation_log'])

			self.all_losses.append(loss)

		self._render(samples)


	def _render(self, samples):

		print('# rendering ...')
		self.ax0.cla()
		self.ax1.cla()

		domain = np.linspace(0, 1, 100)


		# plot the approximations to the loss function
		Z = np.zeros((3, len(domain)))
		lambda_values = [1., 0., -1.]
		try:
			for x_index, x in enumerate(domain):
				num, den = self.chooser.network.penalty_contributions(x)
				for batch_index in range(3):
					Z[batch_index, x_index] = (num + lambda_values[batch_index]) / den
		except AttributeError:
			pass

		start_index = 3
		for batch_index in range(3):
			ax = self.ax1
#			min_Z, max_Z = np.amin(Z), np.amax(Z)
#			if not min_Z == max_Z:
#				Z[batch_index] = (Z[batch_index] - min_Z) / (max_Z - min_Z) * 10.
			ax.plot(domain, Z[batch_index], color = colors[batch_index])


		# plot the actual loss function
		Z = np.zeros(len(domain))
		for x_index, x in enumerate(domain):
			value = self.loss_function(x)
			Z[x_index] = value
		min_Z, max_Z = np.amin(Z), np.amax(Z)
		if not min_Z == max_Z:
			Z = (Z - min_Z) / (max_Z - min_Z) * 10.

		self.ax1.plot(domain, Z, color = 'k', alpha = 0.5, lw = 3)



		# plot samples 
		start_index = 3
		for batch_index in range(self.chooser.param_dict['general']['batch_size']):
			ax = self.ax1
			obs_samples = self.chooser.obs_params[batch_index::3]

			for obs_sample in obs_samples:
				value = loss_func(obs_sample[0])
				value = (value - min_Z) / (max_Z - min_Z) * 10.
				ax.plot(obs_sample[0], value, ls = '', marker = 'o', color = colors[batch_index], markersize = 12, alpha = 0.5)

#		print(samples, len(samples))
		ax = self.ax1
#		if len(samples) == 1:
#			ax.plot(samples[0, 0], samples[0, 1], ls = '', marker = 'D', color = colors[batch_index], markersize = 15)
#		elif len(samples) == 3:
		for batch_index in range(self.chooser.param_dict['general']['batch_size']):
			value = loss_func(samples[batch_index])
			value = (value - min_Z) / (max_Z - min_Z) * 10.
			ax.plot(samples[batch_index], value, ls = '', marker = 'D', color = colors[batch_index], markersize = 15)

		# plot loss
		self.running_minima.append(np.log10(np.amin(self.all_losses)))
		self.ax0.plot(self.running_minima, color = 'k')

		plt.pause(0.5)
		print('# done rendering')


	def submit_random_batch(self):
		sets = self.chooser.choose(num_samples = self.chooser.param_dict['general']['num_batches'] * self.chooser.param_dict['general']['batch_size'])
		self._submit(sets)

	def submit_sampled_batch(self):
		obs  = pickle_load(self.chooser.param_dict['general']['evaluation_log'])
		sets = self.chooser.choose(num_samples = self.chooser.param_dict['general']['num_batches'], observations = obs)
		sets = sets[0]
		self._submit(sets)



	def run(self):
		print('# submitting random batch')
		self.submit_random_batch()
		for run_index in range(1, self.chooser.param_dict['general']['max_evals']):
			print('# working on iteration %d' % run_index)
			self.submit_sampled_batch()

#========================================================================

if __name__ == '__main__':

	np.random.seed(100691)
	manager = Manager('config_debug_1d.txt', loss_func)
	manager.run()