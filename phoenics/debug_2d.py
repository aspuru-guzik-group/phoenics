#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import copy
import time
import numpy as np 

import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import seaborn as sns
sns.set_context('paper', font_scale = 1.5, rc = {'lines.linewidth': 4})
sns.set_style('ticks')


from phoenics.phoenics import Phoenics
from Utils.utils import pickle_load, pickle_dump

#========================================================================

colors = ['b', 'y', 'r']

#========================================================================

np.random.seed(1000)
#targets = np.random.uniform(low = 0., high = 1., size = (4, 2))
#np.random.seed(1000)
#offsets = np.random.uniform(low = 0., high = 1.0, size = (4,))

targets = np.random.uniform(low = 0., high = 1., size = (4, 2))
normalization = np.array([6., 30.])
targets[:, 0] = 6 * targets[:, 0] - 4.
targets[:, 1] = 30 * targets[:, 1] + 30

#def loss_func(vector):
#	distance = np.amin([np.amax(np.abs(vector - targets[index])) + offsets[index] for index in range(len(targets))])
#	return distance #+ np.random.normal(0., 0.25)

def loss_func(vector):

	distances = np.array([np.linalg.norm( (vector - target) / normalization) for target in targets])
	return np.amin(distances)**0.25


#	vector_copy = np.copy(vector)
#	vector_copy[1] = (vector_copy[1] - 45.) / 15.	
#	return np.mean(np.abs(vector_copy))
#	return np.linalg.norm(vector_copy)    
	
#	vector = 64 * np.array(vector) - 32
#	vector[0] += 12
#	vector[1] -= 8
#	a = 20.
#	b = 0.2
#	c = 2 * np.pi
#	n = float(len(vector))
#	result = - a * np.exp( - b * np.sqrt(np.sum(vector**2) / n ) ) - np.exp( np.sum(np.cos(c * vector)) / n ) + a + np.exp(1.)
#	return result

#========================================================================

class Manager(object):

	def __init__(self, config_file, loss_function):
		self.loss_function = loss_function
		self.chooser       = Phoenics(config_file)

		self.submitted_sets = []
		self.evaluated_sets = []
		self.all_losses     = []
		self.running_minima = []

		self.fig = plt.figure(figsize = (9, 6))
		self.ax0 = plt.subplot2grid((2, 3), (0, 0))
		self.ax1 = plt.subplot2grid((2, 3), (0, 1))
		self.ax2 = plt.subplot2grid((2, 3), (0, 2))
		self.ax3 = plt.subplot2grid((2, 3), (1, 0))
		self.ax4 = plt.subplot2grid((2, 3), (1, 1))
		self.ax5 = plt.subplot2grid((2, 3), (1, 2))
		self.axs = [self.ax0, self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]

		content = open('logfile.dat', 'w')
		content.close()

		plt.ion()



	def _submit(self, samples):
		for sample in samples:

			sample_dict = {}
			lower, upper = 0, self.chooser.var_sizes[0]
			for var_index, var_name in enumerate(self.chooser.var_names):
				sample_dict[var_name] = {'samples': sample[lower:upper]}
				if var_index == len(self.chooser.var_names) - 1:
					break
				lower = upper
				upper += self.chooser.var_sizes[var_index + 1]
			self.submitted_sets.append(copy.deepcopy(sample_dict))
			pickle_dump(self.submitted_sets, self.chooser.param_dict['general']['submission_log'])

			loss = self.loss_function(sample)
			sample_dict['loss'] = loss
			self.evaluated_sets.append(copy.deepcopy(sample_dict))
			pickle_dump(self.evaluated_sets, self.chooser.param_dict['general']['evaluation_log'])

			# log events
			content = open('logfile.dat', 'a')
			newline = '%d\t%.5e\t' % (len(self.evaluated_sets), loss)
			for element in sample:
				newline += '%.4f  ' % (element)
			content.write(newline + '\n')
			content.close()

			self.all_losses.append(loss)

		self._render(samples)



	def _render(self, samples):
		print('# rendering ...')
		for ax in self.axs:
			ax.cla()

		domain_x = np.linspace(-4.0, 2.0, 30)
		domain_y = np.linspace(30.0, 60.0, 30)
		X, Y = np.meshgrid(domain_x, domain_y)

		# plot the approximations to the loss function
		Z = np.zeros((3, len(domain_x), len(domain_y)))
		try:
			lambda_values = self.chooser.network.lambda_values
			for x_index, x in enumerate(domain_x):
				for y_index, y in enumerate(domain_y):
					sample = np.array([x, y])
					num, den = self.chooser.network.penalty_contributions(sample)
					for batch_index in range(3):
						Z[batch_index, y_index, x_index] = (num + lambda_values[batch_index]) / (den)
		except AttributeError:
			pass

		start_index = 3
		for batch_index in range(3):
			ax = self.axs[start_index + batch_index]
			min_Z, max_Z = np.amin(Z[batch_index]), np.amax(Z[batch_index])
			if not min_Z == max_Z:
				Z[batch_index] = (Z[batch_index] - min_Z) / (max_Z - min_Z) * 10.
			levels = np.linspace(0., 10., 100)
			ax.contourf(X, Y, Z[batch_index], cmap = cm.binary, levels = levels, interpolation = 'none')
			


		# plot the actual loss function
		Z = np.zeros((len(domain_x), len(domain_y)))
		for x_index, x in enumerate(domain_x):
			for y_index, y in enumerate(domain_y):
				value = self.loss_function(np.array([x, y]))
				Z[y_index, x_index] = value
		min_Z, max_Z = np.amin(Z), np.amax(Z)
		if not min_Z == max_Z:
			Z = (Z - min_Z) / (max_Z - min_Z) * 10.
		levels = np.linspace(0., 10., 200)

		self.ax1.contourf(X, Y, Z, cmap = cm.binary, levels = levels)


		# plot samples 
		start_index = 3
		for batch_index in range(self.chooser.param_dict['general']['batch_size']):
			ax = self.axs[start_index + batch_index]
			obs_samples = self.chooser.obs_params[batch_index::3]

			for obs_sample in obs_samples:
				ax.plot(obs_sample[0], obs_sample[1], ls = '', marker = 'o', color = colors[batch_index], markersize = 6, alpha = 0.5)
				self.ax1.plot(obs_sample[0], obs_sample[1], ls = '', marker = 'o', color = colors[batch_index], markersize = 6, alpha = 0.5)


		if len(samples) == 1:
			ax.plot(samples[0, 0], samples[0, 1], ls = '', marker = 'D', color = colors[batch_index], markersize = 8)
			self.ax1.plot(samples[0, 0], samples[0, 1], ls = '', marker = 'D', color = colors[batch_index], markersize = 8)
		elif len(samples) == 3:
			for batch_index in range(self.chooser.param_dict['general']['batch_size']):
				ax = self.axs[start_index + batch_index]
				ax.plot(samples[batch_index][0], samples[batch_index][1], ls = '', marker = 'D', color = colors[batch_index], markersize = 8)
				self.ax1.plot(samples[batch_index][0], samples[batch_index][1], ls = '', marker = 'D', color = colors[batch_index], markersize = 8)

		# plot loss
		self.running_minima.append(np.amin(self.all_losses))
		self.ax0.plot(self.running_minima, color = 'k')

#		for target in targets:
#			self.ax1.plot(target[0], target[1], color = 'g', ls = '', marker = 'X', alpha = 0.5, markersize = 10)
#			self.ax3.plot(target[0], target[1], color = 'g', ls = '', marker = 'X', alpha = 0.5, markersize = 10)
#			self.ax4.plot(target[0], target[1], color = 'g', ls = '', marker = 'X', alpha = 0.5, markersize = 10)
#			self.ax5.plot(target[0], target[1], color = 'g', ls = '', marker = 'X', alpha = 0.5, markersize = 10)

#		for ax in [self.ax1, self.ax3, self.ax4, self.ax5]:
#			ax.set_xlim(0, 1)
#			ax.set_ylim(0, 1)

		plt.pause(0.5)
		print('# done rendering')


	def submit_random_batch(self):
		sets = self.chooser.choose(num_samples = self.chooser.param_dict['general']['num_batches'] * self.chooser.param_dict['general']['batch_size'])
		self._submit(sets)

	def submit_sampled_batch(self):
		obs  = pickle_load(self.chooser.param_dict['general']['evaluation_log'])
		start_time = time.time()
		sets = self.chooser.choose(num_samples = self.chooser.param_dict['general']['num_batches'], observations = obs)
		sets = sets[0]
		print('# ELAPSED TIME', time.time() - start_time)
		self._submit(sets)



	def run(self):
		print('# submitting random batch')
		self.submit_random_batch()
		for run_index in range(1, self.chooser.param_dict['general']['max_evals']):
			print('# working on iteration %d' % run_index)
			self.submit_sampled_batch()

#========================================================================

if __name__ == '__main__':
	import os
	try:
		os.remove('evaluated_sets.pkl')
		os.remove('submitted_sets.pkl')
	except:
		pass
	manager = Manager('config_debug_2d.txt', loss_func)
	manager.run()
