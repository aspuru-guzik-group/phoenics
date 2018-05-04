#!/usr/bin/env python

__author__ = 'Florian Hase'

#========================================================================

import numpy as np 
np.warnings.filterwarnings('ignore')

#========================================================================

class Chimera(object):

	def __init__(self, loss_tolerances, smoothness = 0.0):
		self.smoothness      = smoothness
		self.loss_tolerances = loss_tolerances


	def soft_step(self, value):
		arg = - value / self.smoothness
		if arg < -12.:
			return 1.
		elif arg > 12.:
			return 0.
		else:
			return 1 / (1. + np.exp(arg))


	def hard_step(self, value):
		result = np.empty(len(value))
		result = np.where(value > 0., 1., 0.)
		return result


	def step(self, value):
		if self.smoothness < 1e-5:
			return self.hard_step(value)
		else:
			return self.soft_step(value)


	def _build_tolerances(self):
		shapes = self.unscaled_losses.shape
		scaled_losses = np.zeros((shapes[0] + 1, shapes[1]))

		mins, maxs, tols = [], [], []
		domain = np.arange(shapes[1])

		shift = 0
		for obj_index in range(len(self.unscaled_losses)):

			loss = self.unscaled_losses[obj_index]

			minimum = np.amin(loss[domain])
			maximum = np.amax(loss[domain])
			mins.append(minimum)
			maxs.append(maximum)

			tolerance = minimum + self.loss_tolerances[obj_index] * (maximum - minimum)

			# now shrink the region of interest
			interest = np.where(loss[domain] < tolerance)[0]
			if len(interest) > 0:
				domain   = domain[interest]

			tols.append(tolerance + shift)
			scaled_losses[obj_index] = self.unscaled_losses[obj_index] + shift

			if obj_index < len(self.unscaled_losses) - 1:
				shift -= np.amax(self.unscaled_losses[obj_index + 1][domain]) - tolerance
			else:
				shift -= np.amax(self.unscaled_losses[0][domain]) - tolerance
				scaled_losses[obj_index + 1] = self.unscaled_losses[0] + shift

		self.tols = np.array(tols)
		self.scaled_losses = scaled_losses


	def _construct_objective(self):
		loss = self.scaled_losses[-1].copy()
		for index in range(0, len(self.scaled_losses) - 1)[::-1]:
			loss *= self.step( - self.scaled_losses[index] + self.tols[index])
			loss += self.step(   self.scaled_losses[index] - self.tols[index]) * self.scaled_losses[index]
		self.loss = loss


	def scalarize_objectives(self, losses):
		for index in range(losses.shape[1]):
			min_loss, max_loss = np.amin(losses[:, index]), np.amax(losses[:, index])
			losses[:, index] = (losses[:, index] - min_loss) / (max_loss - min_loss)
			losses = np.where(np.isnan(losses), 0., losses)

		self.unscaled_losses = losses.transpose()
		self._build_tolerances()
		self._construct_objective()

		return self.loss.transpose()

#========================================================================

if __name__ == '__main__':

	# example on three one-dimensional objectives

	import matplotlib.pyplot as plt 
	fig = plt.figure(figsize = (5, 8))
	ax0 = plt.subplot2grid((2, 1), (0, 0))
	ax1 = plt.subplot2grid((2, 1), (1, 0))

	def obj_0(x):
		if x < 1:
			result = - 2 * x + 1
		elif x < 3.0:
			result = x - 2
		elif x < 3.5:
			result = 7 - 2 * x
		else:
			result = 2 * x - 7
		return result

	def obj_1(x):
		return 1 - 2 * np.exp(-0.15 * (x-2.0)**2) + 1.25 * np.exp( - 10 * (x - 1.7)**2)

	def obj_2(x):
		x = 6 - (x + 2)
		return 0.01 * (x+1)**2 + 0.01 * np.exp(- (x-2))


	tolerances = np.array([0.6, 0.4, 0.1])
	chimera    = Chimera(tolerances)


	domain = np.linspace(-1., 5., 1000)
	loss_0 = np.zeros(len(domain))
	for index, element in enumerate(domain):
		loss_0[index] = obj_0(element)
	loss_1 = obj_1(domain)
	loss_2 = obj_2(domain)
	loss_0 = (loss_0 - np.amin(loss_0)) / (np.amax(loss_0) - np.amin(loss_0))
	loss_1 = (loss_1 - np.amin(loss_1)) / (np.amax(loss_1) - np.amin(loss_1))
	loss_2 = (loss_2 - np.amin(loss_2)) / (np.amax(loss_2) - np.amin(loss_2))
	losses = np.array([loss_0, loss_1, loss_2])

	scaled = chimera.scalarize_objectives(losses.transpose())


	for index in range(len(losses)):
		ax0.plot(domain, losses[index], marker = '.', ls = '', label = 'obj %d' % index)
	ax1.plot(domain, scaled, marker = 'o', ls = '', color = 'k', label = 'chimera')

	ax0.legend()
	ax1.legend()

	ax0.set_xlabel('domain')
	ax1.set_xlabel('domain')
	ax0.set_ylabel('objective')
	ax1.set_ylabel('objective')

	plt.show()

