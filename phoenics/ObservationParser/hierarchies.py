#!/usr/bin/env python

__author__ = 'Florian Hase'

#========================================================================

import numpy as np 
np.warnings.filterwarnings('ignore')

#========================================================================

class HierarchicalLossShaper(object):

	def __init__(self, loss_tolerances, softness = 0.0):
		self.softness = softness
		self.loss_tolerances = loss_tolerances


	def soft_step(self, value):
		arg = - value / self.softness
		return 1 / (1. + np.exp(arg))

	def hard_step(self, value):
		result = np.empty(len(value))
		result = np.where(value > 0., 1., 0.)
		return result


	def step(self, value):
		if self.softness < 1e-5:
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


	def rescale_losses(self, losses):
		for index in range(losses.shape[1]):
			min_loss, max_loss = np.amin(losses[:, index]), np.amax(losses[:, index])
			losses[:, index] = (losses[:, index] - min_loss) / (max_loss - min_loss)
			losses = np.where(np.isnan(losses), 0., losses)

#		print(losses.shape)
#		quit()
		self.unscaled_losses = losses.transpose()

		self._build_tolerances()
		self._construct_objective()

		return self.loss.transpose()

#========================================================================

if __name__ == '__main__':


#	def obj_0(x):
#		result = np.zeros(len(x))
#		result = np.where(3.5 < x, 2 * x - 7, x)
#		result = np.where(x <= 3.5, 7 - 2 * x, result)
#		result = np.where(x <= 3, x - 2, result)
#		result = np.where(x < 1, -2 * x + 1, result)
#		return result

	def obj_0(x):
		if x < 1:
			result = - 2 * x + 1
		elif x <= 3:
			result = x - 2
		elif x < 3.5:
			result = 7 - 2 * x
		else:
			result = 2 * x - 7
		return result

	def obj_1(x):
		return 1 - 2 * np.exp(-(x-2.5)**2)

	def obj_2(x):
		x = 6 - (x + 2)
		return 0.01 * (x+1)**2 + 0.01 * np.exp(- (x-2))




	import matplotlib.pyplot as plt 
#	tolerances = np.zeros(3) + 0.55
	tolerances = np.array([0.3, 0.4, 0.5])
	shaper = HierarchicalLossShaper(tolerances)


	for num_points in range(2, 500, 10):

		num_points = 1000

		np.random.seed(100691)
		domain = np.random.uniform(-1., 5., num_points)
		loss_0 = np.zeros(len(domain))
		for index, element in enumerate(domain):
			loss_0[index] = obj_0(element)
		loss_1 = obj_1(domain)
		loss_2 = obj_2(domain)
		losses = np.array([loss_0, loss_1, loss_2])

		scaled = shaper.rescale_losses(losses.transpose())

		plt.clf()
		for index in range(len(losses)):
			plt.plot(domain, losses[index], marker = '.', ls = '')


		plt.plot(domain, scaled, marker = 'o', ls = '', color = 'k')
		plt.xlim(-1., 5.)
#		plt.ylim(-0.12, 0.45)
#		plt.pause(10.)
		plt.show()
