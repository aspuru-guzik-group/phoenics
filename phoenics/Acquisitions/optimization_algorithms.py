#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import numpy as np 

#========================================================================

class Optimizer(object):

	dx = 1e-6

	def __init__(self, penalty, *args, **kwargs):
		self.penalty = penalty
		for key, value in kwargs.items():
			setattr(self, str(key), value)


	def _get_gradients(self, sample, step = None):
		if step == None: step = self.dx
		gradients = np.zeros(len(sample))
		perturb   = np.zeros(len(sample))	
		for pos_index, pos in enumerate(self.pos):
			if not pos:
				continue
			perturb[pos_index]  += step
			grad             = (self.penalty(sample + perturb) - self.penalty(sample - perturb)) / (2. * step)
			gradients[pos_index] = grad
			perturb[pos_index]  -= step
		return gradients



#========================================================================

class Adam(Optimizer):
	# https://gist.github.com/Harhro94/3b809c5ae778485a9ea9d253c4bfc90a
	iterations = 0

	def __init__(self, penalty, eta = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, decay = 0., *args, **kwargs):
		super(Adam, self).__init__(penalty, *args, **kwargs)
		self.eta     = eta
		self.beta_1  = beta_1
		self.beta_2  = beta_2
		self.epsilon = epsilon
		self.decay   = decay
		self.initial_decay = self.decay


	def get_update(self, params):
		grads = self._get_gradients(params)
		eta   = self.eta
		if self.initial_decay > 0.:
			eta *= (1. / (1. + self.decay * self.iterations))

		next_iter = self.iterations + 1
		eta_next  = eta * (np.sqrt(1. - np.power(self.beta_2, next_iter)) / (1. - np.power(self.beta_1, next_iter)))

		if not hasattr(self, 'ms'):
			self.ms = [0. for param in params]
			self.vs = [0. for param in params]

		update = [None] * len(params)
		for index, param, grad, mass, vel in zip(range(len(params)), params, grads, self.ms, self.vs):
			m_next = (self.beta_1 * mass) + (1. - self.beta_1) * grad
			v_next = (self.beta_2 * vel)  + (1. - self.beta_2) * np.square(grad)
			p_next = param - eta_next * m_next / (np.sqrt(v_next) + self.epsilon)
			self.ms[index] = m_next
			self.vs[index] = v_next
			update[index]  = p_next

		self.iterations += 1
		return np.array(update)

#========================================================================

class LBFGS(Optimizer):

	def __init__(self, penalty, *args, **kwargs):
		from scipy.optimize import minimize
		self.minimize = minimize
		super(LBFGS, self).__init__(penalty, *args, **kwargs)


	def get_update(self, sample):
		res = self.minimize(self.penalty, sample, method = 'L-BFGS-B')
		return res.x

#========================================================================
#========================================================================


class SimpleDiscrete(Optimizer):

	def __init__(self, penalty, *args, **kwargs):
		super(SimpleDiscrete, self).__init__(penalty, *args, **kwargs)


	def get_update(self, sample):
		grads = self._get_gradients(sample, step = 1)
		new_sample = sample.copy()
		new_sample[self.pos] = np.where(grads[self.pos] > 1e-6,  new_sample[self.pos] - 1, new_sample[self.pos])
		new_sample[self.pos] = np.where(grads[self.pos] < -1e-6, new_sample[self.pos] + 1, new_sample[self.pos])
		return new_sample


#========================================================================
#========================================================================

class SimpleCategorical(Optimizer):

	def __init__(self, penalty, *args, **kwargs):
		super(SimpleCategorical, self).__init__(penalty, *args, **kwargs)

	def get_update(self, sample):
		# randomly check another category
		new_sample     = sample.copy()
		current_lowest = self.penalty(new_sample)
		for pos_index, pos in enumerate(self.pos):
			if not pos:
				continue
			perturbation = new_sample.copy()
			perturbation[pos_index] = np.random.choice(len(self.highest[pos_index]))
			candidate_penalty = self.penalty(perturbation)
			if candidate_penalty < current_lowest:
				new_sample = perturbation.copy()
				current_lowest = candidate_penalty
		return new_sample


#========================================================================
#========================================================================
#========================================================================


if __name__ == '__main__':
	import matplotlib.pyplot as plt 
	import time








#========================================================================

	def loss(sample):
		value = np.ceil(np.abs(sample - 5) - 0.1)
		return value**0.5


	plt.ion()

	domain = np.arange(-25, 26, 1)
	values = loss(domain)
	
#	plt.plot(domain, values, ls = '', marker = 's')
#	plt.show()

	optimizer = SimpleDiscrete(loss, pos = [0])
	sample = np.array([-20])
	for index in range(100):
		update = optimizer.get_update(sample)

		plt.clf()
		plt.plot(domain, values, ls = '', marker = 's', color = 'k')
		plt.plot(sample, loss(sample), ls = '', marker = 'o', color = 'g')
		plt.plot(update, loss(update), ls = '', marker = 'o', color = 'r')
		plt.pause(1)

		sample = update.copy()
	quit()

#========================================================================


	optimizer = Adam()
	sample = -3.2
	for index in range(1000):
		start = time.time()

		update = optimizer.get_update([sample], [4 * sample**3 - 2 * sample])

		print('\tTOOK', time.time() - start)
		update = update[0]
		print(index, sample, sample**4)
		
		plt.clf()
		plt.plot(domain, values, color = 'k', lw = 3)
		plt.plot(sample, sample**4 - sample**2 + 1, ls = '', marker = 'o', color = 'g')
		plt.plot(update, update**4 - update**2 + 1, ls = '', marker = 'o', color = 'r')
		plt.pause(1)

		sample = update

