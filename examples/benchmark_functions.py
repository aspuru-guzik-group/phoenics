#!/usr/bin/env python

import numpy as np 

#===================================================

def dejong(vector):
	# takes k-dimensional vector
	# continuous, convex, unimodal
	# global minimum: x_i = 0 for all i in range(k)
	vector = np.array(vector)
	# rescale onto [-5, 5]
	vector = 10 * vector - 5 
	result = np.sum(vector**2)
	return result

#===================================================

def hyper_ellipsoid(vector):
	# takes k-dimensional vector
	# continuous, convex, unimodal
	# global minimum: x_i = 0 for all i in range(k)
	# rescale onto [-5, 5]
	vector = np.array(vector)
	vector = 10 * vector - 5
	weights = np.arange(1, len(vector) + 1)
	result = 0
	for index, element in enumerate(vector):
		result += weights[index] * element**2
	return result

#===================================================

def rosenbrock_function(vector):
	# takes k-dimensional vector
	# continuous, convex, unimodal, narrow valley
	# global minimum: x_i = 0 for all i in range(k)
	# rescale onto [-2, 2]
	vector = np.array(vector)
	vector = 4 * vector - 2
	result = 0
	for index, element in enumerate(vector[:-1]):
		result += 100 * (vector[index + 1] - element**2)**2  + (1 - element)**2
	return result

#===================================================

def rastrigin_function(vector):
	# takes k-dimensional vector
	# continuous, multi-modal, regular distribution of minima
	# global minimum: x_i = 0 for all i in range(k)
	result = 10. * len(vector)
	# rescale onto [-5, 5]
	vector = 10 * np.array(vector) - 5
	for index, element in enumerate(vector):
		result += element**2 - 10 * np.cos(2 * np.pi * element)
	return result 

#===================================================

def schwefel_function(vector):
	# takes k-dimensional vector
	# continuous, multi-modal
	# global minimum far away from local minima
	# global minimum: 418.9829 * n at x_i = 420.9687
	# rescale onto [-500, 500]
	vector = 1000 * np.array(vector) - 500
	result = 0
	for index, element in enumerate(vector):
		result += - element * np.sin(np.sqrt(np.abs(element)))
	return result

#===================================================

def ackley_path_function(vector):
	# takes k-dimensional vector
	# continuous, multi-model
	# global minimum: x_i = 0 
	# rescale onto [-32, 32]
	vector = 64 * np.array(vector) - 32
	a = 20.
	b = 0.2
	c = 2 * np.pi
	n = float(len(vector))
	vector = np.array(vector)
	result = - a * np.exp( - b * np.sqrt(np.sum(vector**2) / n ) ) - np.exp( np.sum(np.cos(c * vector)) / n ) + a + np.exp(1.)
	return result

#===================================================

def linear_funnel(vector):
	# takes k-dimensional vector
	vector = np.array(vector)
	vector = 10 * vector - 5
	bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
	result = 5
	for bound in bounds[::-1]:
		if np.amax(np.abs(vector)) < bound:
			result -= 1
	result = np.amin([4, result])
	return result


def narrow_funnel(vector):
	# takes k-dimensional vector
	vector = np.array(vector)
	vector = 100 * vector - 50
	bounds = [1.0, 2.0, 3.0, 4.0, 5.0]
	bounds = np.array(bounds)**2
	result = 5
	for bound in bounds[::-1]:
		if np.amax(np.abs(vector)) < bound:
			result -= 1
	result = np.amin([4, result])
	return result


def discrete_ackley(vector):
	# takes k-dimensional vector
	vector = np.array(vector)
	vector = 100 * vector - 50
	bounds = [1.0, 2.0, 3.0, 4.0, 5.0]
	bounds = np.array(bounds)**2
	result = 5
	for bound in bounds[::-1]:
		if np.amax(np.abs(vector)) < bound:
			result -= 1
	bounds = [1.25, 2.0, 2.5]
	bounds = np.array(bounds)**2

	domain = np.linspace(-50, 50, 10)
	dx = domain[1] - domain[0]
	imaged = np.array([np.amin(np.abs(element - domain)) for element in vector]) 
	new_res = 5
	for bound in bounds[::-1]:
		if np.amax(np.abs(imaged)) < bound:
			new_res -= 1
		result = np.amin([result, new_res])
	result = np.amin([4, result])
	return result



def discrete_michalewicz(vector):
	vector = np.array(vector)
	vector = 100 * vector - 40

	bounds = [1.0, 2.0, 3.0, 4.0, 5.0]
	bounds = np.array(bounds)**2
	result = 5
	for bound in bounds[::-1]:
		if np.amax(np.abs(vector)) < bound:
			result -= 1
	bounds = [1.25, 2.0, 2.5, 3.0]
	bounds = np.array(bounds)**2
	new_res = 5
	for bound in bounds[::-1]:
		if np.amin(np.abs(vector)) < bound:
			new_res -= 1
	result = np.amin([result, new_res, 4])
	return result



def double_well(vector):
	vector = np.array(vector)
	vector = 100 * vector - 15

	bounds = [1.25, 1.75, 2.0, 2.5, 2.75]
	bounds = np.array(bounds)**2
	result = 5
	for bound in bounds[::-1]:
		if np.amax(np.abs(vector)) < bound:
			result -= 1

	vector -= 50
	vector[0] += 10
	bounds = [2.5, 4.0, 5.0, 6.5]
	bounds = np.array(bounds)**2
	new_res = 5
	for bound in bounds[::-1]:
		if np.amax(np.abs(vector)) < bound:
			new_res -= 1

	result = np.amin([result, 4, new_res])
	return result



def discrete_valleys(vector):
	vector = np.array(vector)
	vector = 100 * vector - 50
	result = 5

	vector = vector - 50
	bounds = [1.25, 2.0, 2.5, 3.0]
	bounds = np.array(bounds)**2
	new_res = 5
	for bound in bounds[::-1]:
		if np.amin(np.abs(vector)) < bound:
			new_res -= 1
	result = np.amin([result, new_res, 4])

	vector = vector + 25
	bounds = [1.25, 2.0, 2.5, 3.0]
	bounds = np.array(bounds)**2
	new_res = 5
	for bound in bounds[::-1]:
		if np.amin(np.abs(vector)) < bound:
			new_res -= 1
	result = np.amin([result, new_res, 4])

	vector = vector + 25
	bounds = [1.25, 2.0, 2.5, 3.0]
	bounds = np.array(bounds)**2
	new_res = 5
	for bound in bounds[::-1]:
		if np.amin(np.abs(vector)) < bound:
			new_res -= 1
	result = np.amin([result, new_res, 4])

	vector = vector + 25
	bounds = [1.25, 2.0, 2.5, 3.0]
	bounds = np.array(bounds)**2
	new_res = 5
	for bound in bounds[::-1]:
		if np.amin(np.abs(vector)) < bound:
			new_res -= 1
	result = np.amin([result, new_res, 4])

	vector = vector + 25
	bounds = [1.25, 2.0, 2.5, 3.0]
	bounds = np.array(bounds)**2
	new_res = 5
	for bound in bounds[::-1]:
		if np.amin(np.abs(vector)) < bound:
			new_res -= 1
	result = np.amin([result, new_res, 4])


	vector = vector - 12.5
	bounds = [1.25, 1.5, 1.75, 2.0, 2.25]
	bounds = np.array(bounds)**2
	for bound in bounds[::-1]:
		if np.amax(np.abs(vector)) < bound:
			result -= 1

	vector[1] -= 50.
	bounds = [1.5, 1.75, 2.0]
	bounds = np.array(bounds)**2
	for bound in bounds[::-1]:
		if np.amax(np.abs(vector)) < bound:
			result -= 1

	vector[0] -= 50.
	bounds = [1.5, 1.75, 2.0]
	bounds = np.array(bounds)**2
	for bound in bounds[::-1]:
		if np.amax(np.abs(vector)) < bound:
			result -= 1

	vector[1] -= 25.
	bounds = [1.5, 1.75, 2.0]
	bounds = np.array(bounds)**2
	for bound in bounds[::-1]:
		if np.amax(np.abs(vector)) < bound:
			result -= 1

	vector[1] += 75.
	bounds = [1.5, 1.75, 2.0]
	bounds = np.array(bounds)**2
	for bound in bounds[::-1]:
		if np.amax(np.abs(vector)) < bound:
			result -= 1

	vector[1] -= 25.
	vector[0] += 25.
	bounds = [1.5, 1.75, 2.0]
	bounds = np.array(bounds)**2
	for bound in bounds[::-1]:
		if np.amax(np.abs(vector)) < bound:
			result -= 1

	if result < 0:
		result = 0
	return result

#======================================================

if __name__ == '__main__':

	import matplotlib.pyplot as plt
	import seaborn as sns
	import matplotlib.cm as cm
	sns.set_context('paper', font_scale = 2.0, rc = {'lines.linewidth': 3})
	sns.set_style('ticks')

	fig = plt.figure(figsize = (9, 6))
	ax0 = plt.subplot2grid((2, 3), (0, 0))
	ax1 = plt.subplot2grid((2, 3), (0, 1))
	ax2 = plt.subplot2grid((2, 3), (0, 2))
	ax3 = plt.subplot2grid((2, 3), (1, 0))
	ax4 = plt.subplot2grid((2, 3), (1, 1))
	ax5 = plt.subplot2grid((2, 3), (1, 2))
	axs = [ax0, ax1, ax2, ax3, ax4, ax5]

	plt.setp(ax1.get_yticklabels(), visible = False)
	plt.setp(ax2.get_yticklabels(), visible = False)
	plt.setp(ax4.get_yticklabels(), visible = False)
	plt.setp(ax5.get_yticklabels(), visible = False)
	plt.setp(ax0.get_xticklabels(), visible = False)
	plt.setp(ax1.get_xticklabels(), visible = False)
	plt.setp(ax2.get_xticklabels(), visible = False)

	losses = [linear_funnel, narrow_funnel, double_well, discrete_ackley, discrete_michalewicz, discrete_valleys]

	for loss_index, loss in enumerate(losses):
		domain = np.linspace(0, 1, 100)

		X, Y = np.meshgrid(domain, domain)
		Z = np.zeros((len(domain), len(domain)))
		for x_index, x in enumerate(domain):
			for y_index, y in enumerate(domain):
				value = loss([x, y])
				Z[x_index, y_index] = value

		levels = np.linspace(-0.5, 4.5, 6)

		pc = axs[loss_index].contourf(X, Y, Z, cmap = cm.RdYlBu, levels = levels)
		
	cbaxes = fig.add_axes([1.0, 0.115, 0.03, 0.815]) 
	cbar = plt.colorbar(pc, cax = cbaxes)
	cbar.set_ticks(range(5))
	cbar.set_ticklabels(range(5))

	plt.tight_layout()
	plt.savefig('discrete_loss_functions.png', bbox_inches = 'tight')
	plt.show()
