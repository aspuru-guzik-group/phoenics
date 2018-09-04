#!/usr/bin/env python 

import numpy as np 

def branin(params, a = 1., b = 5.1 / (4. * np.pi**2), c = 5. / np.pi, r = 6., s = 10., t = 1. / (8. * np.pi)):
	x, y = params['x']['samples'][0], params['y']['samples'][0]
	result = a * (y - b * x**2 + c*x - r)**2 + s * (1 - t) * np.cos(x) + s
	params['branin'] = result
	return params

#===================================================================

if __name__ == '__main__':
	# create contourplot of the branin function

	import matplotlib.pyplot as plt 
	import matplotlib.cm as cm
	import seaborn as sns 

	domain_x = np.linspace(-5, 10, 1000)
	domain_y = np.linspace(0,  15, 1000)
	X, Y     = np.meshgrid(domain_x, domain_y)

	Z = np.zeros((len(domain_x), len(domain_y)))
	for x_index, x in enumerate(domain_x):
		for y_index, y in enumerate(domain_y):
			param_dict = {'x': {'samples': [x]}, 'y': {'samples': [y]}}
			Z[x_index, y_index] = branin(param_dict)['branin']

	levels = np.linspace(np.amin(Z), np.amax(Z), 250)
	plt.contourf(Y, X, Z, cmap = cm.RdGy, levels = levels)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Branin optimiation')
	plt.show()