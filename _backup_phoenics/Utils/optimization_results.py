#!/usr/bin/env python

__author__ = 'Florian Hase'

#========================================================================

import numpy as np

#========================================================================

class OptimizeResult(object):

	message = ''
	nfev    = 0
	nit     = 0
	params  = []
	values  = []

	def __init__(self):
		self.status = -1

	def __str__(self):
		result = ''
		for index, element in enumerate(self.values):
			result += '# RESULT: %.5f\n' % element
		result += '========\n'
		return result[:-1]


	def add(self, param, value):
		self.params.append(param)
		self.values.append(value)


	def analyze(self):
		self.x  = self.params[np.argmin(self.values)]
		self.fx = np.amin(self.values)