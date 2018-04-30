#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import numpy as np 

from Acquisitions.optimization_algorithms import LBFGS, Adam, SimpleDiscrete, SimpleCategorical
from Utils.utils import VarDictParser

# FIXME: penalties are here expected to be minima!

#========================================================================

class ParameterOptimizer(VarDictParser):
	# for now, only LBFGS is supported as float optimization method

	dx = 1e-7

	def __init__(self, var_dicts):
		super(ParameterOptimizer, self).__init__(var_dicts)



	def _within_bounds(self, sample, option = None):
		pos = getattr(self, '_%s' % option)

		if np.any(sample[pos] < self.var_p_lows[pos]) or np.any(sample[pos] > self.var_p_highs[pos]):
			return False
		else:
			return True



	def _construct_positions(self, ignore):
		self.pos_floats = self._floats.copy()
		self.pos_ints   = self._ints.copy()
		self.pos_cats   = self._cats.copy()
		for pos in ignore:
			self.pos_floats[pos] = False
			self.pos_ints[pos]   = False
			self.pos_cats[pos]   = False

	#====================================================================


	def _float_optimization(self, sample):
		proposal = self.opt_float.get_update(sample)
		if self._within_bounds(proposal, option = 'floats'):
			return proposal
		else:
			return sample


	def _integer_optimization(self, sample):
		proposal = self.opt_int.get_update(sample)
		if self._within_bounds(proposal, option = 'ints'):
			return proposal
		else:
			return sample


	def _categorical_optimization(self, sample, highest = 1):
		proposal = self.opt_cat.get_update(sample)
		if self._within_bounds(proposal, option = 'cats'):
			return proposal
		else:
			return sample


	#====================================================================


	def optimize(self, penalty, sample, max_iter = 1, ignore = []):

		self.penalty   = penalty

		# choose optimization algorithms
		self._construct_positions(ignore)

		self.opt_float = LBFGS(self.penalty, pos = self.pos_floats)
		self.opt_int   = SimpleDiscrete(self.penalty, pos = self.pos_ints)
		self.opt_cat   = SimpleCategorical(self.penalty, pos = self.pos_cats, highest = self.var_p_options)

		sample_copy = sample.copy()
		# run optimization algorithms
		for num_iter in range(max_iter):

			# one step of LBFGS
			if np.any(self.pos_floats):
				optimized = self._float_optimization(sample_copy)
			# one step of integer perturbation 
			if np.any(self.pos_ints):
				optimized = self._integer_optimization(optimized)
			# one step of categorical perturbation
			if np.any(self.pos_cats):
				optimized = self._categorical_optimization(optimized)

			# check if we converged
			if np.linalg.norm(sample_copy - optimized) == 0:
				break
			else:
				sample_copy = optimized.copy()

		return optimized
