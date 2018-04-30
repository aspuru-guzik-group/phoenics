#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import pymc3 as pm 
import theano.tensor as T

import numpy as np 
from scipy.stats import dlaplace

#========================================================================


class DiscreteLaplace(pm.Discrete):

	def __init__(self, loc, scale, *args, **kwargs):
		super(DiscreteLaplace, self).__init__(*args, **kwargs)
		self.loc   = loc  
		self.scale = scale
		self.a     = 1. / (2. * self.scale**2)
		self.ea    = T.exp( - self.a)

		# normalization factor for discrete laplace
		x_upper = pm.math.ceil(self.loc) - self.loc
		x_lower = self.loc - pm.math.floor(self.loc)

		self.prefactor = T.log( (1 - T.exp(-self.a)) / (T.exp(-self.a * x_upper) + T.exp(-self.a * x_lower)) )



	def logp(self, value):
		return self.prefactor - pm.math.abs_(value - self.loc) * self.a 

#========================================================================
