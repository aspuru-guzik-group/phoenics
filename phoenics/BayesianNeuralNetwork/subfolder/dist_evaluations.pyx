#!/usr/bin/env python 

from __future__ import division
import cython
import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

from libc.math cimport exp, abs, tanh, round

#============================================================================

@cython.cdivision(True)
cdef double _gauss(double x, double loc, double scale):
	cdef double argument, result
	argument = (x - loc)**2 / (2. * scale**2)
	if argument > 5.:
		result = 0.
	else:
		# the number below is sqrt(2 * pi)
		result = exp( - argument) / (2.5066282746310002 * scale)
	return result

@cython.cdivision(True)
cdef double _discrete_laplace(double x, double loc, double scale):
	cdef double argument, result
	cdef double x_round = round(x)
	argument = abs(x_round - loc) / (2. * scale**2)
	if argument > 5.:
		result = 0.
	else:
		result = exp( - argument) * tanh(1. / (4. * scale**2))
	return result

#============================================================================

cdef class DistEvaluator:

	cdef int num_samples, num_obs, num_dim
	cdef double sqrt2pi

	cdef np.ndarray np_mus, np_sds
	cdef np.ndarray np_sds_square, np_tau_half, np_tanh_sds, np_gauss_prefactor
	cdef np.ndarray np_losses, np_var_types
	cdef np.ndarray np_probs


	def __init__(self, mus, sds, losses, var_types):

		self.np_mus = mus
		self.np_sds = sds
		self.np_losses = losses
		self.np_var_types = var_types

		self.num_samples = mus.shape[0]
		self.num_obs     = mus.shape[1]
		self.num_dim     = mus.shape[2]
	
		self.sqrt2pi  = (2. * 3.141592)**0.5
		self.np_probs = np.zeros(self.num_obs)


#	@cython.boundscheck(False)
	cdef double [:] _probs(self, double [:] x):

		cdef int sample_index, obs_index, dim_index
		cdef double total_prob

		cdef double [:, :, :] mus = self.np_mus
		cdef double [:, :, :] sds = self.np_sds
		cdef double [:] var_types = self.np_var_types

		cdef double [:] probs = self.np_probs
		for obs_index in range(self.num_dim):
			probs[obs_index] = 0.


		for sample_index in range(self.num_samples):
			for obs_index in range(self.num_obs):
				total_prob = 1.
				for dim_index in range(self.num_dim):
					if var_types[dim_index] == 0:
						func = _gauss
					elif var_types[dim_index] == 1:
						func = _discrete_laplace
					total_prob *= func(x[dim_index], mus[sample_index, obs_index, dim_index], sds[sample_index, obs_index, dim_index])
				probs[obs_index] += total_prob
		
		for obs_index in range(self.num_obs):
			probs[obs_index] /= self.num_samples

		return probs

	


	cpdef get_penalty(self, np.ndarray x):

		# first, we get the probabilities
		cdef double [:] x_memview = x
		probs_x = self._probs(x_memview)

		# then we get numerator and denominator
		cdef double num = 0.
		cdef double den = 0.
		cdef double [:] losses = self.np_losses

		for obs_index in range(self.num_obs):
			num += losses[obs_index] * probs_x[obs_index]
			den += probs_x[obs_index]

		return num, den


