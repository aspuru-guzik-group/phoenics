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
cdef double _gauss_periodic(double x, double loc, double scale, double var_range):
	cdef double argument, result, distance

	distance = abs(x - loc)
	if var_range - distance < distance:
		distance = var_range - distance

	argument = ( distance / scale)**2 * 0.5
	if argument > 2000.:
		result = 0.
	else:
		# the number below is sqrt(2 * pi)
		result = exp( - argument) / (2.5066282746310002 * scale)
	return result


@cython.cdivision(True)
cdef double _gauss(double x, double loc, double scale):
	cdef double argument, result
	argument = ( (x - loc) / scale)**2 * 0.5
	if argument > 2000.:
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
	if argument > 2000.:
		result = 0.
	else:
		result = exp( - argument) * tanh(1. / (4. * scale**2))
	return result


#============================================================================

cdef class DistEvaluator:

	cdef int num_samples, num_obs, num_dim
	cdef double sqrt2pi

	cdef np.ndarray np_mus_float, np_sds_float
	cdef np.ndarray np_mus_int, np_sds_int
	cdef np.ndarray np_cat_probs
	cdef np.ndarray np_sds_square, np_tau_half, np_tanh_sds, np_gauss_prefactor
	cdef np.ndarray np_losses
	cdef np.ndarray np_var_types, np_var_periodics, np_var_ranges
	cdef np.ndarray np_probs


	def __init__(self, mus_float, sds_float, mus_int, sds_int, cat_probs, losses, var_types, var_periodics, var_ranges):

		self.np_mus_float = mus_float
		self.np_sds_float = sds_float
		self.np_mus_int   = mus_int
		self.np_sds_int   = sds_int
		self.np_cat_probs = cat_probs
		self.np_losses    = losses
		self.np_var_types = var_types
		self.np_var_periodics = var_periodics
		self.np_var_ranges    = var_ranges

		self.num_samples = mus_float.shape[0]
		self.num_obs     = mus_float.shape[1]
		self.num_dim     = mus_float.shape[2]
	
		self.sqrt2pi  = (2. * 3.141592)**0.5
		self.np_probs = np.zeros(self.num_obs)


	@cython.boundscheck(False)
	cdef double [:] _probs(self, double [:] x):

		cdef int sample_index, obs_index, dim_index, cat_index
		cdef double total_prob

		cdef double [:, :, :] mus_float    = self.np_mus_float
		cdef double [:, :, :] sds_float    = self.np_sds_float
		cdef double [:, :, :] mus_int      = self.np_mus_int
		cdef double [:, :, :] sds_int      = self.np_sds_int
		cdef double [:, :, :, :] cat_probs = self.np_cat_probs

		cdef double [:] var_types     = self.np_var_types
		cdef long   [:] var_periodics = self.np_var_periodics
		cdef double [:] var_ranges    = self.np_var_ranges


		cdef double [:] probs = self.np_probs
		for obs_index in range(self.num_obs):
			probs[obs_index] = 0.

		for sample_index in range(self.num_samples):
			for obs_index in range(self.num_obs):
				total_prob = 1.
				cat_index = 0
				for dim_index in range(self.num_dim):

					if var_types[dim_index] == 0:
						if var_periodics[dim_index] == 1:
							total_prob *= _gauss_periodic(x[dim_index], mus_float[sample_index, obs_index, dim_index], sds_float[sample_index, obs_index, dim_index], var_ranges[dim_index])
						else:
							total_prob *= _gauss(x[dim_index], mus_float[sample_index, obs_index, dim_index], sds_float[sample_index, obs_index, dim_index])
					
					elif var_types[dim_index] == 1:
						total_prob *= _discrete_laplace(x[dim_index], mus_int[sample_index, obs_index, dim_index], sds_int[sample_index, obs_index, dim_index])
					
					elif var_types[dim_index] == 2:
						total_prob *= cat_probs[cat_index, sample_index, obs_index, int(round(x[dim_index]))]
						cat_index += 1
				probs[obs_index] += total_prob
		
		for obs_index in range(self.num_obs):
			probs[obs_index] /= self.num_samples

		return probs

	


	@cython.boundscheck(False)
	cpdef get_penalty(self, np.ndarray x):

		# first, we get the probabilities
		cdef double [:] x_memview = x
		probs_x = self._probs(x_memview)

		# then we get numerator and denominator
		cdef double num = 0.
		cdef double den = 0.
		cdef double [:] losses = self.np_losses

		cdef double temp_0, temp_1

		for obs_index in range(self.num_obs):
			temp_0 = losses[obs_index]
			temp_1 = probs_x[obs_index]
			num += temp_0 * temp_1
			den += temp_1

		return num, den


