
from __future__ import division
import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

#=======

cdef class DistEvaluator:

	cdef int num_samples, num_obs
	cdef double sqrt2pi
	cdef np.ndarray mus, sds, observed_losses, var_types
	cdef np.ndarray sds_square, tau_half, tanh_sds
	cdef np.ndarray gauss_prefactor, laplace_prefactor	

	def __init__(self, num_samples, mus, sds, observed_losses, var_types):
		self.num_samples = num_samples
		self.mus = mus
		self.sds = sds
		self.observed_losses = observed_losses
		self.num_obs         = len(self.observed_losses)
		self.var_types       = var_types

		self.sqrt2pi = np.sqrt(2 * np.pi)
		self.gauss_prefactor = 1. / (self.sqrt2pi * self.sds)

		# now we pre-compute a few things
		self.sds_square = np.square(self.sds)
		self.tau_half   = 1. / (2. * self.sds_square)
		self.tanh_sds = np.tanh(1. / (4. * self.sds_square))


	cdef _gauss(self, float x, int sample_index, int dim_index):	
		result = np.empty(self.mus.shape[1])
		for obs_index in range(self.mus.shape[1]):
			argument = (x - self.mus[sample_index, obs_index, dim_index])**2 * self.tau_half[sample_index, obs_index, dim_index]
			if argument > 4.:
				result[obs_index] = 0.
			else:
				result[obs_index] = np.exp( - argument) * self.gauss_prefactor[sample_index, obs_index, dim_index]
		return result



	cdef _discrete_laplace(self, float x, int sample_index, int dim_index):
		x = np.around(x)
		result = np.empty(self.mus.shape[1])
		for obs_index in range(self.mus.shape[1]):
			argument = np.abs(x - self.mus[sample_index, obs_index, dim_index]) * self.tau_half[sample_index, obs_index, dim_index]
			if argument > 4.:
				result[obs_index] = 0.
			else:
				result[obs_index] = np.exp( - argument) * self.tanh_sds[sample_index, obs_index, dim_index]
		return result



	cdef _prob(self, x_1d, int sample_index):#mu_2d, sd_2d):
		# x_1d:  (#dim)
		# mu_2d, sd_2d: (#obs, #dim)

#		cdef np.ndarray total_prob = np.ones(self.num_obs)

		total_prob = np.ones(self.num_obs)
		for dim_index in range(self.var_types.shape[0]):
			if self.var_types[dim_index] == 0:
				func = self._gauss
			elif self.var_types[dim_index] == 1:
				func = self._discrete_laplace
			else:
				raise NotImplementedError()
			total_prob *= func(self, x_1d[dim_index], sample_index, dim_index)
#			total_prob *= func(self, x_1d[dim_index], mu_2d[:, dim_index], sd_2d[:, dim_index])
		return total_prob


#		return np.prod(self._gauss(x_1d, mu_2d, sd_2d), axis = 1)


#	cdef _prob(self, x_1d, mu_2d, sd_2d):



	cdef _probs(self, x):

		probs = np.empty(len(self.observed_losses))
	
		for sample_index in range(self.num_samples):
			probs += self._prob(x, sample_index)
	
		return probs / self.num_samples

#		return np.mean( [self._prob(x, i) for i in range(self.num_samples)], axis = 0)
	


	cpdef get_penalty(self, x):
#		print('X', x)
		probs_x = self._probs(x)
		return np.dot(self.observed_losses, probs_x), np.sum(probs_x)


