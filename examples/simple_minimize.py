#!/usr/bin/env python

import sys, json, copy
sys.path.append('../')

import numpy as np 

from phoenics.phoenics import Phoenics

#========================================================

def dummy_loss(x):
	return 0.25 * x**4 - x**2 + 0.25 * x

#========================================================

# check the configuration file
config_file = 'my_model.txt'

# create an instance of Phoenics
phoenics = Phoenics(config_file)

# start minimization procedure
result = phoenics.minimize(dummy_loss, max_iter = 10)

# print the minimum
print('===============')
print(result.x, result.fx)
print('===============')

# print the path
for sample_index, sample in enumerate(result.params):
	print(sample_index, result.values[sample_index], sample)

#========================================================