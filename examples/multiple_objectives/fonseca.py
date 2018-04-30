#!/usr/bin/env python 

import numpy as np 

def fonseca(params):
	vector = np.array([params[key_name]['samples'][0] for key_name in ['x', 'y']])
	obj_0  = 1 - np.exp( - np.sum((vector - 1. / np.sqrt(len(vector)))**2))
	obj_1  = 1 - np.exp( - np.sum((vector + 1. / np.sqrt(len(vector)))**2))
	params['obj_0'] = obj_0
	params['obj_1'] = obj_1
	return params
