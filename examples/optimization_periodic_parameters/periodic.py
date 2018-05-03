#!/usr/bin/env python 

import numpy as np 


def periodic(params):
	x, y   = params['params']['samples'][0], params['params']['samples'][1]
	result = np.cos(x / (4. * np.pi)) + np.sin(y / (4. * np.pi))	
	params['obj'] = result
	return params