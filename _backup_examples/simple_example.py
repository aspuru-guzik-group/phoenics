#!/usr/bin/env python

import sys, json, copy
sys.path.append('../')

import numpy as np 

from phoenics.phoenics import Phoenics

#========================================================

# check the configuration file
config_file = 'my_experiment.txt'

# create an instance of Phoenics
phoenics = Phoenics(config_file)

# phoenics is a uniform random sampler without any observations
sampled_params = phoenics.choose()

# you can get as many samples as you like
sampled_params = phoenics.choose(num_samples = 3)

# now let's fake a loss function for phoenics
evaluated_params = copy.deepcopy(sampled_params)
prior_losses = []
for sample_dict in evaluated_params:
    # calculate a dummy loss 
    loss = np.linalg.norm(sample_dict['param0']['samples']) + np.linalg.norm(sample_dict['param1']['samples'])
    prior_losses.append(loss)
    # store loss in the dictionary at the 'loss' keyword
    sample_dict['loss'] = loss

# phoenics now starts sampling
new_params = phoenics.choose(observations = evaluated_params)

print(new_params)