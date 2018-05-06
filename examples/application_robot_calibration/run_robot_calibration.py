#!/usr/bin/env python 

import sys 
sys.path.append('../../phoenics')
import pickle

from phoenics       import Phoenics 
from robot_emulator import RobotEmulator

#========================================================================

class OptimizationManager(object):

	def __init__(self, config_file, loss_function):

		# creates instance of Phoenics optimizer
		self.phoenics      = Phoenics(config_file)
		self.loss_function = loss_function 


	def optimize(self, max_iter = 100):

		observations = []

		for num_iter in range(max_iter):

			# query for new parameters based on prior observations
			params = self.phoenics.choose(observations = observations)

			# use parameters for evaluation ...
			# ... experimentally or computationally
			for param in params:
				observation = self.loss_function(param)
				observations.append(observation)

			# log observations in a pickle file for future analysis
			pickle.dump(observations, open('observations.pkl', 'wb'))

			# print observations to file 
			logfile = open('logfile.dat', 'a')
			for param in params:
				new_line = ''
				for var_name in sorted(self.phoenics.var_names):
					for param_value in param[var_name]['samples']:
						new_line += '%.5e\t' % (param[var_name]['samples'])
				for obj_name in sorted(self.phoenics.loss_names):
					new_line += '%.5e\t' % (param[obj_name])
				logfile.write(new_line + '\n')
			logfile.close()

#========================================================================

if __name__ == '__main__':

	robot   = RobotEmulator()

	logfile = open('logfile.dat', 'w')
	logfile.close()

	manager = OptimizationManager('config.json', robot.run_experiment)
	manager.optimize()
