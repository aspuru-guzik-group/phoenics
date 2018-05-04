#!/usr/bin/env python 

import sys 
sys.path.append('../../phoenics')
import uuid, time
import pickle
import numpy as np 
from threading import Thread

from phoenics import Phoenics 
from branin   import branin as loss

#========================================================================

class OptimizationManager(object):

	def __init__(self, config_file, loss_function):

		# creates instance of Phoenics optimizer
		self.phoenics      = Phoenics(config_file)
		self.loss_function = loss_function 

		# list for keeping track of submitted evaluations
		self.max_submissions       = 8
		self.submitted_evaluations = []



	def _evaluate(self, param, identifier):
		# set random delay time
		delay_time  = np.random.uniform(low = 0., high = 20.)
		time.sleep(delay_time)
		observation = self.loss_function(param)
		setattr(self, identifier, observation)



	def optimize(self, max_iter = 10):

		observations = []

		while len(observations) < max_iter:
			print('...iterating...', len(observations))

			# submit evaluations, if possible
			if len(self.submitted_evaluations) < self.max_submissions:
				params = self.phoenics.choose(observations = observations)
				np.random.shuffle(params)
				submission_index = 0
				while len(self.submitted_evaluations) < self.max_submissions:
					identifier = str(uuid.uuid4())
					thread = Thread(target = self._evaluate, args = (params[submission_index], identifier))
					thread.start()
					submission_index += 1
					setattr(self, identifier, 'running')
					self.submitted_evaluations.append(identifier)


			# check if evaluations terminated
			for identifier in self.submitted_evaluations:
				observation = getattr(self, identifier)
				if isinstance(observation, dict):

					observations.append(observation)
					self.submitted_evaluations.remove(identifier)

					# log observations in a pickle file for future analysis
					pickle.dump(observations, open('observations.pkl', 'wb'))

					# print observations to file 
					logfile = open('logfile.dat', 'a')
					new_line = ''
					for var_name in sorted(self.phoenics.var_names):
						for param_value in observation[var_name]['samples']:
							new_line += '%.5e\t' % (observation[var_name]['samples'])
					for obj_name in sorted(self.phoenics.loss_names):
						new_line += '%.5e\t' % (observation[obj_name])
					logfile.write(new_line + '\n')
					logfile.close()

			time.sleep(0.1)

#========================================================================

if __name__ == '__main__':

	logfile = open('logfile.dat', 'w')
	logfile.close()

	manager = OptimizationManager('config.json', loss)
	manager.optimize()
