#!/usr/bin/env python 

import sys
sys.path.append('./model_training')
import numpy as np 

from model import Model

#======================================================================

class RobotEmulator(object):

	def __init__(self):

		hyperparam_dict = {'REG': 0.1, 'LEARNING_RATE': 10**-3.0}
		dataset_file    = 'model_training/data_set/experimental_data.pkl'
		index_file      = 'model_training/data_set/cross_validation_indices.pkl'

		self.robot = Model(dataset_file, index_file, model_path = './model_training/', plot = True)
		self.robot.initialize_models(batch_size = len(self.robot.train_features[0]))
		self.robot.set_hyperparameters(hyperparam_dict)


	def run_experiment(self, params):
		param_vector = np.array([params['param%d' % index]['samples'] for index in range(6)])
		param_vector = np.reshape(param_vector, (1, len(param_vector)))
		preds        = self.robot.predict(param_vector)
		params['peak_area']      = - preds['averages'][0, 0]
		params['execution_time'] = preds['averages'][0, 1]
		params['sample']         = params['param1']['samples']
		return params


#======================================================================

if __name__ == '__main__':

	# example for using the robot emulator

	robot = RobotEmulator()

	import matplotlib.pyplot as plt 
	import seaborn as sns 

	fig = plt.figure(figsize = (12, 6))
	ax0 = plt.subplot2grid((1, 2), (0, 0))
	ax1 = plt.subplot2grid((1, 2), (0, 1))

	plt.ion()

	ax0.plot([0, 2500], [0, 2500], color = 'k', lw = 2, ls = '-')
	ax1.plot([40, 140], [40, 140], color = 'k', lw = 2, ls = '-')
	plt.pause(0.05)

	pred_targets = []
	for feature_index, feature in enumerate(robot.robot.test_features):

		feature_dict = {'param%d' % index: {'samples': feature[index]} for index in range(6)}
		preds = robot.run_experiment(feature_dict)

		pred_target = np.array([- feature_dict['peak_area'], feature_dict['execution_time']])

		ax0.plot(robot.robot.test_targets[feature_index, 0], pred_target[0], marker = '.')
		ax1.plot(robot.robot.test_targets[feature_index, 1], pred_target[1], marker = '.')

		plt.pause(0.05)
