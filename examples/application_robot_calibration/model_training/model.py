#!/usr/bin/env python 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pickle 
import numpy as np 
import tensorflow as tf

from single_model import SingleModel


#=====================================================================

NUM_FOLDS = 10

#=====================================================================

class Model(object):

	def __init__(self, data_file, index_file, model_path, plot = False):
	
		self.models_are_loaded = False
		self.model_path = model_path
		self.plot       = plot

		self.dataset = pickle.load(open(data_file, 'rb'))
		self.indices = pickle.load(open(index_file, 'rb'))

		self._read_indices()
		self._assemble_training_sets()


	def _read_indices(self):
		self.work_indices  = self.indices['work_indices']
		self.test_indices  = self.indices['test_indices']
		self.train_indices = [self.indices['cross_validation_sets'][index]['train_indices'] for index in range(NUM_FOLDS)]
		self.valid_indices = [self.indices['cross_validation_sets'][index]['valid_indices'] for index in range(NUM_FOLDS)]


	def _assemble_training_sets(self):

		params  = self.dataset['parameters']
		areas   = self.dataset['peak_area']
		times   = self.dataset['execution_time']

		self.features = params[self.work_indices]
		self.targets  = np.array([areas[self.work_indices], times[self.work_indices]]).transpose()

		self.test_features = params[self.test_indices]
		self.test_targets  = np.array([areas[self.test_indices], times[self.test_indices]]).transpose()

		max_features  = np.amax(np.amax(self.features, axis = 0), axis = 0)
		min_features  = np.amin(np.amin(self.features, axis = 0), axis = 0)
		mean_features = np.mean(np.mean(self.features, axis = 0), axis = 0)
		std_features  = np.std(np.std(self.features, axis = 0), axis = 0)
		max_targets   = np.amax(self.targets, axis = 0)
		min_targets   = np.amin(self.targets, axis = 0)
		mean_targets  = np.mean(self.targets, axis = 0)
		details_dict  = {'min_features': min_features, 'max_features': max_features, 
						 'mean_features': mean_features, 'std_features': std_features,
						 'min_targets': min_targets, 'max_targets': max_targets, 'mean_targets': mean_targets, 
						 'features_shape': self.features.shape, 'targets_shape': self.targets.shape}
		pickle.dump(details_dict, open('dataset_details.pkl', 'wb'))
		self.dataset_details = 'dataset_details.pkl'

		self.train_features, self.train_targets = [], []
		self.valid_features, self.valid_targets = [], []
		for index in range(NUM_FOLDS):

			train_features = params[self.train_indices[index]]
			valid_features = params[self.valid_indices[index]]

			train_targets  = np.array([areas[self.train_indices[index]], times[self.train_indices[index]]]).transpose()
			valid_targets  = np.array([areas[self.valid_indices[index]], times[self.valid_indices[index]]]).transpose()

			self.train_features.append(train_features)
			self.train_targets.append(train_targets)
			self.valid_features.append(np.concatenate([valid_features for i in range(len(train_features) // len(valid_features))]))
			self.valid_targets.append(np.concatenate([valid_targets for i in range(len(train_targets) // len(valid_targets))]))



	def initialize_models(self, batch_size = 1):
		self.models = []
		self.graphs = [tf.Graph() for i in range(NUM_FOLDS)]
		for fold_index in range(NUM_FOLDS):
			with self.graphs[fold_index].as_default():
				single_model = SingleModel(self.graphs[fold_index], self.dataset_details, scope = 'fold_%d' % fold_index, batch_size = batch_size)
				self.models.append(single_model)

	def set_hyperparameters(self, hyperparam_dict):
		for model in self.models:
			model.set_hyperparameters(hyperparam_dict)

	def construct_models(self):
		for model_index, model in enumerate(self.models):
			print('constructing model %d ...' % model_index)
			with self.graphs[model_index].as_default():
				model.construct_graph()


	def _load_models(self, batch_size = 1):
		self.models = []
		self.graphs = [tf.Graph() for i in range(NUM_FOLDS)]
		for fold_index in range(NUM_FOLDS):
			with self.graphs[fold_index].as_default():
				single_model = SingleModel(self.graphs[fold_index], self.dataset_details, scope = 'fold_%d' % fold_index, batch_size = batch_size)
				single_model.restore('%s/Fold_%d/model.ckpt' % (self.model_path, fold_index))
				self.models.append(single_model)
		self.models_are_loaded = True



	def train(self):
		for model_index, model in enumerate(self.models):
			model.train(self.train_features[model_index], self.train_targets[model_index],
						self.valid_features[model_index], self.valid_targets[model_index], 
						model_path = '%s/Fold_%d' % (self.model_path, model_index), plot = self.plot)


	def predict(self, features):

		if not self.models_are_loaded: self._load_models(batch_size = len(features))

		pred_dict = {'samples': [], 'averages': [], 'uncertainties': []}

		for fold_index in range(NUM_FOLDS):
			single_pred_dict = self.models[fold_index].predict(features)
			for key in pred_dict.keys():
				pred_dict[key].append(single_pred_dict[key])

		for key in pred_dict.keys():
			pred_dict[key] = np.array(pred_dict[key])

		pred_dict['averages'] = np.mean(pred_dict['averages'], axis = 0)
		return pred_dict

#=====================================================================

if __name__ == '__main__':

	hyperparam_dict = {'REG': 0.1, 'LEARNING_RATE': 10**-3.0}
	dataset_file    = 'data_set/experimental_data.pkl'
	index_file      = 'data_set/cross_validation_indices.pkl'

	model = Model(dataset_file, index_file, model_path = './', plot = True)
	model.initialize_models(batch_size = len(model.train_features[0]))
	model.set_hyperparameters(hyperparam_dict)
	

#=== TRAINING

	model.construct_models()
	model.train()


#=== PREDICTING

#	model.set_hyperparameters()
#	pred = model.predict(model.test_features)

#	import matplotlib.pyplot as plt 
#	import seaborn as sns 
#	plt.plot(model.test_targets, pred['averages'], ls = '', marker = '.')
#	plt.show()	
