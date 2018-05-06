#!/usr/bin/env python 

import os
import pickle 
import numpy as np 
import tensorflow as tf 
import edward as ed 

#==========================================================================

class SingleModel(object):

	NUM_SAMPLES   = 100

	REG           = 0.1
	LEARNING_RATE = 10**-3.5
	MLP_SIZE      = 148

	def __init__(self, graph, dataset_details, scope, batch_size):

		self.graph           = graph
		self.batch_size      = batch_size 
		self.scope           = scope
		self.dataset_details = dataset_details 

		self.is_graph_constructed = False

		self._read_scaling_details()
#		self._construct_graph()


	def _generator(self, arrays, batch_size):
		starts = [0] * len(arrays)
		while True:
			batches = []
			for i, array in enumerate(arrays):
				start = starts[i]
				stop  = start + batch_size
				diff  = stop - array.shape[0]
				if diff <= 0:
					batch = array[start : stop]
					starts[i] += batch_size
				else:
					batch = np.concatenate([array[start:], array[:diff]])
					starts[i] = diff
				batches.append(batch)
			yield batches



	def _read_scaling_details(self):
		details             = pickle.load(open(self.dataset_details, 'rb'))
		self.scaling        = {key: details[key] for key in details}
		self.features_shape = self.scaling['features_shape']
		self.targets_shape  = self.scaling['targets_shape']


	def get_scaled_features(self, features):
		scaled = (features - self.scaling['min_features']) / (self.scaling['max_features'] - self.scaling['min_features'])
		return scaled

	def get_scaled_targets(self, targets):
#		scaled = (targets) / (targets + self.scaling['mean_targets'])
		scaled = targets / self.scaling['mean_targets']
		return scaled

	def get_raw_targets(self, targets):
#		raw = self.scaling['mean_targets'] * (targets / (1. - targets))
		raw = targets * self.scaling['mean_targets']
		return raw


	def set_hyperparameters(self, hyperparam_dict):
		for key, value in hyperparam_dict.items():
			setattr(self, key, value)



	def construct_graph(self):

		leaky_relu     = lambda y: tf.nn.leaky_relu(y, 0.2)
		mlp_activation = leaky_relu

		with self.graph.as_default():
			with tf.name_scope(self.scope):

				self.is_training = tf.placeholder(tf.bool, shape = ())
				self.x_ph = tf.placeholder(tf.float32, [self.batch_size, self.features_shape[1]])
				self.y_ph = tf.placeholder(tf.float32, [self.batch_size, self.targets_shape[1]])

				#=== PRIOR
				dim = self.features_shape[1]
				self.weight_0 = ed.models.Laplace(loc = tf.zeros((dim, self.MLP_SIZE)), scale = self.REG * tf.ones((dim, self.MLP_SIZE)))
				self.bias_0   = ed.models.Laplace(loc = tf.zeros(self.MLP_SIZE),        scale = self.REG * tf.ones(self.MLP_SIZE))
				
				self.weight_1 = ed.models.Laplace(loc = tf.zeros((self.MLP_SIZE, self.MLP_SIZE)), scale = self.REG * tf.ones((self.MLP_SIZE, self.MLP_SIZE)))
				self.bias_1   = ed.models.Laplace(loc = tf.zeros(self.MLP_SIZE),                  scale = self.REG * tf.ones(self.MLP_SIZE))
				
				self.weight_2 = ed.models.Laplace(loc = tf.zeros((self.MLP_SIZE, self.targets_shape[1])), scale = self.REG * tf.ones((self.MLP_SIZE, self.targets_shape[1])))
				self.bias_2   = ed.models.Laplace(loc = tf.zeros(self.targets_shape[1]),                  scale = self.REG * tf.ones(self.targets_shape[1]))

				self.arg_0    = tf.matmul(self.x_ph, self.weight_0) + self.bias_0
				self.net_0    = mlp_activation(self.arg_0)
				self.drop_0   = tf.layers.dropout(self.net_0, rate = 0.2, training = self.is_training)

				self.arg_1    = tf.matmul(self.drop_0, self.weight_1) + self.bias_1
				self.net_1    = mlp_activation(self.arg_1)
				self.drop_1   = tf.layers.dropout(self.net_1, rate = 0.2, training = self.is_training)


				self.arg_2    = tf.matmul(self.drop_1, self.weight_2) + self.bias_2

				self.alpha    = ed.models.Laplace(loc = tf.zeros(self.arg_2.get_shape()[-1]), scale = tf.ones(self.arg_2.get_shape()[-1]))

				self.net_2    = tf.maximum(0., self.arg_2) + self.alpha * tf.minimum(0., self.arg_2)

				self.y        = ed.models.Normal(self.net_2, scale = 10.**-3.5)
				self.priors   = [self.weight_0, self.bias_0, self.weight_1, self.bias_1, self.weight_2, self.bias_2]


				#=== POSTERIOR
				dim = self.features_shape[1]
				self.q_weight_0       = ed.models.Laplace(loc = tf.Variable(tf.zeros([dim, self.MLP_SIZE])),                   scale = tf.nn.softplus(tf.Variable(tf.zeros([dim, self.MLP_SIZE]))))
				self.q_bias_0         = ed.models.Laplace(loc = tf.Variable(tf.zeros([self.MLP_SIZE])),                        scale = tf.nn.softplus(tf.Variable(tf.zeros([self.MLP_SIZE]))))
				self.q_weight_1       = ed.models.Laplace(loc = tf.Variable(tf.zeros([self.MLP_SIZE, self.MLP_SIZE])),         scale = tf.nn.softplus(tf.Variable(tf.zeros([self.MLP_SIZE, self.MLP_SIZE]))))
				self.q_bias_1         = ed.models.Laplace(loc = tf.Variable(tf.zeros([self.MLP_SIZE])),                        scale = tf.nn.softplus(tf.Variable(tf.zeros([self.MLP_SIZE]))))
				self.q_weight_2       = ed.models.Laplace(loc = tf.Variable(tf.zeros([self.MLP_SIZE, self.targets_shape[1]])), scale = tf.nn.softplus(tf.Variable(tf.zeros([self.MLP_SIZE, self.targets_shape[1]]))))
				self.q_bias_2         = ed.models.Laplace(loc = tf.Variable(tf.zeros([self.targets_shape[1]])),                scale = tf.nn.softplus(tf.Variable(tf.zeros([self.targets_shape[1]]))))
				self.q_alpha          = ed.models.Laplace(loc = tf.zeros(self.arg_2.get_shape()[-1]),             scale = tf.nn.softplus(tf.Variable(tf.zeros(self.arg_2.get_shape()[-1]))))

				self.y_post     = ed.copy(self.y, {self.weight_0: self.q_weight_0, self.bias_0: self.q_bias_0,
												   self.weight_1: self.q_weight_1, self.bias_1: self.q_bias_1,
												   self.weight_2: self.q_weight_2, self.bias_2: self.q_bias_2,
												   self.alpha: self.q_alpha})

				self.posteriors = [self.q_weight_0, self.q_bias_0, self.q_weight_1, self.q_bias_1, self.q_weight_2, self.q_bias_2]



	def _construct_inference(self):
		self.is_graph_constructed = True

		print('constructing graph')
		with self.graph.as_default():

			self.inference_dict = {}
			for prior_index, prior_element in enumerate(self.priors):
				self.inference_dict[prior_element] = self.posteriors[prior_index]
			self.optimizer      = tf.train.AdamOptimizer(self.LEARNING_RATE)

			self.inference = ed.KLqp(self.inference_dict, data = {self.y: self.y_ph})

			try:
				self.inference.initialize(optimizer = self.optimizer, n_iter = 10**8, var_list = tf.trainable_variables())
			except RecursionError:
				print('recursion error')
				print(self.inference_dict)
				quit()

			self.sess = tf.Session(graph = self.graph)
			with self.sess.as_default():
				tf.global_variables_initializer().run()




	def train(self, train_features, train_targets, valid_features, valid_targets, model_path, plot = False):

		if not os.path.isdir(model_path): os.mkdir(model_path)
		logfile = open('%s/logfile.dat' % model_path, 'w')
		logfile.close()


		if not self.is_graph_constructed: self._construct_inference()

		with self.graph.as_default():
			with self.sess.as_default():

				train_feat_scaled = self.get_scaled_features(train_features)
				train_targ_scaled = self.get_scaled_targets(train_targets)

				valid_feat_scaled = self.get_scaled_features(valid_features)
				valid_targ_scaled = self.get_scaled_targets(valid_targets)
				min_target, max_target = np.minimum(np.amin(train_targets, axis = 0), np.amin(valid_targets, axis = 0)), np.maximum(np.amax(train_targets, axis = 0), np.amax(valid_targets, axis = 0))

				batch_train_data = self._generator([train_feat_scaled, train_targ_scaled], self.batch_size)
				batch_valid_data = self._generator([valid_feat_scaled, valid_targ_scaled], self.batch_size)

				train_errors = []
				valid_errors = []

				if not os.path.isdir(model_path): os.mkdir(model_path)
				self.saver = tf.train.Saver()

				if plot:
					import matplotlib.pyplot as plt 
					import seaborn as sns 
					colors = sns.color_palette('RdYlGn', 4)

					plt.ion()
					plt.close()
					plt.style.use('dark_background')
					fig = plt.figure(figsize = (16, 8))
					ax0 = plt.subplot2grid((1, 2), (0, 0))
					ax1 = plt.subplot2grid((1, 2), (0, 1))


				for epoch in range(self.inference.n_iter):

					train_x, train_y = next(batch_train_data)
					valid_x, valid_y = next(batch_valid_data)

					self.inference.update(feed_dict = {self.x_ph: train_x, self.y_ph: train_y, self.is_training: True})

					if epoch % 1000 == 0:

						valid_preds = self.y_post.sample(self.NUM_SAMPLES).eval(feed_dict = {self.x_ph: valid_x, self.is_training: False})
						valid_preds_mean = np.mean(valid_preds, axis = 0)
						valid_error = np.sqrt(np.mean(np.square(valid_preds_mean - valid_y), axis = 0))
						valid_errors.append(valid_error)

						train_preds = self.y_post.sample(self.NUM_SAMPLES).eval(feed_dict = {self.x_ph: train_x, self.is_training: False})
						train_preds_mean = np.mean(train_preds, axis = 0)
						train_error = np.sqrt(np.mean(np.square(train_preds_mean - train_y), axis = 0))
						train_errors.append(train_error)

						logfile = open('%s/logfile.dat' % model_path, 'a')
						logfile.write('%d\t%.5e\t%.5e\t%.5e\t%.5e\n' % (epoch, train_error[0], train_error[1], valid_error[0], valid_error[1]))
						logfile.close()

						# get minimum of validation; get changes of validation
						# break if minimum far away but changes detected

						if valid_error[0] < 1.05 * np.amin(np.array(valid_errors)[:, 0]) and valid_error[1] < 1.05 * np.amin(np.array(valid_errors)[:, 1]):
							last_save_index = len(valid_errors)
							print('saving model ...')
							self.saver.save(self.sess, '%s/model.ckpt' % model_path)

						if len(valid_errors) > 20:
							changes = []
							for change_index in range(1, 21):
								changes.append(valid_errors[ - change_index] - valid_errors[ - change_index - 1])
							average_changes = np.abs(np.mean(changes))
							std_changes     = np.std(changes)
							print('EVALUATION', len(valid_errors) - last_save_index, average_changes, std_changes)
							if (average_changes < 10**-3 and std_changes < 10**-2) or len(valid_errors) - last_save_index > 100:
								break

						if plot:
							train_preds_scaled = self.get_raw_targets(train_preds_mean)
							train_trues_scaled = self.get_raw_targets(train_y)
							valid_preds_scaled = self.get_raw_targets(valid_preds_mean)
							valid_trues_scaled = self.get_raw_targets(valid_y)

							ax0.cla()
							ax1.cla()

							ax0.plot([min_target[0], max_target[0]], [min_target[0], max_target[0]], lw = 3, color = 'w', alpha = 0.5)
							ax0.plot(train_trues_scaled[:, 0], train_preds_scaled[:, 0], marker = '.', ls = '', color = colors[-1], alpha = 0.5)
							ax0.plot(valid_trues_scaled[:, 0], valid_preds_scaled[:, 0], marker = '.', ls = '', color = colors[0], alpha = 0.5)

							ax1.plot([min_target[1], max_target[1]], [min_target[1], max_target[1]], lw = 3, color = 'w', alpha = 0.5)
							ax1.plot(train_trues_scaled[:, 1], train_preds_scaled[:, 1], marker = '.', ls = '', color = colors[-1], alpha = 0.5)
							ax1.plot(valid_trues_scaled[:, 1], valid_preds_scaled[:, 1], marker = '.', ls = '', color = colors[0], alpha = 0.5)

#							ax1.plot(train_errors, color = colors[-1], lw = 3, alpha = 0.8)
#							ax1.plot(valid_errors, color = colors[0], lw = 3, alpha = 0.8)
#							ax1.set_yscale('log')

							plt.pause(0.05)



	def restore(self, model_path):
		if not self.is_graph_constructed: 
			self.construct_graph()
			self._construct_inference()

		self.sess  = tf.Session(graph = self.graph)
		self.saver = tf.train.Saver()
		self.saver.restore(self.sess, model_path)




	def predict(self, input_raw):\

		input_scaled = self.get_scaled_features(input_raw)

		with self.sess.as_default():
			output_scaled = self.y_post.sample(self.NUM_SAMPLES).eval(feed_dict = {self.x_ph: input_scaled, self.is_training: False})

		output_raw      = self.get_raw_targets(output_scaled)
		output_raw_mean = np.mean(output_raw, axis = 0)
		output_raw_std  = np.std(output_raw, axis = 0)

		return {'samples': output_raw, 'averages': output_raw_mean, 'uncertainties': output_raw_std} 
