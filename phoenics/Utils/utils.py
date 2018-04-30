#!/usr/bin/env python 

import json 
import pickle

import numpy as np

#=======================================================================

PERIODIC_DICT = {'True': 1, 'False': 0}

#=======================================================================

def pickle_load(file_name):
	return pickle.load(open(file_name, 'rb'))

def pickle_dump(dump_dict, file_name):
	pickle.dump(dump_dict, open(file_name, 'wb'))

#=======================================================================

class ParserJSON(object):

	def __init__(self, file_name = None):
		self.file_name = file_name


	def parse(self, file_name = None):
		if file_name:
			self.json = json.loads(file_name).read()
		else:
			self.json = json.loads(open(self.file_name).read())
		self.param_dict = self.json


#=======================================================================


class VarDictParser(object):

	def __init__(self, var_dicts):
		# we need to sort variables by type
		self.var_dicts     = var_dicts
		self.total_size    = 0
		self.complete_size = 0

		# store all information about all variables
		self._store_all_infos()
		# learn types of each variable
		self._store_variable_types()
		# get infos broken up by variable type
		self._store_type_specific_infos()


	def _store_all_infos(self):
		# get lists for storing all variables 
		for attr in ['var_sizes', 'var_names', 'var_lows', 'var_highs', 'var_types', 'var_options', 'var_keep_num', 'var_periodic', 'var_ranges']:
			setattr(self, attr, [])
		for attr in ['var_p_sizes', 'var_p_names', 'var_p_lows', 'var_p_highs', 'var_p_types', 'var_p_options', 'var_p_keep_num', 'var_p_periodic', 'var_p_ranges']:
			setattr(self, attr, [])
		for attr in ['var_e_sizes', 'var_e_names', 'var_e_lows', 'var_e_highs', 'var_e_types', 'var_e_options', 'var_e_begin', 'var_e_end', 'var_e_keep_num', 'var_e_periodic', 'var_e_ranges']:
			setattr(self, attr, [])
		# and store information
		for var_dict in self.var_dicts:
			var_name = list(var_dict)[0]
			var_size = var_dict[var_name]['size']
			self.total_size += var_size

			self.var_names.append(var_name)
			self.var_sizes.append(var_size)

			if 'keep_num' in var_dict[var_name]:
				self.var_keep_num.append(var_dict[var_name]['keep_num'])
			else:
				self.var_keep_num.append(var_dict[var_name]['size'])
			
			if 'periodic' in var_dict[var_name]:
				self.var_periodic.append(PERIODIC_DICT[var_dict[var_name]['periodic']])
			else:
				self.var_periodic.append(PERIODIC_DICT['False'])

			self.var_types.append(var_dict[var_name]['type'])
			if 'low' in var_dict[var_name].keys():
				self.var_lows.append(float(var_dict[var_name]['low']))
				self.var_highs.append(float(var_dict[var_name]['high']))
				self.var_ranges.append(float(var_dict[var_name]['high']) - float(var_dict[var_name]['low']))
				self.var_options.append('')
			else:
				self.var_lows.append(0.)
				self.var_highs.append(1.)
				self.var_options.append(var_dict[var_name]['options'])


			self.var_p_names.extend([var_name for i in range(var_size)])
			self.var_p_sizes.extend([var_size for i in range(var_size)])

			if 'keep_num' in var_dict[var_name]:
				self.var_p_keep_num.extend([var_dict[var_name]['keep_num'] for i in range(var_size)])
			else:
				self.var_p_keep_num.extend([var_dict[var_name]['size'] for i in range(var_size)])

			if 'periodic' in var_dict[var_name]:
				self.var_p_periodic.extend([PERIODIC_DICT[var_dict[var_name]['periodic']] for i in range(var_size)])
			else:
				self.var_p_periodic.extend([PERIODIC_DICT['False'] for i in range(var_size)])

			self.var_p_types.extend([var_dict[var_name]['type'] for i in range(var_size)])
			if 'low' in var_dict[var_name].keys():
				self.var_p_lows.extend([float(var_dict[var_name]['low']) for i in range(var_size)])
				self.var_p_highs.extend([float(var_dict[var_name]['high']) for i in range(var_size)])
				self.var_p_ranges.extend([float(var_dict[var_name]['high']) - float(var_dict[var_name]['low']) for i in range(var_size)])
				self.var_p_options.extend(['' for i in range(var_size)])
			else:
				self.var_p_lows.extend([0. for i in range(var_size)])
				self.var_p_highs.extend([1. for i in range(var_size)])
				self.var_p_options.extend([var_dict[var_name]['options'] for i in range(var_size)])

			if 'options' in var_dict[var_name].keys():
				var_size *= len(var_dict[var_name]['options'])
			begin_index         = self.complete_size
			self.complete_size += var_size
			end_index           = self.complete_size

			self.var_e_names.extend([var_name for i in range(var_size)])
			self.var_e_sizes.extend([var_size for i in range(var_size)])

			if 'keep_num' in var_dict[var_name]:
				self.var_e_keep_num.extend([var_dict[var_name]['keep_num'] for i in range(var_size)])
			else:
				self.var_e_keep_num.extend([var_dict[var_name]['size'] for i in range(var_size)])

			if 'periodic' in var_dict[var_name]:
				self.var_e_periodic.extend([PERIODIC_DICT[var_dict[var_name]['periodic']] for i in range(var_size)])
			else:
				self.var_e_periodic.extend([PERIODIC_DICT['False'] for i in range(var_size)])

			self.var_e_types.extend([var_dict[var_name]['type'] for i in range(var_size)])
			self.var_e_begin.extend([begin_index for i in range(var_size)])
			self.var_e_end.extend([end_index for i in range(var_size)])
			if 'low' in var_dict[var_name].keys():
				self.var_e_lows.extend([float(var_dict[var_name]['low']) for i in range(var_size)])
				self.var_e_highs.extend([float(var_dict[var_name]['high']) for i in range(var_size)])
				self.var_e_ranges.extend([float(var_dict[var_name]['high']) - float(var_dict[var_name]['low']) for i in range(var_size)])
				self.var_e_options.extend(['' for i in range(var_size)])
			else:
				self.var_e_lows.extend([0. for i in range(var_size)])
				self.var_e_highs.extend([1. for i in range(var_size)])
				self.var_e_options.extend([var_dict[var_name]['options'] for i in range(var_size)])


		# need to convert everything into numpy arrays
		for attr in ['var_sizes', 'var_names', 'var_lows', 'var_highs', 'var_types', 'var_options', 'var_periodic', 'var_ranges']:
			setattr(self, attr, np.array(getattr(self, attr)))
		for attr in ['var_p_sizes', 'var_p_names', 'var_p_lows', 'var_p_highs', 'var_p_types', 'var_p_options', 'var_p_periodic', 'var_p_ranges']:
			setattr(self, attr, np.array(getattr(self, attr)))
		for attr in ['var_e_sizes', 'var_e_names', 'var_e_lows', 'var_e_highs', 'var_e_types', 'var_e_options', 'var_e_periodic', 'var_e_ranges']:
			setattr(self, attr, np.array(getattr(self, attr)))
		self.var_periodic.astype(np.int32)
		self.var_p_periodic.astype(np.int32)
		self.var_e_periodic.astype(np.int32)

		# get the ranges
		self.var_ranges   = self.var_highs - self.var_lows
		self.var_p_ranges = self.var_p_highs - self.var_p_lows
		self.var_e_ranges = self.var_e_highs - self.var_e_lows

		# and store everything in dictionaries, just in case
		self.var_infos = {'var_names': self.var_names, 'var_sizes': self.var_sizes, 'var_types': self.var_types,
						  'var_lows': self.var_lows, 'var_highs': self.var_highs, 'var_options': self.var_options}
		self.var_p_infos = {'var_p_names': self.var_p_names, 'var_p_sizes': self.var_p_sizes, 'var_p_types': self.var_p_types,
						    'var_p_lows': self.var_p_lows, 'var_p_highs': self.var_p_highs, 'var_p_options': self.var_p_options,}

	
	def _store_variable_types(self):
		# now we need to know the variable type for each entry
		for attr in ['_floats', '_ints', '_cats']:
			setattr(self, attr, np.array([False for i in range(self.total_size)]))
		self.var_p_type_indicators = np.empty(self.total_size)

		for var_index, var_type in enumerate(self.var_p_types):
			if var_type == 'float':
				self._floats[var_index] = True
				self.var_p_type_indicators[var_index] = 0
			elif var_type == 'integer':
				self._ints[var_index] = True
				self.var_p_type_indicators[var_index] = 1
			elif var_type == 'categorical':
				self._cats[var_index] = True
				self.var_p_type_indicators[var_index] = 2
			else:
				raise NotImplementedError()


	def _store_type_specific_infos(self):
		float_dict = {}
		for attr, values in self.var_p_infos.items():
			float_dict[attr] = values[self._floats]
		self.var_p_infos_floats = float_dict

		int_dict = {}
		for attr, values in self.var_p_infos.items():
			int_dict[attr] = values[self._ints]
		self.var_p_infos_ints = int_dict

		cat_dict = {}
		for attr, values in self.var_p_infos.items():
			cat_dict[attr] = values[self._cats]
		self.var_p_infos_cats = cat_dict


#=======================================================================

class ObsDictParser(object):

	def __init__(self, obs_dicts):

		for att in ['loss_names', 'loss_hierarchies', 'loss_types', 'loss_tolerances']:
			setattr(self, att, [])

		# we need to get information sorted by hierarchy
		for obs_dict in obs_dicts:
			name = list(obs_dict.keys())[0]
			self.loss_names.append(name)

			self.loss_hierarchies.append(obs_dict[name]['hierarchy'])
			self.loss_types.append(obs_dict[name]['type'])
			self.loss_tolerances.append(obs_dict[name]['tolerance'])

		sort_indices = np.argsort(self.loss_hierarchies)
		for att in ['loss_names', 'loss_hierarchies', 'loss_types', 'loss_tolerances']:
			att_list = getattr(self, att)
			setattr(self, att, np.array(att_list)[sort_indices])

#		for att in ['loss_names', 'loss_hierarchies', 'loss_types', 'loss_tolerances']:
#			print(getattr(self, att))
#		quit()

#=======================================================================


if __name__ == '__main__':
	parser = ParserJSON('config.txt')
	parser.parse()
	print(parser.param_dict)
	quit()
