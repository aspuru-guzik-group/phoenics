#!/usr/bin/env python 

import json 
import pickle

import numpy as np

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
		self.var_dicts  = var_dicts
		self.total_size = 0

		# store all information about all variables
		self._store_all_infos()
		# learn types of each variable
		self._store_variable_types()
		# get infos broken up by variable type
		self._store_type_specific_infos()


	def _store_all_infos(self):
		# get lists for storing all variables 
		for attr in ['var_sizes', 'var_names', 'var_lows', 'var_highs', 'var_types']:
			setattr(self, attr, [])
		for attr in ['var_p_sizes', 'var_p_names', 'var_p_lows', 'var_p_highs', 'var_p_types']:
			setattr(self, attr, [])
		# and store information
		for var_dict in self.var_dicts:
			var_size = var_dict[list(var_dict)[0]]['size']
			self.total_size += var_size
			var_name = list(var_dict)[0]

			self.var_names.append(var_name)
			self.var_sizes.append(var_size)
			self.var_lows.append(float(var_dict[var_name]['low']))
			self.var_highs.append(float(var_dict[var_name]['high']))
			self.var_types.append(var_dict[var_name]['type'])

			self.var_p_names.extend([var_name for i in range(var_size)])
			self.var_p_sizes.extend([var_size for i in range(var_size)])
			self.var_p_lows.extend([float(var_dict[var_name]['low']) for i in range(var_size)])
			self.var_p_highs.extend([float(var_dict[var_name]['high']) for i in range(var_size)])
			self.var_p_types.extend([var_dict[var_name]['type'] for i in range(var_size)])

		# need to convert everything into numpy arrays
		for attr in ['var_sizes', 'var_names', 'var_lows', 'var_highs', 'var_types']:
			setattr(self, attr, np.array(getattr(self, attr)))
		for attr in ['var_p_sizes', 'var_p_names', 'var_p_lows', 'var_p_highs', 'var_p_types']:
			setattr(self, attr, np.array(getattr(self, attr)))

		# get the ranges
		self.var_ranges   = self.var_highs - self.var_lows
		self.var_p_ranges = self.var_p_highs - self.var_p_lows

		# and store everything in dictionaries, just in case
		self.var_infos = {'var_names': self.var_names, 'var_sizes': self.var_sizes, 'var_types': self.var_types,
						  'var_lows': self.var_lows, 'var_highs': self.var_highs}
		self.var_p_infos = {'var_p_names': self.var_p_names, 'var_p_sizes': self.var_p_sizes, 'var_p_types': self.var_p_types,
						    'var_p_lows': self.var_p_lows, 'var_p_highs': self.var_p_highs}

	
	def _store_variable_types(self):
		# now we need to know the variable type for each entry
		for attr in ['_floats']:
			setattr(self, attr, np.array([False for i in range(self.total_size)]))
		self.var_p_type_indicators = np.empty(self.total_size)

		for var_index, var_type in enumerate(self.var_p_types):
			if var_type == 'float':
				self._floats[var_index] = True
				self.var_p_type_indicators[var_index] = 0
			else:
				raise NotImplementedError()



	def _store_type_specific_infos(self):
		float_dict = {}
		for attr, values in self.var_infos.items():
			float_dict[attr] = values[self._floats]
		self.var_p_infos_floats = float_dict

#=======================================================================


if __name__ == '__main__':
	parser = ParserJSON('config.txt')
	parser.parse()
	print(parser.param_dict)
	quit()
