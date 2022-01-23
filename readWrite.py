#################################
#			IMPORTS				#
#################################


import os
import ast
import pandas as pd
import numpy as np
import time



#################################
#		CLASS FUNCTIONS			#
#################################

class Read_Write_Data:

	# Function Description :	This function reads input data file and stores its contents to a pandas dataframe.
	# Input Parameters : 		config - Configuration file contents
	# Return Values : 			data - Returns the input dataframe
	def read_data(self, config):

		print('Executing load_data()')
		current_time = time.time()

		# Getting configuration file details
		Input_List = list(ast.literal_eval(config['File_Data']))
		input_base_path = Input_List[0]['input_base_path']
		input_file_name = Input_List[0]['input_file_name']
		
		# Contatenate file path
		path_file_nm = input_base_path + input_file_name
		print('File Name : ', path_file_nm)
		
		# Read input data file. 
		data = pd.read_csv(path_file_nm, dtype=str)

		
		# Pre-processing null values
		data.replace('', np.nan, inplace=True)
		print('Input Dataset Columns : ', data.columns.tolist())
		print('Input Dataset Columns Data Types : ', data.dtypes)
		print('Input Dataframe size in memory(kB) : ', data.memory_usage(deep=True).sum()/1024)

		print('Exiting load_data(), Time taken to load : (seconds)', time.time() - current_time)
		return data




