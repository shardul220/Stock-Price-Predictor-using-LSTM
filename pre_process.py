#################################
#			IMPORTS				#
#################################


import os
import ast
import pandas as pd
import numpy as np
import time
import re


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



#################################
#		CLASS FUNCTIONS			#
#################################

class Pre_Process_Data:

	def process_missing_values(self, data):

		percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False).round(2)
		sum_missing = data.isna().sum().sort_values(ascending = False)
		missing_stock_data  = pd.concat([percent, sum_missing], axis=1, keys=['Percent', "Missing Count"])
		missing_stock_data.head(7)




	# Function Description :	This function is used to replace datatypes with the givin datatypes.
	# Input Parameters : 		data - Input dataframe.
	#							column_name - Name of the column to be processed
	#							value - Value with which datatypes will be changed.
	# Return Values : 			data - Returns the processed data
	def process_columns_datatypes(self, data, column_name, dtype):
		
		data[column_name] = data[column_name].astype(dtype)
		return data

	

	# Function Description :	This function is used to replace specific values in the data with the given values.
	# Input Parameters : 		data - Input dataframe.
	#							column_name - Name of the column to be processed
	#							condition - Condition for replacement
	#							value - Value with which the fields will be replaced.
	# Return Values : 			data - Returns the processed data
	def replace_values(self, data, column_name, condition, value):

		
		given_rows = data[column_name].isin([condition])
		data.loc[given_rows, column_name] = value

		return data


	# Function Description :	This function is used to extract day, month or year from the given date column.
	# Input Parameters : 		data - Input dataframe.
	#							data_col_name - Name of the column to be processed (date)
	#							column_name_new - Name of the new column to be formed
	#							flag - day/month/year
	# Return Values : 			data - Return the preprocessed data.
	def process_dates(self, data, data_col_name):

		fld = data[data_col_name]
		if not np.issubdtype(fld.dtype, np.datetime64):
			data[data_col_name] = fld = pd.to_datetime(fld, infer_datetime_format=True)
		targ_pre = re.sub('[Dd]ate$', '', data_col_name)
		for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
				'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
			data[targ_pre+n] = getattr(fld.dt,n.lower())
		data[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
		

		return data


	# Function Description :	This function is used to compute percent of one column w.r.t. another
	# Input Parameters : 		data - Input dataframe.
	#							column_name_new - Name of the new column to be formed
	#							column_numerator - Name of the column to be the numerator
	#							column_denomenator - Name of the column to be denomenator
	# Return Values : 			data - Return the preprocessed data.
	def get_percentages(self, data, column_name_new, column_numerator, column_denomenator):

		# Take the percentage of two columns and save it into a new column.
		#data[column_name_new] = data.apply(lambda row:((row[column_numerator]/row[column_denomenator])*100) ,axis=1)
		data.loc[:, column_name_new] = (data.loc[:, column_numerator] / data.loc[:, column_denomenator]) * 100
		return data



	# Function Description:	To split the data into feature and target columns
	# 						X -predictor column and Y - target column
	# Input Parameters: 	config - External configuration file.
	#						data - Processed data is passed as an input which is to be split into train and test.
	# Return Values: 		X_full_column_names - List of columns for the pre-processed dataframe.
	# 				 		column_names - List of columns in cosideration for modelling
	# 				 		X_full - Input full data before splitting.
	#				 		X - Feature data
	#				 		Y - Target Data
	def seggregate_features(self, config, data):

		print("Extracting features and targets")
		X = data.drop('Close', axis=1)
		Y = data['Close']
        
		
		
		return X ,Y



	# Function Description:	To get the features data after droppping and processing columns for testing.
	# Input Parameters: 	logger - For the logging output file.
	#						config - External configuration file.
	#						internal_config - Internal configuration file.
	#						data - Processed data is passed as an input which is to be split into train and test.
	# Return Values: 		X_features - Input full feature data without scaling.
	def seggregate_features_and_targets(self, config, data):

		print("Extracting features and targets")
		X = data.drop('Close', axis=1)
		Y = data['Close']

		

		column_names = X.columns
		
		return column_names, X ,Y


	# Function Description:	Prepare train and test split
	# Input Parameters: 	X - Feature data.
	#						Y - Target data.
	#						train_ratio - split ratio.
	# Return Values: 		train_x - Training data feature columns
	# 				 		train_y - Training data target column
	# 				 		test_x - Test data feature columns
	# 				 		test_y - Test data target column
	def split_train_test(self, X, Y, train_ratio):

		# Splitting the data
		train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=train_ratio, random_state = 0)

		return train_x, test_x, train_y, test_y


