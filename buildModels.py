#################################
#			IMPORTS				#
#################################


import os
import pandas as pd
import numpy as np
import time

import pickle
import logging
import configparser
import ast
import sys
import csv


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import parallel_backend
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM



#################################
#		CLASS FUNCTIONS			#
#################################

class Build_Model:



	# Function Description: Returns the best configuration(params) for given method using cross validation and grid search
	# Input Parameters: 	name - Name of the model
	# 					  	model - Model with parameters
	# 					  	parameters - Optional Parameter(s) passed
	#						train_x - Training data feature columns
	# 				 		train_y - Training data target column
	# Return Values :		best_params_ - Best parameters for the given method
	# 				  		best_score_ - Score for the given best configuration
	#				  		best_estimator - Name of the best estimator
	def best_config(self, name, model, parameters, train_x, train_y):

		# Apply GridSearchCV for the given model and with all the parameters
		print('Finding best configuration params using Grid Search for : ' + name)
		clf = GridSearchCV(model, parameters, cv=3, verbose=5, n_jobs=4)

		
		with parallel_backend('threading'):
			clf.fit(train_x, train_y)
			
		best_estimator = clf.best_estimator_
		print('Best parameters: ' + name+'__'+str(clf.best_params_))
		
		return [name+'__'+str(clf.best_params_), clf.best_score_, best_estimator]


	# Function Description: Returns the best classifier model from a set of models for given training data using cross-validation.
	# Input Parameters: 	classifier_families - Name of the classifiers with parameters
	#					  	train_x - Predictor Column
	#					  	train_y - Target Column 
	# 					  	technique - Name of the feature selection technique used
	# 					  	columnName - Selected column name
	#						methods_list - List to store output for all the methods
	# Return Values :		best_classifier - Returns the best classifier along with the parameter
	def best_classifier_model(self, classifier_families, train_x, train_y, methods_list):
		
		best_quality = 0.0
		best_classifier = None
		classifiers = []

		# To call the modelling process for all the mentioned models
		for name, model, parameters in classifier_families:
			tempDict = {}
			best_conf = self.best_config(name, model, parameters, train_x, train_y)
			tempDict['bestParam'] = ast.literal_eval(best_conf[0].split('__')[1])#dict(best_conf[0].split('__')[1])
			tempDict['accuracy'] = float(best_conf[1])
			for method in methods_list:
				if method['modelMethodName'] == best_conf[0].split('__')[0]:
					method['bestConfiguration'].append(tempDict)
			classifiers.append(best_conf)

		
		for name, quality, classifier in classifiers:
			print('Best Configuration Parameter for Classifier: %s : with accuracy of %f', name, quality)
			if (quality > best_quality):
				best_quality = quality
				best_classifier = [name, classifier]
		
		print('Best classifier : %s', best_classifier[0])
		return best_classifier[0], best_classifier[1], method['bestConfiguration']


	# Function Description: Generates a model builder object for each of the method mentioned in the config
	# Input Parameters: 	models_list - List of algorithms for which models are to be built
	#						hyper_params - Dictionary of tuning parameters for all the algorithms for which models are to be built
	#						methods_list - List to store output for all the methods
	# Return Values:		Returns list of built models
	def classifier_models(self, models_list, hyper_params, methods_list):

		models = []
		print('Models List: ', models_list)
		input()
		# LinearRegression with parameters
		if("Linear" in set(models_list)):
			lg_tuned_parameters = hyper_params['Linear']
			models.append(["Linear", linear_model.LinearRegression(), lg_tuned_parameters])
			classifier_dict = {}
			classifier_dict['modelMethodName'] = 'Linear'
			classifier_dict['tuningParameters'] = lg_tuned_parameters
			classifier_dict['bestConfiguration'] = []
			methods_list.append(classifier_dict)

		# if("LSTM" in set(models_list)):
		# 	lstm_tuned_parameters = hyper_params['LSTM']
		# 	models.append(["LSTM", LSTM(units=50), lstm_tuned_parameters])
		# 	classifier_dict = {}
		# 	classifier_dict['modelMethodName'] = 'LSTM'
		# 	classifier_dict['tuningParameters'] = lstm_tuned_parameters
		# 	classifier_dict['bestConfiguration'] = []
		# 	methods_list.append(classifier_dict)
		

		#KNclassifier with parameters
		if("KNRegressor" in set(models_list)):
			KNRegressor_tuned_parameters = hyper_params['KNRegressor']
			models.append(["KNRegressor",KNeighborsRegressor(n_jobs=-1),KNRegressor_tuned_parameters])
			classifier_dict = {}
			classifier_dict['modelMethodName'] = 'KNRegressor'
			classifier_dict['tuningParameters'] = KNRegressor_tuned_parameters
			classifier_dict['bestConfiguration'] = []
			methods_list.append(classifier_dict)

		#DecisionTree with parameters
		if("DecisionTree" in set(models_list)):
			DecisionTree_tuned_parameters = hyper_params['DecisionTree']
			models.append(["DecisionTree",DecisionTreeClassifier(),DecisionTree_tuned_parameters])
			classifier_dict = {}
			classifier_dict['modelMethodName'] = 'DecisionTree'
			classifier_dict['tuningParameters'] = DecisionTree_tuned_parameters
			classifier_dict['bestConfiguration'] = []
			methods_list.append(classifier_dict)

		#MLPClassifier with parameters
		if("MLPClassifier" in set(models_list)):
			MLPClassifier_tuned_parameters =hyper_params['MLPClassifier']
			models.append(["MLPClassifier",MLPClassifier(alpha=1e-5, max_iter=100),MLPClassifier_tuned_parameters])
			classifier_dict = {}
			classifier_dict['modelMethodName'] = 'MLPClassifier'
			classifier_dict['tuningParameters'] = MLPClassifier_tuned_parameters
			classifier_dict['bestConfiguration'] = []
			methods_list.append(classifier_dict)

		print(methods_list)
		input()

		return models, methods_list

	

	# Function Description: Function to build and identify the best classifier model for Churn Analysis
	# Input Parameters: 	config - Configuration file contents
	#						features - Features returned from get_classifier_features
	# 				 		train_y - Training data target column
	# 				 		test_y - Test data target column
	# 				 		full_X - Input full data before splitting.
	#						pca - pca with parameters
	# 						file_info_dict - Dictionary containing all the required file paths.
	# Return Values : 		temp_output_list = List of outputs of models for the Model_Summary file	
	#						prob_res_df - Prediction probabilities results
	#						feature_names - Selected feature names
	#						model_results - Dictionary of model results for MongoDB
	def build_models(self, config, train_x, train_y, test_x, test_y):

		print("Initialising model building/training process")

		# Some initialisations
		current_time = time.time()
		best_accuracy_score = 0.0
		best_recall_score = 0.0
		best_f1_score = 0.0
		df_confusion = None
		class_report, class_report_dict = None, None
		best_clf_model = None
		clf_name = None
		selected_clf_feature = None
		model_results = {}
		methods_list = []

		models_list = set(set(ast.literal_eval(config["models_list"])))
		hyper_params = dict(ast.literal_eval(config["hyper_params"]))
		
		# To generate the model summary
		temp_output_list = []
		
		# Call to the modelling process
		print("Adding Classifier Models")
		classifier_families, methods_list = self.classifier_models(models_list, hyper_params, methods_list)
		
		print(methods_list)
		

		print('Modelling ... ' )
		name, model, best_config_res = self.best_classifier_model(classifier_families, train_x, train_y, methods_list)
		clf_name = name
		
		pred_y = model.predict(test_x)
		#pred_proba_y = model.predict_proba(test_x)
		#score = accuracy_score(test_y, pred_y)
		
		y_actu = pd.Series(test_y, name='Actual')
		y_pred = pd.Series(pred_y, name='Predicted')
		#y_proba_pred = pd.Series(pred_proba_y.tolist(), name='Predicted Probabilities')
		
		print("Prediction for Current Model with following Best Classifier is Completed", name)
		
		

# 		df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
		#class_report, class_report_dict = classification_report(test_y, pred_y), classification_report(test_y, pred_y, output_dict=True)
		best_clf_model = model
			
			
		
		model_results['methods'] = methods_list
		model_results['bestClassifierModel'] = {}

		print("################")		
		print("Best Model Configuration is %s", clf_name)
		temp_output_list.append(clf_name)
		model_results['bestClassifierModel']['modelDetails'] = {}
		model_results['bestClassifierModel']['modelDetails']['modelMethodName'] = clf_name.split('__')[0]
		model_results['bestClassifierModel']['modelDetails']['modelParams'] = ast.literal_eval(clf_name.split('__')[1])

		
		# print("Model Confusion Matrix: ")
		# print(df_confusion)
		# temp_output_list.append(str(df_confusion))
		# model_results['bestClassifierModel']['modelConfusionMatrix'] = str(df_confusion)

		print("Model Classification Report: ")
		print(class_report)
		temp_output_list.append(str(best_config_res))
		model_results['bestClassifierModel']['modelReport'] = class_report_dict

		print('Time taken to build current model with all mentioned methods : %s seconds', time.time() - current_time)
        

		return pred_y,temp_output_list, model_results


		