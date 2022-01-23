#################################
#           IMPORTS             #
#################################


import os
import csv
import ast
import configparser
import numpy as np
import pandas as pd
import gc
import time

from pre_process import Pre_Process_Data
from readWrite import Read_Write_Data
from buildModels import Build_Model



#################################
#       CLASS FUNCTIONS         #
#################################

class Generate_Models:

    # Function Description :    Initialises modelling process for the single model approach
    #                           Calls the preprocessing module for the data arrangement
    #                           Splits the data into training and testing sets
    #                           Calls modelling process and predicts probabilities for the test dataset
    # Input Parameters :        config - External configuration file.
    #                           
    # Return Values :           None.
    def initiate_processing(self, config, eachCompany):

        target_column = set(set(ast.literal_eval(config["DATA_PROCESSING"]["target_column"])))


        # Calling module to load data.
        objReadWrite = Read_Write_Data()
        input_dataframe = objReadWrite.read_data(config['INPUT'])
        input_data = input_dataframe
        
        # Calling module to pre-process data
        objPreProcessing = Pre_Process_Data()

        Input_List = ast.literal_eval(config['DATA_PROCESSING']['Datatypes_processing'])
        for eachDict in Input_List:
            input_dataframe = objPreProcessing.process_columns_datatypes(input_dataframe, eachDict['Column_Name'], eachDict['dtype'])


    
        input_dataframe = objPreProcessing.process_dates(input_dataframe, 'Date')
    
        
        # Calling module to get the feature and target variables.
        #columnName, full_X, X ,Y = objPreProcessing.seggregate_features(config, input_dataframe)


        print(eachCompany)

        temp_input_dataframe = input_dataframe[input_dataframe['company_name']==eachCompany]
        temp_input_dataframe.set_index('Date', drop=True, inplace=True)
        temp_input_dataframe.drop(['company_name','Adj Close', 'High', 'Low', 'Open', 'Volume'], axis=1, inplace=True)

        columnName, X ,Y = objPreProcessing.seggregate_features_and_targets(config, temp_input_dataframe)

        # Call to split data into train and test
        train_x, test_x, train_y, test_y = objPreProcessing.split_train_test(X, Y, 0.75)
        #full_X.to_csv('Preprocessed_Training_Data.csv', sep=',')
        gc.collect()

        # To pass whole data as test data
        # test_x = X
        # test_y = Y

        # Numpy conversions for further processing
#         train_y = np.array(train_y[list(target_column)].values).ravel()
#         test_y = np.array(test_y[list(target_column)].values).ravel()



        # Calling module to build the models
        objBuildModel = Build_Model()


        # Feature selection process
        #features, pca, train_y, feature_results_dict = objBuildModel.get_classifier_features(logger, internal_config['CLASSIFIER'], train_x, train_y, test_x, columnName, trn_OP_model_coeff_filenm)

        prob_res_df_list, features_list = [], []
        y_pred,current_output, model_results_dict = objBuildModel.build_models(config['CLASSIFIER'],train_x, train_y, test_x, test_y)
        print(current_output, model_results_dict)
#           input()



        
        return input_dataframe, y_pred, train_x, train_y, test_x, test_y
        