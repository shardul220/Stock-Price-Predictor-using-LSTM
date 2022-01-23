#################################
#			IMPORTS				#
#################################


import importlib
import os
import configparser

from run import Generate_Models

# Main Function
if __name__ == '__main__':

	config = configparser.ConfigParser()
	config.read('config.ini')

	runObj = Generate_Models()
	runObj.initiate_processing(config)