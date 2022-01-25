from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
from os import path




class KnasDatasetKmnistLogging:
	'''
		This class has implemented the KnasDatasetKmnistLogging which manages the 
		logging opeartions for the dataset
	'''
	def __init__(self):

		self.loggingCodes = {
			'TRAIN_ROOT_DIR_EXIST': 'Train Root Directory Exists, Loading... ',
			'TEST_ROOT_DIR_EXIST': 'Test Root Directory Exists, Loading... ',
			'TRAIN_DATA_DOWNLOAD': 'Downloading The Training Data',
			'TEST_DATA_DOWNLOAD': 'Downloading The Testing Data'

		}

	def knas_kmnist_log_message(self,message,status):

		'''
			This function will log the message with its corresponding status
		'''

		print("[{0}] {1} ".format(status,message))




class KnasDatasetKmnist:
	'''
		This class has implemented the KnasDatasetKmnist which manages the dataset
		for the KNAS program
	'''

	def __init__(self,datasetLogMod):

		# Root directory of the training data
		self.trainDataRoot = str()

		# Root directory of the testing data
		self.testDataRoot = str()

		# Training Data Object
		self.trainingData=None

		# Testing Data Object
		self.testingData=None

		# Logging module handler
		self.logModHand= datasetLogMod


	def set_traindata_root_dir(self,trainRootDirectory):

		'''
			This function sets the root directory for the training data
		'''
		self.trainDataRoot=trainRootDirectory


	def set_testdata_root_dir(self,testRootDirectory):

		'''
			This function sets the root directory for the testing data
		'''
		self.testDataRoot=testRootDirectory


	
	def load_traindata(self):

		'''
			This function will return the training data of the dataset
		'''

		# Check for the training data root directory
		if path.exists(self.trainDataRoot):
			self.logModHand.knas_kmnist_log_message(self.logModHand.loggingCodes['TRAIN_ROOT_DIR_EXIST'],'INF')

		else:

			self.logModHand.knas_kmnist_log_message(self.logModHand.loggingCodes['TRAIN_DATA_DOWNLOAD'],'INF')

		self.trainingData = KMNIST(root= self.trainDataRoot , train=True, download=True,transform=ToTensor())

			#TODO check return value

		
	def load_testdata(self):

		'''
			This function will return the testing data of the dataset
		'''

		if path.exists(self.testDataRoot):
			self.logModHand.knas_kmnist_log_message(self.logModHand.loggingCodes['TEST_ROOT_DIR_EXIST'],'INF')

		else:

			self.logModHand.knas_kmnist_log_message(self.logModHand.loggingCodes['TEST_DATA_DOWNLOAD'],'INF')


		self.testingData = KMNIST(root=self.testDataRoot, train=False, download=True, transform=ToTensor())



	def get_traindata(self):

		'''
			This function returns the training data
		'''
		return self.trainingData


	def get_testdata(self):

		'''
			This function returns the testing data
		'''
		return self.testingData








