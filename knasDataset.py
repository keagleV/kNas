from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch import Generator

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

	def __init__(self):

		# Root directory of the training data
		self.trainDataRoot = str()

		# Root directory of the testing data
		self.testDataRoot = str()

		# Training Data Object
		self.trainingData=None

		# Validation Data Object
		self.valData=None

		# Testing Data Object
		self.testingData=None

		# Logging module handler
		self.logModHand= KnasDatasetKmnistLogging()

		# Dataset Transform
		self.DatasetTransforms = transforms.Compose([
           transforms.Resize(32),
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,)),


        ])


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

		self.trainingData = KMNIST(root= self.trainDataRoot , train=True, download=True,transform=self.DatasetTransforms)

			#TODO check return value

		
	def load_testdata(self):

		'''
			This function will return the testing data of the dataset
		'''

		if path.exists(self.testDataRoot):
			self.logModHand.knas_kmnist_log_message(self.logModHand.loggingCodes['TEST_ROOT_DIR_EXIST'],'INF')

		else:

			self.logModHand.knas_kmnist_log_message(self.logModHand.loggingCodes['TEST_DATA_DOWNLOAD'],'INF')


		self.testingData = KMNIST(root=self.testDataRoot, train=False, download=True, transform=self.DatasetTransforms)



	def split_training_dataset(self,trainSplit,seed):
		'''
			This function will split the training dataset to training data and testing data
		'''
		

		# Number of samples which are going to be used for training the model

		numTrainSamples = int(len(self.trainingData) * trainSplit)
		

		# Number of samples which are going to be used as validation of the model
		numValSamples = int(len(self.trainingData) * (1-trainSplit))


		(self.trainingData, self.valData) = random_split(self.trainingData,[numTrainSamples, numValSamples],
							generator= Generator().manual_seed(seed))


	def get_traindata(self):

		'''
			This function returns the training portion of the training data
		'''
		return self.trainingData


	def get_valdata(self):
		'''
			This function returns the validation part of the training data
		'''
		return self.valData

	def get_testdata(self):

		'''
			This function returns the testing data
		'''
		return self.testingData



	def get_traindata_dataloader(self,batchSize):

		'''
			This function returns the dataloader of the training data
		'''
		return DataLoader(self.trainingData, shuffle=True, batch_size=batchSize)

	def get_valdata_dataloader(self,batchSize):

		'''
			This function returns the dataloader of the validation data
		'''
		return DataLoader(self.valData, shuffle=True, batch_size=batchSize)


	def get_testdata_dataloader(self,batchSize):

		'''
			This function returns the dataloader of the testing data
		'''
		return DataLoader(self.testingData, shuffle=True, batch_size=batchSize)






