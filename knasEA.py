from knasLN import CNLayer
from knasLN import DFCLayer
from knasModel import KNasModel

from random import randint
from random import choice
from random import random

from math import floor




class KNasEALogging():
	'''
		This class has implemented the logging operations for the EA
		algorithm used in the KNAS program
	'''
	def __init__(self):

		self.loggingCodes = {
			
			'INIT_POPULATION_GEN':'Generating The Initial Population',

		
		}

	def knasea_log_message(self,message,status,lineNumer=None):

		'''
			This function will log the message with its corresponding status
		'''

		print("[{0}]{1} {2} ".format(status,'' if lineNumer is None else ' L-'+str(lineNumer)+'  ' ,message))




class KNasEAIndividual:
	'''
		This class has implemented the individual solutions of the EA algorithm
	'''
	def __init__(self,maxCNLayers,device):

		# Fitness value of the individual
		self.fitnessVal=int()

		# Number of learnable parameters of this individual
		self.numLearnParams = int()

		# Number of CN layers in the first part of the network
		self.cnLayersCount=int()

		# List of CN layers in the first part of the network
		self.cnLayersList=list()

		# DFC layer, the only layer in the last part of the network
		self.dfcLayer=None

		# Device to be used to create the layers of this individual
		self.device=device

		# Learning rate
		self.learningRate = float()

		# Input dimenesion: 1 * dim * dim 
		self.inputDimension = 32

		# Possible values for the filter count
		self.filterPossValues = [ 2, 4 , 8, 16, 32, 64, 128 ]

		# Possible values for the batch norm
		self.batchNormMaxValue = 100

		# Possible values for the activation function
		self.actFuncPossValues = ['relu','sigmoid']

		# Possible values for the dfc hidden layer neurons
		self.dfcHiLaPossNeuronValues = [ 2, 4 , 8, 16, 32, 64, 128 , 256 , 512 ]


		self.create_random_individual(maxCNLayers)



	def create_random_individual(self,maxCNLayers):
		'''
			This function creates a random indivi
		'''

		# Setting the leraning rate
		self.learningRate=random()
		

		# Decide on number of the CN layers
		self.cnLayersCount = randint(1,maxCNLayers)


		# Creating CN Layers

		# Output channesl of the last layer
		lastLayerOutCh = 1

		for i in range(self.cnLayersCount):



			# Number of filters in this layer
			currLaFilterCnt = choice(self.filterPossValues)

			# In the case of having convultional layer, 
			# the output dimentsion would be:
			#
			# 						(W-F+2p)/s + 1
			# W: dim size
			# F: kernel size
			# P: padding
			# S: stride
			#
			#
			self.inputDimension = ( (self.inputDimension - 2 + 2)//1 ) +1



			# Batch norm value
			batchNorm = choice([randint(1,self.batchNormMaxValue),None])

			if batchNorm:
				self.numLearnParams+=1

			# Activation function value
			actFunction= choice([ *self.actFuncPossValues , None ])
			
			if actFunction:
				self.numLearnParams+=1
			

			# Dropout value
			dropout = choice ([random(),None])

			if dropout:
				self.numLearnParams +=1


			# Maxpool value
			maxPool = choice([ 2, None ])

			

			# If we have maxpooling , we divide the dimension
			if maxPool:
				# Increment learnable parameters
				self.numLearnParams+=1
				
				self.inputDimension = (self.inputDimension // 2 )


			# Adding a new CN layer to the previous layers
			self.cnLayersList.append(CNLayer(lastLayerOutCh,currLaFilterCnt,2,1,1,batchNorm,actFunction,dropout,maxPool).create_cn_layer().to(self.device))
			
			# Updating the last layer output channel
			lastLayerOutCh = currLaFilterCnt
			



		# Creating DFC layer

		# First define number of the hidden layers
		numOfHiddenLayers = choice([0,1,2])


		if numOfHiddenLayers==0:
			self.dfcLayer= DFCLayer( lastLayerOutCh * (self.inputDimension**2)  ,10,None,None,None,None,None,None,None,None).create_dfc_layer().to(self.device)
		else:

			# We have to create at least 1 layer until this moment


			fhNumOfNeurons = choice(self.dfcHiLaPossNeuronValues)

			# Increment learnable params for the number of first hidden layer neurons
			self.numLearnParams+=1


			# Batch norm value
			fhBatchNorm = choice([randint(1,self.batchNormMaxValue),None])

			if fhBatchNorm:
				self.numLearnParams+=1
			
			# Activation function value
			fhActFunction= choice([ *self.actFuncPossValues,None])
			
			if fhActFunction:
				self.numLearnParams+=1

			
			# Dropout value
			fhDropout = choice ([random(),None])

			if fhDropout:
				self.numLearnParams+=1

			# We only have one hidden layer
			if numOfHiddenLayers==1:
				self.dfcLayer= DFCLayer(lastLayerOutCh * (self.inputDimension**2)  ,10,fhNumOfNeurons,fhBatchNorm,fhActFunction,fhDropout,None,None,None,None).create_dfc_layer().to(self.device)

			else:

				# Creating parameters for the second layer
				secNumOfNeurons = choice(self.dfcHiLaPossNeuronValues)

				# Increment learnable params for the number of first hidden layer neurons
				self.numLearnParams+=1

				# Batch norm value
				secBatchNorm = choice([randint(1,self.batchNormMaxValue),None])

				if secBatchNorm:
					self.numLearnParams+=1


				# Activation function value
				secActFunction= choice([*self.actFuncPossValues,None])

				if secActFunction:
					self.numLearnParams+=1


				# Dropout value
				secDropout = choice ([random(),None])

				if secDropout:
					self.numLearnParams+=1


				# Creating the dfc layer
				self.dfcLayer= DFCLayer( lastLayerOutCh * (self.inputDimension**2) ,10,fhNumOfNeurons,fhBatchNorm,fhActFunction,fhDropout,secNumOfNeurons,secBatchNorm,secActFunction,secDropout).create_dfc_layer().to(self.device)

	

class KNasEA:

	'''
		This class has implemented the evolutionary methods used in 
		developing the KNAS program
	'''

	def __init__(self,datasetModule,knasParams):

		# Population size
		self.popSize=10

		# Number of generations
		self.genNum=20


		# Device to be used
		self.device = knasParams["DEVICE"]


		# Maximum number of CN layers in an individual
		self.maxCNLayers=knasParams['MAX_CN_LAYERS']


		# Logging module handler
		self.logModHand= KNasEALogging()

		# KNas model handler
		self.knasModelHand = KNasModel(datasetModule,knasParams)

		
		# Calling the dataset set up function
		self.knasModelHand.knas_setup_dataset()
		

		




	def generate_initial_population(self):
		'''
			This function generates the initial population
		'''
		self.logModHand.knasea_log_message(self.logModHand.loggingCodes['INIT_POPULATION_GEN'],'INF')

		# List of individuals as population
		population=list()

		for i in range(self.popSize):
			population.append(KNasEAIndividual(self.maxCNLayers,self.device))


		return population





	def calculate_fitness(self,population):

		'''
			This function will calculate the fitness value of the individuals
			in the population
		'''
		for ind in population:

			performanceStatus = self.knasModelHand.knas_create_eval_model(ind.cnLayersList,ind.dfcLayer,ind.learningRate)

			# Calcualting the fitness value based on the performance status
			

			print("HI")
			print(performanceStatus)
			print(ind.numLearnParams)
			exit(0)
			# Updating the individuals




