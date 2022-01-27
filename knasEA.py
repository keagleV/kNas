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
			
			'''
			 In the case of having convultional layer, 
			 the output dimentsion would be:
			
			 						(W-F+2p)/s + 1
				 W: dim size
				 F: kernel size
				 P: padding
				 S: stride
				
			'''
			self.inputDimension = ( (self.inputDimension - 2 + 2)//1 ) +1

			# Batch norm value
			batchNorm = choice([randint(1,self.batchNormMaxValue),None])

			# Activation function value
			actFunction= choice([ *self.actFuncPossValues , None ])
			
			# Dropout value
			dropout = choice ([random(),None])

			# Maxpool value
			maxPool = choice([ 2, None ])

			# If we have maxpooling , we divide the dimension
			if maxPool:
				self.inputDimension = (self.inputDimension // 2 )


			# Adding a new CN layer to the previous layers
			numLearnParams,cnLayer = CNLayer(lastLayerOutCh,currLaFilterCnt,2,1,1,batchNorm,actFunction,dropout,maxPool).create_cn_layer()
			
			# Adding the learnable parameters count
			self.numLearnParams += numLearnParams
			
			self.cnLayersList.append(cnLayer.to(self.device))
			
			# Updating the last layer output channel
			lastLayerOutCh = currLaFilterCnt
			



		# Creating DFC layer

		# First define number of the hidden layers
		numOfHiddenLayers = choice([0,1,2])


		if numOfHiddenLayers==0:
			numLearnParams,self.dfcLayer= DFCLayer( lastLayerOutCh * (self.inputDimension**2)  ,10,None,None,None,None,None,None,None,None).create_dfc_layer()
			self.dfcLayer = self.dfcLayer.to(self.device)
			self.numLearnParams+=numLearnParams
		else:

			# We have to create at least 1 layer until this moment


			fhNumOfNeurons = choice(self.dfcHiLaPossNeuronValues)

			# Increment learnable params for the number of first hidden layer neurons
			self.numLearnParams+=1


			# Batch norm value
			fhBatchNorm = choice([randint(1,self.batchNormMaxValue),None])


			# Activation function value
			fhActFunction= choice([ *self.actFuncPossValues,None])
			

			# Dropout value
			fhDropout = choice ([random(),None])


			# We only have one hidden layer
			if numOfHiddenLayers==1:
				numLearnParams,self.dfcLayer= DFCLayer(lastLayerOutCh * (self.inputDimension**2)  ,10,fhNumOfNeurons,fhBatchNorm,fhActFunction,fhDropout,None,None,None,None).create_dfc_layer()
				self.dfcLayer = self.dfcLayer.to(self.device)
				self.numLearnParams += numLearnParams
			else:

				# Creating parameters for the second layer
				secNumOfNeurons = choice(self.dfcHiLaPossNeuronValues)

				# Increment learnable params for the number of first hidden layer neurons
				self.numLearnParams+=1

				# Batch norm value
				secBatchNorm = choice([randint(1,self.batchNormMaxValue),None])


				# Activation function value
				secActFunction= choice([*self.actFuncPossValues,None])


				# Dropout value
				secDropout = choice ([random(),None])


				# Creating the dfc layer
				numLearnParams,self.dfcLayer= DFCLayer( lastLayerOutCh * (self.inputDimension**2) ,10,fhNumOfNeurons,fhBatchNorm,fhActFunction,fhDropout,secNumOfNeurons,secBatchNorm,secActFunction,secDropout).create_dfc_layer()
				self.dfcLayer = self.dfcLayer.to(self.device)
				self.numLearnParams += numLearnParams
	

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

		# Crossover probability
		self.crossProb= 0.4

		# Mutation probability
		self.mutProb = 0.1

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
		

		




	def knasea_generate_initial_population(self):
		'''
			This function generates the initial population
		'''
		self.logModHand.knasea_log_message(self.logModHand.loggingCodes['INIT_POPULATION_GEN'],'INF')

		# List of individuals as population
		population=list()

		for i in range(self.popSize):
			population.append(KNasEAIndividual(self.maxCNLayers,self.device))


		return population





	def knasea_calculate_fitness(self,population):

		'''
			This function will calculate the fitness value of the individuals
			in the population
		'''
		for ind in population:

			performanceStatus = self.knasModelHand.knas_create_eval_model(ind.cnLayersList,ind.dfcLayer,ind.learningRate)

			# Calcualting the fitness value based on the performance status
			
			# average of epochs
			print("HI")
			print(performanceStatus)
			print(ind.numLearnParams)

			ind.fitnessVal=randint(1,100)

			# Updating the individuals


		return population

	def knasea_crossover(self,ind1 , ind2 ):
		'''
			This function will perform the crossover method on two 
			individuals selected from the population
		'''

		return ind1,ind2
	
	
	def knasea_mutation(self,ind1 , ind2 ):
		'''
			This function will perform the mutation method on two 
			offsprings created from the crossover phase
		'''

		return ind1,ind2




