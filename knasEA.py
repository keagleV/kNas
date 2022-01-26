

from knasLN import CNLayer
from knasLN import DFCLayer

from random import randint
from random import choice
from random import random


from knasModel import KNasModel



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
	def __init__(self,maxCNLayers):

		# Fitness value of the individual
		self.fitnessVal=int()

		# Number of CN layers in the first part of the network
		self.cnLayersCount=int()

		# List of CN layers in the first part of the network
		self.cnLayersList=list()

		# DFC layer, the only layer in the last part of the network
		self.dfcLayer=None

		# Learning rate
		self.learningRate = float()

		# Possible values for the filter count
		self.filterPossValues = [ 2, 4 , 8, 16, 32, 64, 128 ]

		# Possible values for the batch norm
		self.batchNormMaxValue = 100

		# Possible values for the activation function
		self.actFuncPossValues = ['relu','sigmoid',None]

		# Possible values for the activation function
		self.actFuncPossValues = ['relu','sigmoid',None]

		# Possible values for the dfc hidden layer neurons
		self.dfcHiLaPossNeuronValues = [ 2, 4 , 8, 16, 32, 64, 128 , 256 , 512 ]


		self.create_random_individual(maxCNLayers)



	def create_random_individual(self,maxCNLayers):
		'''
			This function creates a random indivi
		'''

		

		# Decide on number of the CN layers
		self.cnLayersCount = randint(1,maxCNLayers)


		# Creating CN Layers

		# Last layer output channels
		lastLayerOutCh=1
		for i in range(self.cnLayersCount):



			# Number of filters in this layer
			# currLaFilterCnt = choice(self.filterPossValues)
			currLaFilterCnt = lastLayerOutCh*2
			# Batch norm value
			batchNorm = choice([randint(1,self.batchNormMaxValue),None])

			# Activation function value
			actFunction= choice(self.actFuncPossValues)


			# Dropout value
			dropout = choice ([random(),None])

			# Maxpool value
			maxPool = 2

			self.cnLayersList.append(CNLayer(lastLayerOutCh,currLaFilterCnt,2,1,1,batchNorm,actFunction,dropout,maxPool).create_cn_layer())
			
			# Updating the last layer output channel
			lastLayerOutCh = currLaFilterCnt
			

		# Creating DFC layer
		#TODO



		# First define number of the hidden layers
		numOfHiddenLayers = choice([0,1,2])

		if numOfHiddenLayers==0:
			self.dfcLayer= DFCLayer( lastLayerOutCh ,10,None,None,None,None,None,None,None,None).create_dfc_layer()
		else:

			# We have to create at least 1 layer until this moment

			fhNumOfNeurons = choice(self.dfcHiLaPossNeuronValues)

			# Batch norm value
			fhBatchNorm = choice([randint(1,self.batchNormMaxValue),None])

			# Activation function value
			fhActFunction= choice(self.actFuncPossValues)

			# Dropout value
			fhDropout = choice ([random(),None])

			# We only have one hidden layer
			if numOfHiddenLayers==1:
				self.dfcLayer= DFCLayer(lastLayerOutCh ,10,fhNumOfNeurons,fhBatchNorm,fhActFunction,fhDropout,None,None,None,None).create_dfc_layer()

			else:

				# Creating parameters for the second layer
				secNumOfNeurons = choice(self.dfcHiLaPossNeuronValues)

				# Batch norm value
				secBatchNorm = choice([randint(1,self.batchNormMaxValue),None])

				# Activation function value
				secActFunction= choice(self.actFuncPossValues)

				# Dropout value
				secDropout = choice ([random(),None])

				self.dfcLayer= DFCLayer(lastLayerOutCh,10,fhNumOfNeurons,fhBatchNorm,fhActFunction,fhDropout,secNumOfNeurons,secBatchNorm,secActFunction,secDropout).create_dfc_layer()

	








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
			population.append(KNasEAIndividual(self.maxCNLayers))


		# for po in population:
		# 	print(po.cnLayersList)
		# 	print("--------")
		# 	print(po.dfcLayer)

		# 	print("XXXXXXXXXXXXXXXXXXX")

		return population





	def calculate_fitness(self,population):

		'''
			This function will calculate the fitness value of the individuals
			in the population
		'''
		for ind in population:

			performanceStatus = self.knasModelHand.knas_create_eval_model(ind.cnLayersList,ind.dfcLayer,0.2)


			# Updating the individuals






# obj=KNasEA(maxCNLayers=6)
# obj.generate_initial_population()


# obj=KNasEAIndividual(6)

# for la in obj.cnLayersList:
# 	print(la)
# # print(obj.cnLayersList)
# print("XXXXXXXX")
# print(obj.dfcLayer)



