from knasLN import CNLayer
from knasLN import DFCLayer
from knasModel import KNasModel
from torch.nn import ReLU
from torch.nn import MaxPool2d
from random import randint
from random import choice
from random import choices
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


	def make_individual_valid(self):
		'''
			This function makes the individual a valid individual
			by updating the output and input channels of the layers
			based on their previous layer
		'''	

		# Input dimension to calculate the output size of the cn layers
		inputDimension = self.inputDimension

		# Number of output channels in the last cnn layer
		lastLayerOutCh = 0 

		

		for i,l in enumerate(self.cnLayersList):

			# Updating the dimension after the cn layer
			inputDimension = ( (inputDimension - 2 + 2)//1 ) +1

			# Check for the maxpool layer, maxpool layer is the last component of 
			# the layer, i.e l[-1]
			if isinstance(l[-1],MaxPool2d):
				inputDimension = inputDimension // 2 


			
			# Updating the first layer input channel to 1, there is a possibility
			# and it does not equal 1

			if i == 0:
				list(l)[0].in_channels=1
				# print("0")
				# print(l)
			else:
				# Update input channels based on the previous layer output channel
				list(l)[0].in_channels = list(self.cnLayersList[i-1])[0].out_channels
				# print("non")
				# print(l)

			self.cnLayersList[i]=l

			# Updating the last layer output channel
			lastLayerOutCh = list(self.cnLayersList[i])[0].out_channels

		print("LIST")
		print(self.cnLayersList)
		print(lastLayerOutCh * (inputDimension**2))
		# Setting the dfc layer's input channel
		list(self.dfcLayer)[0].in_features = lastLayerOutCh * (inputDimension**2)



	def create_random_individual(self,maxCNLayers):
		'''
			This function creates a random indivi
		'''

		# Input dimension to calculate the output size
		inputDimension = self.inputDimension


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
			 In the case of having convolutional layer, 
			 the output dimension would be:
			
			 						(W-F+2p)/s + 1
				 W: dim size
				 F: kernel size
				 P: padding
				 S: stride
				
			'''
			
			inputDimension = ( (inputDimension - 2 + 2)//1 ) +1

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
				inputDimension = inputDimension // 2 


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
			numLearnParams,self.dfcLayer= DFCLayer( lastLayerOutCh * (inputDimension**2)  ,10,None,None,None,None,None,None,None,None).create_dfc_layer()
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
				numLearnParams,self.dfcLayer= DFCLayer(lastLayerOutCh * (inputDimension**2)  ,10,fhNumOfNeurons,fhBatchNorm,fhActFunction,fhDropout,None,None,None,None).create_dfc_layer()
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
				numLearnParams,self.dfcLayer= DFCLayer( lastLayerOutCh * (inputDimension**2) ,10,fhNumOfNeurons,fhBatchNorm,fhActFunction,fhDropout,secNumOfNeurons,secBatchNorm,secActFunction,secDropout).create_dfc_layer()
				self.dfcLayer = self.dfcLayer.to(self.device)
				self.numLearnParams += numLearnParams
	



class KNasEA:

	'''
		This class has implemented the evolutionary methods used in 
		developing the KNAS program
	'''

	def __init__(self,datasetModule,knasParams):

		# Population size
		self.popSize=2

		# Number of generations
		self.genNum= 1

		# Crossover probability
		self.crossProb= 12

		# Mutation probability
		self.mutProb = 12

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

		# Layers of the network in the individuals
		ind1Layers = [ *ind1.cnLayersList ]
		
		ind2Layers = [ *ind2.cnLayersList ]


		# Number of layers in each individual
		numLayersInd1 = len(ind1Layers)
		numLayersInd2 = len(ind2Layers)

		# Setting the individual 1 as the individual 
		# which has lowest number of layers
		if numLayersInd2 < numLayersInd1:

			# Switch individuals
			ind1,ind2 = ind2,ind1
			ind1Layers,ind2Layers = ind2Layers,ind1Layers
			numLayersInd1,numLayersInd2 =  numLayersInd2,numLayersInd1

		# Offsprings
		off1 = None
		off2 = None


		if random() <= self.crossProb:


			# Chossing crossover points in the first individual
			point11,point12 = tuple(sorted(choices(range(numLayersInd1+1),k=2)))

			# The portion has index: from point11 to (point12)+1, 
			# so we have to convert point12 to a index value by
			# subtracting 1 from it: 
			point12-=1


			# Chossing crossover points in the second individual, for this:
			# Since the length of this portion should be same as the first
			# individual, we first choose a random number and then randomly 
			# choose its right side or left side to create a portion with the
			# size we want.
			sizeOfThePortion = point12 - point11 + 1 


			point21 = choice(range(numLayersInd2+1))
			point22 = point21
			
			if point21 - sizeOfThePortion < 0 :
				# Chossing the right side
				point22 += sizeOfThePortion
			elif point21 + sizeOfThePortion > numLayersInd2+1:
				# Chossing the left side
				point21 -= sizeOfThePortion

			else:
				# Chossing the side randomly,
				side = choice(["right","left"])

				if side == "right":
					point22 += sizeOfThePortion
				else:
					point21 -= sizeOfThePortion

			# Converting the point22 to index
			point22 -=1 

			# Creating the offsprings

			off1 = ind1Layers[:point11] + ind2Layers[point21:point22+1] + ind1Layers[point12+1:]

			off2 = ind2Layers[:point21] + ind1Layers[point11:point12+1] + ind2Layers[point22+1:]
	

			# Swapping dfc layers with 50% probability
			if random() <= 0.5:
				ind1.dfcLayer,ind2.dfcLayer=ind2.dfcLayer,ind1.dfcLayer

			# Updating the individuals and making them valid
			ind1.cnLayersList = off1
			ind1.make_individual_valid()


			ind2.cnLayersList = off2
			ind1.make_individual_valid()


		return ind1,ind2
	
	
	def knasea_mutation(self,ind1 , ind2 ):
		'''
			This function will perform the mutation method on two 
			offsprings created from the crossover phase
		'''

		return ind1,ind2




