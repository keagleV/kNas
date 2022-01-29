from knasLN import CNLayer
from knasLN import DFCLayer
from knasModel import KNasModel


from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import LogSoftmax
from torch.nn import Sequential
from torch.nn import Dropout
from torch.nn import Flatten
from torch.nn import BatchNorm2d
from random import randint
from random import choice
from random import choices
from random import random
from statistics import mean


class KNasEALogging():
	'''
		This class has implemented the logging operations for the EA
		algorithm used in the KNAS program
	'''
	def __init__(self):

		self.loggingCodes = {
			
			'INIT_POPULATION_GEN':'Generating The Initial Population ...',

		
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
	def __init__(self,knasParams):

		# Fitness value of the individual
		self.fitnessVal=int()

		# Performance status parameters
		self.performanceStatus = None

		# Number of learnable parameters of this individual
		self.numLearnParams = int()

		# Number of CN layers in the first part of the network
		self.cnLayersCount=int()

		# List of CN layers in the first part of the network
		self.cnLayersList=list()

		# DFC layer, the only layer in the last part of the network
		self.dfcLayer=None

		# Device to be used to create the layers of this individual
		self.device = knasParams['DEVICE']

		# Learning rate
		self.learningRate = float()

		# Input dimenesion: 1 * dim * dim 
		self.inputDimension = knasParams['INPUT_DIM']

		# Possible values for the filter count
		self.filterPossValues = knasParams['FILTER_POSS_VALUES']

		# Possible values for the batch norm
		self.batchNormMaxValue = 100

		# Possible values for the activation function
		self.actFuncPossValues = ['relu','sigmoid']

		# Possible values for the dfc hidden layer neurons
		self.dfcHiLaPossNeuronValues = knasParams['HIDDEN_LAYERS_NEURONS_POSS_VALUE']


		self.create_random_individual(knasParams['MAX_CNN'],knasParams['KERNEL_SIZE'],knasParams['PADDING'],knasParams['STRIDE'])


	# def weight_reset(self,m):
	# 	if isinstance(m, Conv2d) or isinstance(m, Linear):
	# 		m.reset_parameters()



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
			inch=0
			och=0
			batchNorm=None

			if i == 0:
				inch = 1
				och = list(l)[0].out_channels

			else:
				
				# Update input channels based on the previous layer output channel
				
				inch = list(self.cnLayersList[i-1])[0].out_channels
				och = list(l)[0].in_channels

				# Checking for batchnorm, check the length first since
				# the layer can have only conv2d
				if len(list(l))>1 and isinstance(list(l)[1],BatchNorm2d):
					batchNorm=BatchNorm2d(och)
				

			# List of final layers
			finalComps = [Conv2d(in_channels=inch, 
							out_channels=och,
							kernel_size=2,
							stride=1,
							padding=1)
							]

			if batchNorm:
				finalComps.append(batchNorm)
				finalComps+=list(l)[2:]
			else:
				finalComps+=list(l)[1:]


			self.cnLayersList[i]= Sequential(*finalComps).to(self.device)

			# Updating the last layer output channel
			lastLayerOutCh = list(self.cnLayersList[i])[0].out_channels


		# Setting the dfc layer's input channel
		inch= lastLayerOutCh * (inputDimension**2)
		och= list(self.dfcLayer)[0].out_features
		self.dfcLayer = Sequential( *([Linear(inch,och)] + list(self.dfcLayer[1:] )) ).to(self.device)
		

		# self.dfcLayer.apply(self.weight_reset)
		# list(self.dfcLayer)[0].in_features = lastLayerOutCh * (inputDimension**2)



	def create_random_individual(self,maxCNLayers , kernelSize , padding , stride):
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
			
			inputDimension = ( (inputDimension - kernelSize + 2*padding )// stride ) + 1

			# Batch norm value
			batchNorm = choice([randint(1,self.batchNormMaxValue),None])

			# Activation function value
			actFunction= choice([ *self.actFuncPossValues , None ])
			
			# Dropout value
			dropout = choice ([random(),None])

			# Maxpool value
			maxPool = choice([ kernelSize , None ])

			# If we have maxpooling , we divide the dimension
			if maxPool:
				inputDimension = inputDimension // kernelSize 


			# Adding a new CN layer to the previous layers
			numLearnParams,cnLayer = CNLayer(lastLayerOutCh,currLaFilterCnt,kernelSize,padding,stride,batchNorm,actFunction,dropout,maxPool).create_cn_layer()
		
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
		self.popSize= knasParams['POP_SIZE']

		# Number of generations
		self.genNum= knasParams['GEN_NUM']

		# Crossover probability
		self.crossProb= knasParams['CROSS_PROB']

		# Mutation probability
		self.mutProb = knasParams['MUT_PROB']

		# Mutation opearations probabilities
		self.mutAddProb = knasParams['MUT_ADD_PROB']
		self.mutModProb = knasParams['MUT_MOD_PROB']
		self.mutRemProb = knasParams['MUT_REM_PROB']

		# Mutation add operations
		self.mutAddCnLayer = 12
		self.mutAddBatchNorm = knasParams['MUT_ADD_BATCHNORM_PROB']
		self.mutAddAcFunc = knasParams['MUT_ADD_ACTFUNC_PROB']
		self.mutAddDropout = knasParams['MUT_ADD_DROPOUT_PROB']
		self.mutAddMaxPool = knasParams['MUT_ADD_MAXPOOL_PROB']

		# Mutation modify operations
		self.mutModAcFunc = knasParams['MUT_MOD_ACTFUNC_PROB']
		self.mutModDropout = knasParams['MUT_MOD_DROPOUT_PROB']
		self.mutModFilters = knasParams['MUT_MOD_FILTERS_PROB']


		# Mutation modify operations
		self.mutRemCnLayer = knasParams['MUT_REM_CNLAYER_PROB']
		self.mutRemBatchNorm = knasParams['MUT_REM_BATCHNORM_PROB']
		self.mutRemAcFunc = knasParams['MUT_REM_ACTFUNC_PROB']
		self.mutRemDropout = knasParams['MUT_REM_DROPOUT_PROB']
		self.mutRemMaxPool= knasParams['MUT_REM_MAXPOOL_PROB']


		# KNAS parameters for later usage
		self.knasParams = knasParams

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
			population.append(KNasEAIndividual(self.knasParams))

		return population





	def knasea_calculate_fitness(self,population):

		'''
			This function will calculate the fitness value of the individuals
			in the population
		'''
		for ind in population:

			performanceStatus = self.knasModelHand.knas_create_eval_model(ind.cnLayersList,ind.dfcLayer,ind.learningRate)

			# Setting the performance status of the individual
			ind.performanceStatus = performanceStatus
			

			# Calcualting the fitness value based on the performance status

			# Calculating the mean values of epochs
			meanTrainDuration = mean(performanceStatus['training_time'])

			meanValAcc= mean(performanceStatus['val_acc'])

			meanTestAcc= mean(performanceStatus['test_acc'])

			# Setting the fitness value
			ind.fitnessVal = meanTestAcc + 1/(abs(meanValAcc - meanTestAcc )+1) + 1/meanTrainDuration + 1/ind.numLearnParams



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
	
	

	def knasea_mutation(self,ind):
		'''
			This function will perform the mutation method on the 
			offspring created from the crossover phase
		'''

		# List of new layers after mutation
		newLayers = []


		for l in ind.cnLayersList:

			listl = list(l)


			if random() < self.mutProb:

				# Decide on the type of operation
				op = choices(["add","mod","rem"],weights=[self.mutAddProb,self.mutModProb,self.mutRemProb],k=1)[0]
				

				if op=="add":

					# Decide on the type of add operation
					addOp =  choices(["addCnLayer","addBatch","addAf","addDr","addMax"],weights=[self.mutAddCnLayer,self.mutAddBatchNorm,self.mutAddAcFunc,self.mutAddDropout,self.mutAddMaxPool],k=1)[0]
					
					# Adding a new CN layer
					if (addOp=="addCnLayer"):

						# Check for the maximum number of layers constraint
						# Add only if there is space to add
						if len(ind.cnLayersList) < self.knasParams["MAX_CNN"]:

							# Creating a random CN layer, by only sending the
							# parameters of KNAS

							numLearnParams,newCnLayer = CNLayer(None,None,None,None,None,None,
											None,None,None,self.knasParams).create_random_cn_layer()

							# Adding number of learnable parameters of this new layer to the individual
							ind.numLearnParams += numLearnParams

							# Adding the new layer to the list
							newLayers.append(newCnLayer)


						# Setting the add operation for the layer we choosed in the iteration
						addOp =  choices(["addBatch","addAf","addDr","addMax"],weights=[self.mutAddBatchNorm,self.mutAddAcFunc,self.mutAddDropout,self.mutAddMaxPool],k=1)[0]



					# Possible components of the given cn layer
					components = [None]*5

					for com in listl:
						if isinstance(com,Conv2d):
							components[0] = com
						elif isinstance(com,BatchNorm2d):
							components[1] = com
						elif isinstance(com,ReLU):
							components[2] = com
						elif isinstance(com,Sigmoid):
							components[2] = com
						elif isinstance(com,Dropout):
							components[3] = com
						elif isinstance(com,MaxPool2d):
							components[4] = com
					
					# Adding batch norm only if it does not exist
					if ( addOp == "addBatch" ) and (components[1]==None):

						# Number of filters in conv2d componendt
						filterCount = components[0].out_channels
						
						components[1] = BatchNorm2d(filterCount)

					# Adding activation function only if it does not exist
					elif ( addOp == "addAf" ) and (components[2]==None):
						
						# Randomly choose an activation function
						components[2] = choice([ReLU(),Sigmoid()])

					# Adding dropout only if it does not exist
					elif ( addOp == "addDr") and (components[3]==None):
						
						# Adding a random dropout object
						components[3] = Dropout(random())
						
					# Adding maxpool only if it does not exist
					elif (addOp == "addMax") and (components[4]==None):
						
						# Adding a maxpool layer
						components[4] = MaxPool2d(kernel_size=(2,2))
					
					# Removing the None objects from the components list and
					# updating the listl list
					listl = [com for com in components if com!=None]









				elif op == "mod":

					# Decide on the type of modify operation
					modOp =  choices(["modAf","modDr","modFil"],weights=[self.mutModAcFunc,self.mutModDropout,self.mutModFilters],k=1)[0]

					if modOp == "modAf":

						# Switching the activation function instance in the sequential
						for i,com in enumerate(listl):
							if isinstance(com,Sigmoid):
								listl[i] = ReLU()
								break
							elif isinstance(com,ReLU):
								listl[i]= Sigmoid()
								break


					elif modOp == "modDr":

						# Modifying the dropout probability instance in the sequential
						for i,com in enumerate(listl):
							
							if isinstance(com,Dropout):
								
								# Creating new instance
								listl[i]= Dropout(random())
								
								break


					elif modOp == "modFil":

						# Number of filters to be set
						numFilters = choice(self.knasParams['FILTER_POSS_VALUES'])

						# Parameters of current conv2d
						inputChannels = listl[0].in_channels
						kernelSize = listl[0].kernel_size
						padding = listl[0].padding
						stride = listl[0].stride

						listl[0]= Conv2d(in_channels=inputChannels, 
							out_channels=numFilters,
							kernel_size= kernelSize ,
							stride=stride,
							padding=padding)
							




				elif op=="rem":

					# Decide on the type of remove operation
					remOp =  choices(["remCnLayer","remBatch","remAf","remDr","remMax"],weights=[self.mutRemCnLayer,self.mutRemBatchNorm,self.mutRemAcFunc,self.mutRemDropout,self.mutRemMaxPool],k=1)[0]

					if remOp == "remCnLayer":
						continue

					elif remOp == "remBatch":
						
						# Removing the BatchNorm2d instance in the sequential
						for com in listl:
							if isinstance(com,BatchNorm2d):
								listl.remove(com)
								break


					elif remOp == "remAf":

						# Removing the activation function instance in the sequential
						for com in listl:
							if isinstance(com,Sigmoid) or isinstance(com,ReLU):
								listl.remove(com)
								break
						

					elif remOp == "remDr":

						# Removing the dropout instance in the sequential
						for com in listl:
							if isinstance(com,Dropout):
								listl.remove(com)
								break

					elif remOp == "remMax":

						# Removing the maxpool instance in the sequential
						for com in listl:
							if isinstance(com,MaxPool2d):
								listl.remove(com)
								break

				newLayers.append(Sequential(*listl))


		# Updating the cn layers of the individual
		ind.cnLayersList = newLayers

		return ind




