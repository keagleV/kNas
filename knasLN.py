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
from torch.nn import BatchNorm1d

from random import randint
from random import choice
from random import random


class CNLayer():

	'''
		This class has implemented the CN layer of the network
	'''

	def __init__(self,in_channels,out_channels,
						kernel_size,stride,padding,batchNorm,
						activationFunc,dropoutProb,maxPoolKernelSize,knasParams=None):

		self.in_channels= in_channels
		
		self.out_channels= out_channels
		
		self.kernel_size=kernel_size
		
		self.stride=stride
		
		self.padding=padding

		self.batchNorm=batchNorm

		self.activationFunc = activationFunc

		self.dropoutProb = dropoutProb

		self.maxPoolKernelSize = maxPoolKernelSize

		self.knasParams = knasParams


	def create_random_cn_layer(self):
		'''
			This function will create a random cn layer
		'''

		# Setting parameter of a cn layer randomly

		self.in_channels= 1
		
		self.out_channels= choice(self.knasParams['FILTER_POSS_VALUES'])
		
		self.kernel_size= self.knasParams['KERNEL_SIZE']
		
		self.stride= self.knasParams['STRIDE']
		
		self.padding= self.knasParams['PADDING']

		self.batchNorm= choice([True , False]) 

		self.activationFunc = choice(["relu", 'sigmoid',None])

		self.dropoutProb = choice([random(),None])

		self.maxPoolKernelSize = choice([self.knasParams['KERNEL_SIZE'],None])



		# Calling create_cn_layer to create a cn layer with random parameters just set
		return self.create_cn_layer()
	


	def create_cn_layer(self):

		'''
			this function creates a CN layer based on the parameters given in 
			the constructor of the class
		'''

		# List of modules to be added to the layer
		listOfModules = []

		# Number of learnable parameters
		numLearnParams=0

		# Adding the first component which is convolutional filters
		listOfModules.append(Conv2d(in_channels=self.in_channels, 
							out_channels=self.out_channels,
							kernel_size=self.kernel_size,
							stride=self.stride,
							padding=self.padding)
							)

		# Check for the batchnorm
		if self.batchNorm!=None:
			numLearnParams+=1
			listOfModules.append(BatchNorm2d(self.out_channels))


		# Check for the activation function
		if self.activationFunc:

			numLearnParams+=1
			#TODO check for sth other than relu, sigmoid

			if self.activationFunc == "relu":
				listOfModules.append(ReLU())

			elif self.activationFunc == "sigmoid":
				listOfModules.append(Sigmoid())


		# Check for the dropout
		if self.dropoutProb:

			numLearnParams+=1

			# Adding the droupout
			listOfModules.append(Dropout(self.dropoutProb))


		if self.maxPoolKernelSize:

			numLearnParams+=1

			listOfModules.append(MaxPool2d(kernel_size=(self.maxPoolKernelSize,self.maxPoolKernelSize)))


		# Returning the sequence of the modules in the layer
		return (numLearnParams, Sequential(*listOfModules).to(self.knasParams["DEVICE"]))






class DFCLayer():
	'''
		This class has implemented the dense fully connected layer
	'''
	def __init__(self,in_channels,out_channels,fhiNeurons,fhiBatchNorm,
						fhiActivationFunc,fhiDropoutProb,secNeurons,secBatchNorm,
						secActivationFunc,secDropoutProb,knasParams=None):

		self.in_channels= in_channels
		
		self.out_channels= out_channels
		
		# First hidden layer parameters
		self.fhiNeurons=fhiNeurons

		self.fhiBatchNorm=fhiBatchNorm

		self.fhiActivationFunc = fhiActivationFunc

		self.fhiDropoutProb = fhiDropoutProb

		# Second hidden layer parameters
		self.secNeurons=secNeurons

		self.secBatchNorm=secBatchNorm

		self.secActivationFunc = secActivationFunc

		self.secDropoutProb = secDropoutProb


		# KNAS parameters for creating random layer
		self.knasParams = knasParams


	def create_random_dfc_layer(self):
		'''
			This function will create a random dfc layer
		'''
		
		# Setting parameter of a dfc layer randomly

		self.in_channels= 1
		
		self.out_channels= choice(self.knasParams['HIDDEN_LAYERS_NEURONS_POSS_VALUE'])

		self.fhiNeurons= choice(self.knasParams['HIDDEN_LAYERS_NEURONS_POSS_VALUE']+[None])

		self.fhiBatchNorm= choice([True , False]) 

		self.fhiActivationFunc = choice(["relu", 'sigmoid',None])

		self.fhiDropoutProb = choice([random(),None])



		self.secNeurons= choice(self.knasParams['HIDDEN_LAYERS_NEURONS_POSS_VALUE']+[None])

		self.secBatchNorm= choice([True , False]) 

		self.secActivationFunc = choice(["relu", 'sigmoid',None])

		self.secDropoutProb = choice([random(),None])



		# Calling create_dfc_layer to create a dfc layer with random parameters just set

		return self.create_dfc_layer()


	def create_dfc_layer(self):
		'''
			this function creates a dfc layer based on the parameters given in 
			the constructor of the class
		'''

		# List of modules to be added to the layer
		listOfModules = []	
	

		# Number of learnable parameters
		numLearnParams=0


		# Number of neurons in the last hidden layer. This will be used
		# to create the last layer of the dfc.
		lastHiddenLayerNeurons=0


		# Checking for the first hidden layer neurons
		if self.fhiNeurons:


			# Set the last layer neuron count
			lastHiddenLayerNeurons = self.fhiNeurons

			# Incrementing the learning parameters counter for number of 
			# neurons in the first hidden layer
			numLearnParams+=1

			# Adding first component of the first hidden layer which is Linear component
			listOfModules.append(Linear(self.in_channels,self.fhiNeurons))


			# Check for the batch norm
			if self.fhiBatchNorm!=None:
				numLearnParams+=1
				listOfModules.append(BatchNorm1d(self.fhiNeurons))

			# Check for the activation function
			if self.fhiActivationFunc:

				numLearnParams+=1

				if self.fhiActivationFunc == "relu":
					listOfModules.append(ReLU())

				elif self.fhiActivationFunc == "sigmoid":
					listOfModules.append(Sigmoid())

			# Check for the dropout
			if self.fhiDropoutProb:

				numLearnParams+=1

				# Adding the droupout
				listOfModules.append(Dropout(self.fhiDropoutProb))


			# Checking the second hidden layer
			if self.secNeurons:

				# Set the last layer neuron count
				lastHiddenLayerNeurons = self.secNeurons


				# Incrementing the learning parameters counter for number of 
				# neurons in the second hidden layer
				numLearnParams+=1

				# Adding first component of the second hidden layer which is Linear component
				listOfModules.append(Linear(self.fhiNeurons,self.secNeurons))

				# Check for the batch norm
				if self.secBatchNorm!=None:
					numLearnParams+=1
					listOfModules.append(BatchNorm1d(self.secNeurons))

				# Check for the activation function
				if self.secActivationFunc:

					numLearnParams+=1

					if self.secActivationFunc == "relu":
						listOfModules.append(ReLU())

					elif self.secActivationFunc == "sigmoid":
						listOfModules.append(Sigmoid())

				# Check for the dropout
				if self.secDropoutProb:

					numLearnParams+=1

					# Adding the droupout
					listOfModules.append(Dropout(self.secDropoutProb))



		# Check for the case with no hidden layers between
		if len(listOfModules)==0:

			# A dfc layer with no hidden layer, is just a simple Linear component
			listOfModules.append(Linear(self.in_channels,self.out_channels))

		else:
			# This is the last layer after the last hidden layer
			listOfModules.append(Linear(lastHiddenLayerNeurons,self.out_channels))


		# Returning the sequence of the modules in the layer
		return (numLearnParams,Sequential(*listOfModules).to(self.knasParams["DEVICE"]))




class KNasLayersNet(Module):
	'''
		This class has implemented the convultional layer of the KNAS program
	'''

	def __init__(self,cnnLayers,dfcLayer):
		# call the parent constructor
		super(KNasLayersNet, self).__init__()

		
		self.cnnLayers=cnnLayers

		self.dfcLayer=dfcLayer

		
	def forward(self, x):

		# Forward the data through the CN layers
		for cnnLayer in self.cnnLayers:
			x=cnnLayer(x)


		x= x.view(x.size(0),-1)


		self.dfcLayer(x)

		# Forward the data through the last layer, the dfc layer
		return self.dfcLayer(x)

	