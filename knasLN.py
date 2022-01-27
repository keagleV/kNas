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




class CNLayer():

	'''
		This class has implemented the CN layer of the network
	'''

	def __init__(self,in_channels,out_channels,
						kernel_size,stride,padding,batchNorm,
						activationFunc,dropoutProb,maxPoolKernelSize):

		self.in_channels= in_channels
		
		self.out_channels= out_channels
		
		self.kernel_size=kernel_size
		
		self.stride=stride
		
		self.padding=padding

		self.batchNorm=batchNorm

		self.activationFunc = activationFunc

		self.dropoutProb = dropoutProb

		self.maxPoolKernelSize = maxPoolKernelSize


	
	def create_cn_layer(self):

		'''
			this function creates a CN layer based on the parameters given in 
			the constructor of the class
		'''

		# List of modules to be added to the layer
		listOfModules = []


		# Adding the first component which is convolutional filters
		listOfModules.append(Conv2d(in_channels=self.in_channels, 
							out_channels=self.out_channels,
							kernel_size=self.kernel_size,
							stride=self.stride,
							padding=self.padding)
							)

		# Check for the batchnorm
		# if self.batchNorm!=None:
		# 	listOfModules.append(BatchNorm2d(self.batchNorm))

		# Check for the activation function

		if self.activationFunc:
			if self.activationFunc == "relu":
				listOfModules.append(ReLU())

			elif self.activationFunc == "sigmoid":
				listOfModules.append(Sigmoid())

		# Check for the dropout
		if self.dropoutProb:
			# Adding the droupout
			listOfModules.append(Dropout(self.dropoutProb))


		if self.maxPoolKernelSize:
			listOfModules.append(MaxPool2d(kernel_size=(self.maxPoolKernelSize,self.maxPoolKernelSize)))

		# Returning the sequence of the modules in the layer
		return Sequential(*listOfModules)


class DFCLayer():
	'''
		This class has implemented the dense fully connected layer
	'''
	def __init__(self,in_channels,out_channels,fhiNeurons,fhiBatchNorm,
						fhiActivationFunc,fhiDropoutProb,secNeurons,secBatchNorm,
						secActivationFunc,secDropoutProb):

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



	def create_dfc_layer(self):
		'''
			this function creates a dfc layer based on the parameters given in 
			the constructor of the class
		'''

		# List of modules to be added to the layer
		listOfModules = []	



		# Number of neurons in the last hidden layer. This will be used
		# to create the last layer of the dfc.
		lastHiddenLayerNeurons=0


		# Checking for the first hidden layer neurons

		if self.fhiNeurons:

			# Set the last layer neuron count
			lastHiddenLayerNeurons = self.fhiNeurons

			listOfModules.append(Linear(self.in_channels,self.fhiNeurons))

			# Check for the batch norm
			# if self.fhiBatchNorm!=None:
			# 	listOfModules.append(BatchNorm2d(self.fhiBatchNorm))

			# Check for the activation function
			if self.fhiActivationFunc:
				if self.fhiActivationFunc == "relu":
					listOfModules.append(ReLU())

				elif self.fhiActivationFunc == "sigmoid":
					listOfModules.append(Sigmoid())

			# Check for the dropout
			if self.fhiDropoutProb:
				# Adding the droupout
				listOfModules.append(Dropout(self.fhiDropoutProb))


			# Checking the second hidden layer

			if self.secNeurons:

				# Set the last layer neuron count
				lastHiddenLayerNeurons = self.secNeurons

				listOfModules.append(Linear(self.fhiNeurons,self.secNeurons))

				# Check for the batch norm
				# if self.secBatchNorm!=None:
				# 	listOfModules.append(BatchNorm2d(self.secBatchNorm))

				# Check for the activation function
				if self.secActivationFunc:
					if self.secActivationFunc == "relu":
						listOfModules.append(ReLU())

					elif self.secActivationFunc == "sigmoid":
						listOfModules.append(Sigmoid())

				# Check for the dropout
				if self.secDropoutProb:
					# Adding the droupout
					listOfModules.append(Dropout(self.secDropoutProb))



		# Check for the case with no hidden layers between
		if len(listOfModules)==0:
			listOfModules.append(Linear(self.in_channels,self.out_channels))

		else:
			# This is the last layer after the last hidden layer
			listOfModules.append(Linear(lastHiddenLayerNeurons,self.out_channels))


		# Returning the sequence of the modules in the layer
		return Sequential(*listOfModules)


class KNasLayersNet(Module):
	'''
		This class has implemented the convultional layer of the KNAS program
	'''

	def __init__(self, numChannels,cnnLayers,dfcLayer):
		# call the parent constructor
		super(KNasLayersNet, self).__init__()

		
		self.cnnLayers=cnnLayers

		self.dfcLayer=dfcLayer

		
	def forward(self, x):

		# Forward the data through the CN layers

		for cnnLayer in self.cnnLayers:
			x=cnnLayer(x)


		x= x.view(x.size(0),-1)


		# Forward the data through the last layer, the dfc layer
		return self.dfcLayer(x)

	