from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import LogSoftmax
from torch.nn import Sequential
from torch.nn import Dropout

import torch
# #??س
# from torch.nn import BatchNorm1d
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

		if self.activationFunc!=None:
			if self.activationFunc == "relu":
				listOfModules.append(ReLU())

			elif self.activationFunc == "sigmoid":
				listOfModules.append(Sigmoid())

		# Check for the dropout
		if self.dropoutProb!=None:
			# Adding the droupout
			listOfModules.append(Dropout(self.dropoutProb))


		if self.maxPoolKernelSize!=None:
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

		if self.fhiNeurons!=None:

			# Set the last layer neuron count
			lastHiddenLayerNeurons = self.fhiNeurons

			listOfModules.append(Linear(self.in_channels,self.fhiNeurons))

			# Check for the batch norm
			# if self.fhiBatchNorm!=None:
			# 	listOfModules.append(BatchNorm2d(self.fhiBatchNorm))

			# Check for the activation function
			if self.fhiActivationFunc!=None:
				if self.fhiActivationFunc == "relu":
					listOfModules.append(ReLU())

				elif self.fhiActivationFunc == "sigmoid":
					listOfModules.append(Sigmoid())

			# Check for the dropout
			if self.fhiDropoutProb!=None:
				# Adding the droupout
				listOfModules.append(Dropout(self.fhiDropoutProb))


			# Checking the second hidden layer

			if self.secNeurons!=None:

				# Set the last layer neuron count
				lastHiddenLayerNeurons = self.secNeurons

				listOfModules.append(Linear(self.fhiNeurons,self.secNeurons))

				# Check for the batch norm
				# if self.secBatchNorm!=None:
				# 	listOfModules.append(BatchNorm2d(self.secBatchNorm))

				# Check for the activation function
				if self.secActivationFunc!=None:
					if self.secActivationFunc == "relu":
						listOfModules.append(ReLU())

					elif self.secActivationFunc == "sigmoid":
						listOfModules.append(Sigmoid())

				# Check for the dropout
				if self.secDropoutProb!=None:
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

		# print("HEr")
		# print(cnnLayers)
		# print(dfcLayer)



		# initialize first set of CONV => RELU => POOL layers
		# self.conv1 = Sequential(
		# 				Conv2d(in_channels=numChannels, 
		# 					out_channels=16,
		# 					kernel_size=2,
		# 					stride=1,
		# 					padding=1),
		# 				ReLU(),
		# 				MaxPool2d(kernel_size=2))


		# self.conv2 = Sequential(
		# 				Conv2d(in_channels=16, 
		# 					out_channels=32,
		# 					kernel_size=2,
		# 					stride=1,
		# 					padding=1),
		# 				ReLU(),
		# 				MaxPool2d(kernel_size=2))


		# self.conv3 = Sequential(
		# 				Conv2d(in_channels=32, 
		# 					out_channels=64,
		# 					kernel_size=2,
		# 					stride=1,
		# 					padding=1),
		# 				ReLU(),
		# 				MaxPool2d(kernel_size=2))



		# self.relu1 = ReLU()
		# self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		
		'''
			 def __init__(self):
			      super(NeuralNetwork, self).__init__()
			      self.flatten = nn.Flatten()
			      self.linear_relu_stack = nn.Sequential(
			      nn.Linear(28*28, 512),
			      nn.ReLU(),
			      nn.Linear(512, 512),
			      nn.ReLU(),
			      nn.Linear(512, 10),
			      nn.ReLU()
			  )

		'''

		# Dense/Fully Connected layer
		# self.dfc= Sequential (


		# 		# Input layer to the first hidden layer
		# 		Linear(64*4*4,512),
		# 		ReLU(),
				
		# 		# First hidden layer to the second hidden layer
		# 		Linear(512,128),
		# 		ReLU(),
				

		# 		# Last layer which is the output layer
		# 		Linear(128,10),

			# )

		# self.out=Linear(64*4*4,10)


		# # initialize second set of CONV => RELU => POOL layers
		# self.conv2 = Conv2d(in_channels=20, out_channels=50,
		# 	kernel_size=(5, 5))
		# self.relu2 = ReLU()
		# self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# # initialize first (and only) set of FC => RELU layers
		# self.fc1 = Linear(in_features=800, out_features=500)
		# self.relu3 = ReLU()
		# # initialize our softmax classifier
		# self.fc2 = Linear(in_features=500, out_features=classes)
		# self.logSoftmax = LogSoftmax(dim=1)


	def forward(self, x):

		for cnnLayer in self.cnnLayers:
			x=cnnLayer(x)

		x= x.view(x.size(0),-1)

		return self.dfcLayer(x)

		# x=self.conv1(x)

		# x=self.conv2(x)





		# x=self.conv3(self.conv2(self.conv1(x)))

		# x= x.view(x.size(0),-1)

		# return self.dfc(x)
		





		# # pass the input through our first set of CONV => RELU =>
		# # POOL layers
		# x = self.conv1(x)
		# x = self.relu1(x)
		# x = self.maxpool1(x)
		# # pass the output from the previous layer through the second
		# # set of CONV => RELU => POOL layers
		# x = self.conv2(x)
		# x = self.relu2(x)
		# x = self.maxpool2(x)
		# # flatten the output from the previous layer and pass it
		# # through our only set of FC => RELU layers
		# x = flatten(x, 1)
		# x = self.fc1(x)
		# x = self.relu3(x)
		# # pass the output to our softmax classifier to get our output
		# # predictions
		# x = self.fc2(x)
		# output = self.logSoftmax(x)
		# # return the output predictions
		# return output




# in_channels,out_channels,
#kernel_size,stride,padding,batchNorm,
#activationFunc,dropoutProb,maxPool



# x=CNLayer(1,16,2,1,1,None,None,None,None)

# x.create_cn_layer()


# in_channels,out_channels,fhiNeurons,fhiBatchNorm,
# 						fhiActivationFunc,fhiDropoutProb,secNeurons,secBatchNorm,
# 						secActivationFunc,secDropoutProb)




# x=DFCLayer(64*4*4,10,64,None,"relu",None,32,None,None,None)
# print(x.create_dfc_layer()
# )





# from torch.nn import Module
# from torch.nn import Conv2d
# from torch.nn import Linear
# from torch.nn import MaxPool2d
# from torch.nn import ReLU
# from torch.nn import LogSoftmax
# from torch import flatten



# class KNasCNN(Module):
# 	'''
# 		This class has implemented the convultional layer of the KNAS program
# 	'''

# 	def __init__(self, numChannels, classes):
# 		# call the parent constructor
# 		super(KNasCNN, self).__init__()
# 		# initialize first set of CONV => RELU => POOL layers
# 		self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
# 			kernel_size=(5, 5))
# 		self.relu1 = ReLU()
# 		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		

# 		# initialize second set of CONV => RELU => POOL layers
# 		self.conv2 = Conv2d(in_channels=20, out_channels=50,
# 			kernel_size=(5, 5))
# 		self.relu2 = ReLU()
# 		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
# 		# initialize first (and only) set of FC => RELU layers
# 		self.fc1 = Linear(in_features=800, out_features=500)
# 		self.relu3 = ReLU()
# 		# initialize our softmax classifier
# 		self.fc2 = Linear(in_features=500, out_features=classes)
# 		self.logSoftmax = LogSoftmax(dim=1)


# 	def forward(self, x):
# 		# pass the input through our first set of CONV => RELU =>
# 		# POOL layers
# 		x = self.conv1(x)
# 		x = self.relu1(x)
# 		x = self.maxpool1(x)
# 		# pass the output from the previous layer through the second
# 		# set of CONV => RELU => POOL layers
# 		x = self.conv2(x)
# 		x = self.relu2(x)
# 		x = self.maxpool2(x)
# 		# flatten the output from the previous layer and pass it
# 		# through our only set of FC => RELU layers
# 		x = flatten(x, 1)
# 		x = self.fc1(x)
# 		x = self.relu3(x)
# 		# pass the output to our softmax classifier to get our output
# 		# predictions
# 		x = self.fc2(x)
# 		output = self.logSoftmax(x)
# 		# return the output predictions
# 		return output