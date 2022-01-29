from knasLN import KNasLayersNet
from knasModelLogging import KNasModelLogging
from torch import float as tfloat
from torch import cuda
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch import no_grad
from time import time



class KNasModel:
	'''
		This class has implemented the operations to work with dataset, building and
		evaluating a model in the KNAS program
	'''
	def __init__(self,datasetModule,knasParams):
		
		# KNas model dataset handler
		self.datasetHand = datasetModule

		# KNas model parameters
		self.knasParams = knasParams

		# Model logging handler
		self.logModHand=KNasModelLogging()



	def knas_setup_dataset(self):
		'''
			This function will setup the dataset for the KNAS
		'''

		# Setting the root directory of the training data
		self.datasetHand.set_traindata_root_dir(self.knasParams['TRAIN_ROOT_DIR'])
		
		# Setting the root directory of the testing data
		self.datasetHand.set_testdata_root_dir(self.knasParams['TEST_ROOT_DIR'])


		# Loading the training data
		self.datasetHand.load_traindata()


		# Loading the testing data
		self.datasetHand.load_testdata()


		# Splitting the training data set into two groups
		self.datasetHand.split_training_dataset(self.knasParams['TRAIN_SPLIT'],self.knasParams['SPLIT_SEED'])



	def knas_create_eval_model(self,cnnLayers,dfcLayer,learningRate):

		'''
			This function will create the model and evaluate it against the 
			training and the testing dataset
		'''


		# Setting the device
		device = self.knasParams["DEVICE"]


		# Retrieving the training data
		trainData = self.datasetHand.get_traindata()
		
		# Retrieving the training data
		testData = self.datasetHand.get_testdata()


		# Retrieving the training data loader
		trainDataLoader = self.datasetHand.get_traindata_dataloader(self.knasParams['BATCH_SIZE'])

		# Retrieving the validation data loader
		valDataLoader = self.datasetHand.get_valdata_dataloader(self.knasParams['BATCH_SIZE'])


		# Retrieving the test data loader
		testDataLoader= self.datasetHand.get_testdata_dataloader(self.knasParams['BATCH_SIZE'])

		# Calculating the train, val, and test steps based on the batch size
		trainSteps = len(trainDataLoader.dataset) // self.knasParams["BATCH_SIZE"]
		valSteps = len(valDataLoader.dataset) // self.knasParams["BATCH_SIZE"]
		testSteps = len(testDataLoader.dataset) // self.knasParams["BATCH_SIZE"]


		# Just for informing the user for the availability of the cuda
		if cuda.is_available() and (device!="cuda"):
				self.logModHand.knas_model_log_message(self.logModHand.loggingCodes['CUDA_AVAILABLE'],'INF')

		
		# Creating the model
		model = KNasLayersNet( cnnLayers , dfcLayer).to(device)		

		
		# For debug case, print the layers in an individual
		# print("--------dfc------------")
		# for la in cnnLayers:
		# 	print(la)

		# print("--------dfc------------")
		# print(dfcLayer)



		# Optimization
		opt = Adam(model.parameters(), lr= learningRate )


		# Loss function
		lossFn = CrossEntropyLoss()
		

		# Model performance evaluation parameters
		modelPerfomanceStatus = {
			"training_time": [],
			"train_loss": [],
			"train_acc": [],
			"val_loss": [],
			"val_acc": [],
			"test_loss": [],
			"test_acc": [],
		}


		self.logModHand.knas_model_log_message(self.logModHand.loggingCodes['MODEL_TRAINING_STARTED'],'INF')


		epoch=0
		
		for e in range(0, self.knasParams["EPOCHS"]):

			print("Epoch Iter: ",epoch)
			epoch+=1



			# Setting the start time
			startTime=time()
			

			# Setting the model in training mode
			model.train()
			
			# Initializing the total training and validation loss
			totalTrainLoss = 0
			totalValLoss = 0
			testTotalValLoss=0
			
			# Initializing the number of correct predictions in the training
			# and validation step
			trainCorrect = 0
			valCorrect = 0
			testCorrect =0 
			

			# Loopign over the training set
			for (x, y) in trainDataLoader:

				# Sending the input to the device
				(x, y) = (x.to(device), y.to(device))
				
				# Performing a forward pass and calculate the training loss
				pred = model(x)
				loss = lossFn(pred, y)
				

				# Zero out the gradients, perform the backpropagation step,
				# and update the weights
				opt.zero_grad()
				loss.backward()
				opt.step()
				
				# add the loss to the total training loss so far and
				# calculate the number of correct predictions
				totalTrainLoss += loss

				trainCorrect += (pred.argmax(1) == y).type(tfloat).sum().item()




			# Finding the performance parameters on the validation data
			with no_grad():
				
				# Setting the model in evaluation mode
				model.eval()
				
				# Loop over the validation set
				for (x, y) in valDataLoader:
					
					# Sending the input to the device
					(x, y) = (x.to(device), y.to(device))
					
					# Making the predictions and calculate the validation loss
					pred = model(x)
					
					totalValLoss += lossFn(pred, y)
					
					# Calculating the number of correct predictions
					valCorrect += (pred.argmax(1) == y).type(tfloat).sum().item()

			
			# Calculating the average training and validation loss
			avgTrainLoss = totalTrainLoss / trainSteps
			avgValLoss = totalValLoss / valSteps
		

			# Calculating the training and validation accuracy
			trainCorrect = trainCorrect / len(trainDataLoader.dataset)
			valCorrect = valCorrect / len(valDataLoader.dataset)
		

			# Updating the training performance status
			modelPerfomanceStatus["train_loss"].append(avgTrainLoss.to(device).detach().item())
			modelPerfomanceStatus["train_acc"].append(trainCorrect)
			modelPerfomanceStatus["val_loss"].append(avgValLoss.to(device).detach().item())
			modelPerfomanceStatus["val_acc"].append(valCorrect)
			




			# Finding the performance parameters on the test data
			
			# Total loss value of the testing phase
			testTotalValLoss=0

			with no_grad():


				# Setting the model in evaluation mode
				model.eval()
				
				# Looping over the test set
				for (x, y) in testDataLoader:
					
					# Sending the input to the device
					x = x.to(device)
					y=y.to(device)
					
					# Making the predictions and add them to the list
					pred = model(x)
					
					testCorrect += (pred.argmax(1) == y).type(tfloat).sum().item()


					testTotalValLoss+=lossFn(pred, y)

				

				modelPerfomanceStatus["test_acc"].append(testCorrect/len(testDataLoader.dataset))

				modelPerfomanceStatus["test_loss"].append((testTotalValLoss / testSteps).to(device).detach().item())


			# Setting the end time
			endTime= time()
		

			# Setting the training duration of this epoch
			modelPerfomanceStatus["training_time"].append(endTime-startTime)


		self.logModHand.knas_model_log_message(self.logModHand.loggingCodes['MODEL_TRAINING_FINISHED'],'INF')

	
		return modelPerfomanceStatus
