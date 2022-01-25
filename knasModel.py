

class KNasModel:
	'''
		This class has implemented the operations to work with dataset, building and
		evaluating a model in the KNAS program
	'''
	def __init(self,datasetModule,knasParams):
		
		# KNas model dataset handler
		self.datasetHand = datasetModule

		# KNas model parameters
		self.knasParams = knasParams



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


	def knas_create_eval_model(self):

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


		trainSteps = len(trainDataLoader.dataset) // self.knasParams["BATCH_SIZE"]
		valSteps = len(valDataLoader.dataset) // self.knasParams["BATCH_SIZE"]
		testSteps = len(testDataLoader.dataset) // self.knasParams["BATCH_SIZE"]


		# Just for informing the user for the availability of the cuda
		if cuda.is_available() and (device!="cuda"):
				self.logModHand.knas_log_message(self.logModHand.loggingCodes['CUDA_AVAILABLE'],'INF')







		# Creating the model
		model = KNasLayersNet( 1, None , None).to(device)		

















		# Optimization
		opt = Adam(model.parameters(), lr= self.knasParams["INIT_LR"])

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


		self.logModHand.knas_log_message(self.logModHand.loggingCodes['MODEL_TRAINING_STARTED'],'INF')

		startTime=time()

		for e in range(0, self.knasParams["EPOCHS"]):

			print("start")
			

			# set the model in training mode
			model.train()
			# initialize the total training and validation loss
			totalTrainLoss = 0
			totalValLoss = 0
			# initialize the number of correct predictions in the training
			# and validation step
			trainCorrect = 0
			valCorrect = 0
			# loop over the training set
			i=0
			for (x, y) in trainDataLoader:
				print(i)
				i+=1

				# send the input to the device
				(x, y) = (x.to(device), y.to(device))
				# perform a forward pass and calculate the training loss
				pred = model(x)
				loss = lossFn(pred, y)
				# zero out the gradients, perform the backpropagation step,
				# and update the weights
				opt.zero_grad()
				loss.backward()
				opt.step()
				# add the loss to the total training loss so far and
				# calculate the number of correct predictions
				totalTrainLoss += loss

				trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()


			# Finding the performance parameters on the validation data
			with torch.no_grad():
				
				# set the model in evaluation mode
				model.eval()
				# loop over the validation set
				for (x, y) in valDataLoader:
					
					# send the input to the device
					(x, y) = (x.to(device), y.to(device))
					
					# make the predictions and calculate the validation loss
					pred = model(x)
					
					totalValLoss += lossFn(pred, y)
					
					# calculate the number of correct predictions
					valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

			
			# Calculating the average training and validation loss
			avgTrainLoss = totalTrainLoss / trainSteps
			avgValLoss = totalValLoss / valSteps
			

			# Calculating the training and validation accuracy
			trainCorrect = trainCorrect / len(trainDataLoader.dataset)
			valCorrect = valCorrect / len(valDataLoader.dataset)
		

			# Updating the training performance status
			modelPerfomanceStatus["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
			modelPerfomanceStatus["train_acc"].append(trainCorrect)
			modelPerfomanceStatus["val_loss"].append(avgValLoss.cpu().detach().numpy())
			modelPerfomanceStatus["val_acc"].append(valCorrect)
			


			# print the model training and validation information
			# print("[INFO] EPOCH: {}/{}".format(e + 1, self.knasParams["EPOCHS"]))
			# print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
			# 	avgTrainLoss, trainCorrect))
			# print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
			# 	avgValLoss, valCorrect))


			endTime= time()
			
			self.logModHand.knas_log_message(self.logModHand.loggingCodes['MODEL_TRAINING_FINISHED'],'INF')


			# Setting the training duration
			modelPerfomanceStatus["training_time"].append(endTime-startTime)


			# Finding the performance parameters on the test data
			
			# Total loss value of the testing phase
			testTotalValLoss=0

			with torch.no_grad():


				# set the model in evaluation mode
				model.eval()
				
				# initialize a list to store our predictions
				preds = []
				# loop over the test set
				for (x, y) in testDataLoader:
					# send the input to the device
					x = x.to(device)
					# make the predictions and add them to the list
					pred = model(x)
					
					preds.extend(pred.argmax(axis=1).cpu().numpy())
				
					testTotalValLoss+=lossFn(pred, y)

				# generate a classification report
				# testReport=(classification_report(testData.targets.cpu().numpy(),
				# 	np.array(preds), target_names=testData.classes))

				modelPerfomanceStatus["test_acc"].append(accuracy_score(testData.targets.cpu().numpy(),np.array(preds)))
				
				modelPerfomanceStatus["test_loss"].append((testTotalValLoss / testSteps).cpu().detach().numpy())




		#TODO average is rrequired?
		return modelPerfomanceStatus
