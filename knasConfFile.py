from re import match

class KNasConfFile:
	'''
		This class has implemented the KNAS configuration file parsing opearations
	'''

	def __init__(self):

		# INIT_LR_REGEX
		self.maxCNNLayersr = '\s*MAX_CNN\s*=\s*[1-9][0-9]*\s*'

		# BATCH_SIZE_REGEX
		self.batchSizer ='\s*BATCH_SIZE\s*=\s*[1-9][0-9]*\s*'

		# EPOCHS_REGEX
		self.epochsr= '\s*EPOCHS\s*=\s*([+-]?([0-9]*[.])?[0-9]+)\s*'
		
		# TRAIN_SPLIT_REGEX
		self.trainSplitr= '\s*TRAIN_SPLIT\s*=\s*([+-]?([0-9]*[.])?[0-9]+)\s*'

		# DEVICE_REGEX
		self.devicer='\s*DEVICE\s*=\s*(cpu|cuda)\s*'

		# Training Data Root Directory
		self.trainDataRootDirr = '\s*TRAIN_ROOT_DIR\s*=\s*([^\/]+)(\/([^\/]+))?\s*'

		# Testing Data Root Directory
		self.testDataRootDirr = '\s*TEST_ROOT_DIR\s*=\s*([^\/]+)(\/([^\/]+))?\s*'

		# Split Seed
		self.splitSeedr = '\s*SPLIT_SEED\s*=\s*[1-9][0-9]*\s*'

		# Population size
		self.popSize = '\s*POP_SIZE\s*=\s*([1-9][0-9]*[02468])\s*'

		# Generation evolution count
		self.genNum = '\s*GEN_NUM\s*=\s*\[\s*[1-9][0-9]*\s*\]\s*'

	
		# Mutation Prob
		self.crossProb = '\s*CROSS_PROB\s*=\s*\[\s*((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation Prob
		self.mutProb = '\s*MUT_PROB\s*=\s*\[\s*((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'


	def get_splitseed_value(self,definition):

		'''
			This function will return the split seed value from the definition line
		'''
		
		if match(self.splitSeedr,definition):
			return definition.split("=")[1].strip()

		return None



	def get_max_cnn_layer_value(self,definition):

		'''
			This function will return the maximum cnn layers value from the definition line
		'''
		
		if match(self.maxCNNLayersr,definition):
			return definition.split("=")[1].strip()

		return None


	def get_batchSize_value(self,definition):

		'''
			This function will return the batch size value from the definition line
		'''
		
		if match(self.batchSizer,definition):
			return definition.split("=")[1].strip()

		return None


	def get_epochs_value(self,definition):

		'''
			This function will return the epochs value from the definition line
		'''
		
		if match(self.epochsr,definition):
			return definition.split("=")[1].strip()

		return None

	
	def get_trainsplit_value(self,definition):

		'''
			This function will return the train split value from the definition line
		'''
		
		if match(self.trainSplitr,definition):
			return definition.split("=")[1].strip()

		return None

	def get_device_value(self,definition):

		'''
			This function will return the device value from the definition line
		'''
		
		if match(self.devicer,definition):
			return definition.split("=")[1].strip()

		return None


	def get_trainRootDir_value(self,definition):

		'''
			This function will return the train root dir value from the definition line
		'''
		
		if match(self.trainDataRootDirr,definition):
			return definition.split("=")[1].strip()

		return None


	def get_testRootDir_value(self,definition):

		'''
			This function will return the test root dir value from the definition line
		'''
		
		if match(self.testDataRootDirr,definition):
			return definition.split("=")[1].strip()

		return None

	def get_splitseed_value(self,definition):

		'''
			This function will return the split seed value from the definition line
		'''
		
		if match(self.splitSeedr,definition):
			return definition.split("=")[1].strip()

		return None

