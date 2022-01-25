from re import match

class KNasConfFile:
	'''
		This class has implemented the KNAS configuration file parsing opearations
	'''

	def __init__(self):

		# INIT_LR_REGEX
		self.initLRr = '\s*INIT_LR\s*=\s*([+-]?([0-9]*[.])?[0-9]+)\s*'

		# BATCH_SIZE_REGEX
		self.batchSizer ='\s*BATCH_SIZE\s*=\s*([+-]?([0-9]*[.])?[0-9]+)\s*'

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



	def get_splitseed_value(self,definition):

		'''
			This function will return the split seed value from the definition line
		'''
		
		if match(self.splitSeedr,definition):
			return definition.split("=")[1].strip()

		return None



	def get_initlr_value(self,definition):

		'''
			This function will return the init lr value from the definition line
		'''
		
		if match(self.initLRr,definition):
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

