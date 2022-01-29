from re import match

from ast import literal_eval



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
		self.popSizer = '\s*POP_SIZE\s*=\s[1-9][0-9]*\s*'

		# Generation evolution count
		self.genNumr = '\s*GEN_NUM\s*=\s*[1-9][0-9]*\s*'

		# Mutation Prob
		self.crossProbr = '\s*CROSS_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation Prob
		self.mutProbr = '\s*MUT_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'




		# Mutation addition prob
		self.mutAddProbr = '\s*MUT_ADD_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation modification prob
		self.mutModProbr = '\s*MUT_MOD_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove prob
		self.mutRemProbr = '\s*MUT_REM_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'




		# Mutation add batchnorm prob
		self.mutAddBatchProbr = '\s*MUT_ADD_BATCHNORM_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation add activation function prob
		self.mutAddAcFuncProbr = '\s*MUT_ADD_ACTFUNC_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation add dropout prob
		self.mutAddDropoutProbr = '\s*MUT_ADD_DROPOUT_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation add maxpool prob
		self.mutAddMaxPoolProbr = '\s*MUT_ADD_MAXPOOL_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'


		# Mutation modification activation function prob
		self.mutModAcFuncProbr = '\s*MUT_MOD_ACTFUNC_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation modification dropout prob
		self.mutModDropoutProbr = '\s*MUT_MOD_DROPOUT_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation modification filters prob
		self.mutModFiltersProbr = '\s*MUT_MOD_FILTERS_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'




		# Mutation remove cn layer prob
		self.mutRemCnLayerProbr = '\s*MUT_REM_CNLAYER_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove batchnorm prob
		self.mutRemBatchProbr = '\s*MUT_REM_BATCHNORM_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove activation function prob
		self.mutRemAcFuncProbr = '\s*MUT_REM_ACTFUNC_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove dropout prob
		self.mutRemDropoutProbr = '\s*MUT_REM_DROPOUT_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove maxpool prob
		self.mutRemMaxPoolProbr = '\s*MUT_REM_MAXPOOL_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Kernel size
		self.kernelSizer = '\s*KERNEL_SIZE\s*=\s*[1-9][0-9]*\s*'

		# Padding
		self.paddingr = '\s*PADDING\s*=\s*[1-9][0-9]*\s*'

		# Stride
		self.strider = '\s*STRIDE\s*=\s*[1-9][0-9]*\s*'

		# Dimension of the input
		self.inputDimensionr = '\s*INPUT_DIM\s*=\s*[1-9][0-9]*\s*'

		# Possible value for filter
		self.filterPossValuesr = '\s*FILTER_POSS_VALUES\s*=\s*\[\s*([1-9][0-9]*)(\s*,\s*([1-9][0-9]*))*\s*\]\s*'

		# Possible value for hidden layers neurons
		self.HiddenLayerNeuronsPossValuesr = '\s*HIDDEN_LAYERS_NEURONS_POSS_VALUE\s*=\s*\[\s*([1-9][0-9]*)(\s*,\s*([1-9][0-9]*))*\s*\]\s*'




	def get_parameters(self,confFilePath):
		'''
			This function will parse the file and return the parameters
		'''

		# Dictionary of parameters
		parameters = dict()


		fhandle=open(confFilePath)

		fileLines=fhandle.readlines()

		fhandle.close()


		for lineNum,line in enumerate(fileLines):

			line=line.strip()

			# Line number starts from zero, to handle this, we increment it 
			# to adjust it to start from one.
			lineNum+=1

			if line:

				if line.startswith('#'):
					continue

				if match(self.splitSeedr,line):
					parameters['SPLIT_SEED']= int (line.split("=")[1].strip())

				elif match(self.maxCNNLayersr,line):
					parameters['MAX_CNN']= int(line.split("=")[1].strip())

				elif match(self.batchSizer,line):
					parameters['BATCH_SIZE']= int(line.split("=")[1].strip())

				elif match(self.epochsr,line):
					parameters['EPOCHS']= int(line.split("=")[1].strip())

				elif match(self.trainSplitr,line):
					parameters['TRAIN_SPLIT']= float(line.split("=")[1].strip())

				elif match(self.devicer,line):
					parameters['DEVICE']=  line.split("=")[1].strip()

				elif match(self.trainDataRootDirr,line):
					parameters['TRAIN_ROOT_DIR']= line.split("=")[1].strip()
				
				elif match(self.testDataRootDirr,line):
					parameters['TEST_ROOT_DIR']= line.split("=")[1].strip()

				elif match(self.popSizer,line):
					parameters['POP_SIZE']= int(line.split("=")[1].strip())
					
				elif match(self.genNumr,line):
					parameters['GEN_NUM']= int(line.split("=")[1].strip())

				elif match(self.crossProbr,line):
					parameters['CROSS_PROB']= float(line.split("=")[1].strip())

				elif match(self.mutProbr,line):
					parameters['MUT_PROB']= float(line.split("=")[1].strip())

				elif match(self.mutAddProbr,line):
					parameters['MUT_ADD_PROB']= float(line.split("=")[1].strip())

				elif match(self.mutModProbr,line):
					parameters['MUT_MOD_PROB']= float(line.split("=")[1].strip())

				elif match(self.mutRemProbr,line):
					parameters['MUT_REM_PROB']= float(line.split("=")[1].strip())

				
				elif match(self.mutAddBatchProbr,line):
					parameters['MUT_ADD_BATCHNORM_PROB']= float(line.split("=")[1].strip())

				elif match(self.mutAddAcFuncProbr,line):
					parameters['MUT_ADD_ACTFUNC_PROB']= float(line.split("=")[1].strip())

				elif match(self.mutAddDropoutProbr,line):
					parameters['MUT_ADD_DROPOUT_PROB']= float(line.split("=")[1].strip())

				elif match(self.mutAddMaxPoolProbr,line):
					parameters['MUT_ADD_MAXPOOL_PROB']= float(line.split("=")[1].strip())


				elif match(self.mutModAcFuncProbr,line):
					parameters['MUT_MOD_ACTFUNC_PROB']= float(line.split("=")[1].strip())

				elif match(self.mutModDropoutProbr,line):
					parameters['MUT_MOD_DROPOUT_PROB']= float(line.split("=")[1].strip())

				elif match(self.mutModFiltersProbr,line):
					parameters['MUT_MOD_FILTERS_PROB']= float(line.split("=")[1].strip())


				elif match(self.mutRemCnLayerProbr,line):
					parameters['MUT_REM_CNLAYER_PROB']= float(line.split("=")[1].strip())


				elif match(self.mutRemBatchProbr,line):
					parameters['MUT_REM_BATCHNORM_PROB']= float(line.split("=")[1].strip())


				elif match(self.mutRemAcFuncProbr,line):
					parameters['MUT_REM_ACTFUNC_PROB']= float(line.split("=")[1].strip())


				elif match(self.mutRemDropoutProbr,line):
					parameters['MUT_REM_DROPOUT_PROB']= float(line.split("=")[1].strip())


				elif match(self.mutRemMaxPoolProbr,line):
					parameters['MUT_REM_MAXPOOL_PROB']= float(line.split("=")[1].strip())


				elif match(self.inputDimensionr,line):
					parameters['INPUT_DIM']= int(line.split("=")[1].strip())


				elif match(self.kernelSizer,line):
					parameters['KERNEL_SIZE']= int(line.split("=")[1].strip())


				elif match(self.paddingr,line):
					parameters['PADDING']= int(line.split("=")[1].strip())


				elif match(self.strider,line):
					parameters['STRIDE']= int(line.split("=")[1].strip())

						
				elif match(self.filterPossValuesr,line):
					parameters['FILTER_POSS_VALUES']= literal_eval(line.split("=")[1].strip())


				elif match(self.HiddenLayerNeuronsPossValuesr,line):
					parameters['HIDDEN_LAYERS_NEURONS_POSS_VALUE']= literal_eval(line.split("=")[1].strip())

				# Error
				else:
					return None,lineNum



		return True,parameters
	# def get_splitseed_value(self,definition):

	# 	'''
	# 		This function will return the split seed value from the definition line
	# 	'''
		
	# 	if match(self.splitSeedr,definition):
	# 		return definition.split("=")[1].strip()

	# 	return None



	# def get_max_cnn_layer_value(self,definition):

	# 	'''
	# 		This function will return the maximum cnn layers value from the definition line
	# 	'''
		
	# 	if match(self.maxCNNLayersr,definition):
	# 		return definition.split("=")[1].strip()

	# 	return None


	# def get_batchSize_value(self,definition):

	# 	'''
	# 		This function will return the batch size value from the definition line
	# 	'''
		
	# 	if match(self.batchSizer,definition):
	# 		return definition.split("=")[1].strip()

	# 	return None




	# def get_epochs_value(self,definition):

	# 	'''
	# 		This function will return the epochs value from the definition line
	# 	'''
		
	# 	if match(self.epochsr,definition):
	# 		return definition.split("=")[1].strip()

	# 	return None

	



	# def get_trainsplit_value(self,definition):

	# 	'''
	# 		This function will return the train split value from the definition line
	# 	'''
		
	# 	if match(self.trainSplitr,definition):
	# 		return definition.split("=")[1].strip()

	# 	return None

	
	# def get_device_value(self,definition):

	# 	'''
	# 		This function will return the device value from the definition line
	# 	'''
		
	# 	if match(self.devicer,definition):
	# 		return definition.split("=")[1].strip()

	# 	return None


	# def get_trainRootDir_value(self,definition):

	# 	'''
	# 		This function will return the train root dir value from the definition line
	# 	'''
		
	# 	if match(self.trainDataRootDirr,definition):
	# 		return definition.split("=")[1].strip()

	# 	return None


	# def get_testRootDir_value(self,definition):

	# 	'''
	# 		This function will return the test root dir value from the definition line
	# 	'''
		
	# 	if match(self.testDataRootDirr,definition):
	# 		return definition.split("=")[1].strip()

	# 	return None

	# def get_splitseed_value(self,definition):

	# 	'''
	# 		This function will return the split seed value from the definition line
	# 	'''
		
	# 	if match(self.splitSeedr,definition):
	# 		return definition.split("=")[1].strip()

	# 	return None

