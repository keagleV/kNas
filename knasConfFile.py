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


		# Maximum number of hidden layers
		self.maxHiddenLayersr =  '\s*MAX_NUM_HIDDEN_LAYERS\s*=\s*[1-9][0-9]*\s*'
		

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
		self.mutProbClr = '\s*MUT_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'


		# Mutation addition prob
		self.mutAddProbClr = '\s*MUT_ADD_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation modification prob
		self.mutModProbClr = '\s*MUT_MOD_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove prob
		self.mutRemProbClr = '\s*MUT_REM_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'



		# Mutation add cn layer
		self.mutAddCnLayerr = '\s*MUT_ADD_CN_LAYER\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation add batchnorm prob
		self.mutAddBatchProbClr = '\s*MUT_ADD_BATCHNORM_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation add activation function prob
		self.mutAddAcFuncProbClr = '\s*MUT_ADD_ACTFUNC_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation add dropout prob
		self.mutAddDropoutProbClr = '\s*MUT_ADD_DROPOUT_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation add maxpool prob
		self.mutAddMaxPoolProbClr = '\s*MUT_ADD_MAXPOOL_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'


		# Mutation modification activation function prob
		self.mutModAcFuncProbClr = '\s*MUT_MOD_ACTFUNC_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation modification dropout prob
		self.mutModDropoutProbClr = '\s*MUT_MOD_DROPOUT_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation modification filters prob
		self.mutModFiltersProbClr = '\s*MUT_MOD_FILTERS_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'




		# Mutation remove cn layer prob
		self.mutRemCnLayerProbr = '\s*MUT_REM_CNLAYER_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove batchnorm prob
		self.mutRemBatchProbClr = '\s*MUT_REM_BATCHNORM_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove activation function prob
		self.mutRemAcFuncProbClr = '\s*MUT_REM_ACTFUNC_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove dropout prob
		self.mutRemDropoutProbClr = '\s*MUT_REM_DROPOUT_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove maxpool prob
		self.mutRemMaxPoolProbClr = '\s*MUT_REM_MAXPOOL_PROB_CL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'








		#############################







		# Mutation Prob
		self.mutProbDlr = '\s*MUT_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'


		# Mutation addition prob
		self.mutAddProbDlr = '\s*MUT_ADD_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation modification prob
		self.mutModProbDlr = '\s*MUT_MOD_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove prob
		self.mutRemProbDlr = '\s*MUT_REM_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'



		# Mutation add Hidden layer
		self.mutAddHiLayerr = '\s*MUT_ADD_HI_LAYER\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation add batchnorm prob
		self.mutAddBatchProbDlr = '\s*MUT_ADD_BATCHNORM_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation add activation function prob
		self.mutAddAcFuncProbDlr = '\s*MUT_ADD_ACTFUNC_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation add dropout prob
		self.mutAddDropoutProbDlr = '\s*MUT_ADD_DROPOUT_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation add maxpool prob
		self.mutAddMaxPoolProbDlr = '\s*MUT_ADD_MAXPOOL_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'


		# Mutation modification activation function prob
		self.mutModAcFuncProbDlr = '\s*MUT_MOD_ACTFUNC_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation modification dropout prob
		self.mutModDropoutProbDlr = '\s*MUT_MOD_DROPOUT_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation modification filters prob
		self.mutModFiltersProbDlr = '\s*MUT_MOD_FILTERS_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'




		# Mutation remove Hidden layer prob
		self.mutRemHiLayerProbr = '\s*MUT_REM_HIL_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove batchnorm prob
		self.mutRemBatchProbDlr = '\s*MUT_REM_BATCHNORM_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove activation function prob
		self.mutRemAcFuncProbDlr = '\s*MUT_REM_ACTFUNC_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove dropout prob
		self.mutRemDropoutProbDlr = '\s*MUT_REM_DROPOUT_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'

		# Mutation remove maxpool prob
		self.mutRemMaxPoolProbDlr = '\s*MUT_REM_MAXPOOL_PROB_DL\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'








		############################3





		# Learning rate mutation
		self.mutLearningRateProbr = '\s*MUT_LEARNING_RATE_PROB\s*=\s((?!0+(?:\.0+)?$)\d?\d(?:\.\d\d*)?)\s*'


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

				elif match(self.maxHiddenLayersr,line):
					parameters['MAX_NUM_HIDDEN_LAYERS']= int(line.split("=")[1].strip())

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





				elif match(self.mutProbClr,line):
					parameters['MUT_PROB_CL']= float(line.split("=")[1].strip())

				elif match(self.mutAddProbClr,line):
					parameters['MUT_ADD_PROB_CL']= float(line.split("=")[1].strip())

				elif match(self.mutModProbClr,line):
					parameters['MUT_MOD_PROB_CL']= float(line.split("=")[1].strip())

				elif match(self.mutRemProbClr,line):
					parameters['MUT_REM_PROB_CL']= float(line.split("=")[1].strip())

				
				elif match(self.mutAddCnLayerr,line):
					parameters['MUT_ADD_CN_LAYER']= float(line.split("=")[1].strip())

				elif match(self.mutAddBatchProbClr,line):
					parameters['MUT_ADD_BATCHNORM_PROB_CL']= float(line.split("=")[1].strip())

				elif match(self.mutAddAcFuncProbClr,line):
					parameters['MUT_ADD_ACTFUNC_PROB_CL']= float(line.split("=")[1].strip())

				elif match(self.mutAddDropoutProbClr,line):
					parameters['MUT_ADD_DROPOUT_PROB_CL']= float(line.split("=")[1].strip())

				elif match(self.mutAddMaxPoolProbClr,line):
					parameters['MUT_ADD_MAXPOOL_PROB_CL']= float(line.split("=")[1].strip())


				elif match(self.mutModAcFuncProbClr,line):
					parameters['MUT_MOD_ACTFUNC_PROB_CL']= float(line.split("=")[1].strip())

				elif match(self.mutModDropoutProbClr,line):
					parameters['MUT_MOD_DROPOUT_PROB_CL']= float(line.split("=")[1].strip())

				elif match(self.mutModFiltersProbClr,line):
					parameters['MUT_MOD_FILTERS_PROB_CL']= float(line.split("=")[1].strip())


				elif match(self.mutRemCnLayerProbr,line):
					parameters['MUT_REM_CNLAYER_PROB']= float(line.split("=")[1].strip())


				elif match(self.mutRemBatchProbClr,line):
					parameters['MUT_REM_BATCHNORM_PROB_CL']= float(line.split("=")[1].strip())


				elif match(self.mutRemAcFuncProbClr,line):
					parameters['MUT_REM_ACTFUNC_PROB_CL']= float(line.split("=")[1].strip())


				elif match(self.mutRemDropoutProbClr,line):
					parameters['MUT_REM_DROPOUT_PROB_CL']= float(line.split("=")[1].strip())


				elif match(self.mutRemMaxPoolProbClr,line):
					parameters['MUT_REM_MAXPOOL_PROB_CL']= float(line.split("=")[1].strip())



				##############

				elif match(self.mutProbDlr,line):
					parameters['MUT_PROB_DL']= float(line.split("=")[1].strip())

				elif match(self.mutAddProbDlr,line):
					parameters['MUT_ADD_PROB_DL']= float(line.split("=")[1].strip())

				elif match(self.mutModProbDlr,line):
					parameters['MUT_MOD_PROB_DL']= float(line.split("=")[1].strip())

				elif match(self.mutRemProbDlr,line):
					parameters['MUT_REM_PROB_DL']= float(line.split("=")[1].strip())

				
				elif match(self.mutAddHiLayerr,line):
					parameters['MUT_ADD_HI_LAYER']= float(line.split("=")[1].strip())

				elif match(self.mutAddBatchProbDlr,line):
					parameters['MUT_ADD_BATCHNORM_PROB_DL']= float(line.split("=")[1].strip())

				elif match(self.mutAddAcFuncProbDlr,line):
					parameters['MUT_ADD_ACTFUNC_PROB_DL']= float(line.split("=")[1].strip())

				elif match(self.mutAddDropoutProbDlr,line):
					parameters['MUT_ADD_DROPOUT_PROB_DL']= float(line.split("=")[1].strip())

				elif match(self.mutAddMaxPoolProbDlr,line):
					parameters['MUT_ADD_MAXPOOL_PROB_DL']= float(line.split("=")[1].strip())


				elif match(self.mutModAcFuncProbDlr,line):
					parameters['MUT_MOD_ACTFUNC_PROB_DL']= float(line.split("=")[1].strip())

				elif match(self.mutModDropoutProbDlr,line):
					parameters['MUT_MOD_DROPOUT_PROB_DL']= float(line.split("=")[1].strip())

				elif match(self.mutModFiltersProbDlr,line):
					parameters['MUT_MOD_FILTERS_PROB_DL']= float(line.split("=")[1].strip())


				elif match(self.mutRemHiLayerProbr,line):
					parameters['MUT_REM_HIL_PROB']= float(line.split("=")[1].strip())


				elif match(self.mutRemBatchProbDlr,line):
					parameters['MUT_REM_BATCHNORM_PROB_DL']= float(line.split("=")[1].strip())


				elif match(self.mutRemAcFuncProbDlr,line):
					parameters['MUT_REM_ACTFUNC_PROB_DL']= float(line.split("=")[1].strip())


				elif match(self.mutRemDropoutProbDlr,line):
					parameters['MUT_REM_DROPOUT_PROB_DL']= float(line.split("=")[1].strip())


				elif match(self.mutRemMaxPoolProbDlr,line):
					parameters['MUT_REM_MAXPOOL_PROB_DL']= float(line.split("=")[1].strip())















				elif match(self.mutLearningRateProbr,line):
					parameters['MUT_LEARNING_RATE_PROB']=  float(line.split("=")[1].strip())

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

