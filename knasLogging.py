class KNasLogging:

	'''
		This class has implemented the KNasLogging which manages the 
		logging opeartions for the KNas program
	'''
	def __init__(self):

		self.loggingCodes = {
			'CONFIG_FILE_NOT_FOUND':'Configuration File Not Found',

			'CONFIG_FILE_NOT_SPECIFIED':'Configuration File Not Specified',

			'CONFIG_FILE_DEF_ERR': 'Config File Definition Error',

			'CUDA_AVAILABLE': 'CUDA Is Supported',

			'MODEL_TRAINING_STARTED': 'Training Of The Model Has Been Started',

			'MODEL_TRAINING_FINISHED': 'Training Of The Model Finished'

		}

	def knas_log_message(self,message,status,lineNumer=None):

		'''
			This function will log the message with its corresponding status
		'''

		print("[{0}]{1} {2} ".format(status,'' if lineNumer is None else ' L-'+str(lineNumer)+'  ' ,message))
