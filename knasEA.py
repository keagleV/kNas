

from knasLn import CNLayer

from knasLn import DFCLayer



class KNasEAIndividual:
	'''
		This class has implemented the individual solutions of the EA algorithm
	'''
	def __init__(self):


		# Number of CN layers in the first part of the network
		self.cnLayersCount=int()

		# List of CN layers in the first part of the network
		self.cnLayersList=list()

		# DFC layer, the only layer in the last part of the network
		self.dfcLayer=None

		# Learning rate
		self.learningRate = int()
		

	def create_random_individual(self):

		pass









class KNasEA:

	'''
		This class has implemented the evolutionary methods used in 
		developing the KNAS program
	'''

	def __init__(self,popSize=10,genNum=20):

		# Population size
		self.popSize=popSize

		# Number of generations
		self.genNum=genNum

		pass





