import random
import numpy as np

ALPHA = 1

class Neuron:

	def __init__(self, dimension=2):
		self.dimension = dimension
		self.w = np.zeros(2)
		self.b = 0
		self.e = 0
		self.countError = 0
		self.init()

	def init(self):
		self.w[0] = 0.1
		self.w[1] = -0.1
		# for i in range(0, self.neruonNumber):
		# 	for j in range(0, self.dimension):
		# 		self.w[i][j] = random.uniform(-10, 10)


		self.b = 1
	
	def hardlims(self, inp):
		res = 0
		if inp >= 0:
			res = 1
		else:
			res = -1
		return res	 # hotfix

	def calError(self, a, t):
		self.e = ALPHA * (t-a)
		# print(self.e)

	def update(self, inp):
		self.w = self.e * inp.T + self.w
		self.b = self.e + self.b

	def hasError(self):
		if self.e != 0:
			self.countError += 1
			return True
		else:
			return False

	def getErrorNumber(self):
		return self.countError

	def clearCountError(self):
		self.countError = 0

	def print(self):
		print(self.w)
		print(self.b)

