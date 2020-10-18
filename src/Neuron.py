import random
import numpy as np

ALPHA = 1

class Neuron:

	def __init__(self, dimension):
		self.dimension = dimension
		self.w = np.zeros(dimension)
		self.e = 0
		self.countError = 0
		self.init()

	def init(self):
		self.w[0] = random.uniform(-10, 10)
		self.w[1] = random.uniform(-10, 10)
		self.w[2] = random.uniform(-10, 10)
	
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

	def update(self, x, y):
		self.w = self.w + y * x * ALPHA

	def hasError(self, y, t):
		if y != t:
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

