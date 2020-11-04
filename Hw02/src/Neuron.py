import random
import numpy as np
import math

ALPHA = 1

class Neuron:

	def __init__(self, dimension):
		self.dimension = dimension
		self.w = np.zeros(dimension)
		self.init_w = self.w
		self.e = 0
		self.countError = 0
		self.init()

	def init(self):
		# self.w[0] = random.uniform(-10, 10)
		# self.w[1] = random.uniform(-10, 10)
		# self.w[2] = random.uniform(-10, 10)
		self.w[0] = 1
		self.w[1] = 1
		self.w[2] = 1
		self.init_w = self.w
	
	def hardlims(self, inp):
		res = 0
		if inp >= 0:
			res = 1
		else:
			res = -1
		return res	 # hotfix

	def segmoid(self, inp):
		return 1.0 / (1.0 + math.exp(-inp))

	def calError(self, a, t):
		self.e = ALPHA * (t-a)
		# print(self.e)

	def update(self, x, y, y_hat):
		self.w = self.w + ALPHA * (y-y_hat) * x

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

