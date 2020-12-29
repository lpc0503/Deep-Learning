import random
import numpy as np
import math

class Neuron():

	def __init__(self, n_weight):
		self.input = []
		self.weight = []
		self.bias = random.uniform(0, 1)
		self.output = 0.0
		self.delta = 0.0
		for i in range(n_weight):#初始化w為0~1的數字
			self.weight.append(random.uniform(0, 1))


	#將所有input與其對應的w相乘並全部累加起來，利用sigmoid得出output
	def activate(self, inputs):
		self.input = inputs
		self.output = 0
		for (i, w) in enumerate(self.weight):
			self.output += w * inputs[i]
		self.output = self.sigmoid(self.output + self.bias)
		return self.output

	def sigmoid(self, z):#sigmoid fun
		return 1 / (1 + math.exp(-z))

	def calDelta(self, error):#計算誤差
		self.delta = error * self.output * (1-self.output)

	def update(self, learningRate):#更新w跟b
		for (i, w) in enumerate(self.weight):
			w_new = w - learningRate * self.delta * self.input[i]
			self.weight[i] = w_new
		self.bias = self.bias - learningRate * self.delta

	def print_info(self):
		print("weight: {}".format(self.weight))
		print("bias: {}".format(self.bias))