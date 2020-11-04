from Neuron import *
from pch import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as plt

Tou = 0.02

class Network:

	def __init__(self):
		self.neuron = None
		self.testT = []

	def createNeuron(self, dimension=3):
		self.neuron = Neuron(dimension)

	def RMSE(self, len_, sum_):
		# print(sum_)
		# print(len_)
		return math.sqrt(sum_/len_)

	def train(self, trainingData):
		# hot code 
		# TODO : change to match all dimension data
		p = []
		epoch = 0
		
		printToFile("Using {}-Dimension perceptron\nIntial w and b".format(self.neuron.dimension))
		printToFile("w:\n{}\n".format(self.neuron.w))
		epoch = 0
		while True:
			self.neuron.clearCountError()

			sum_ = 0
			for data in trainingData:

				x1, x2, t = data.split(',')
				p = np.array([float(1), float(x1), float(x2)])
				y = self.neuron.segmoid(self.neuron.w.dot(p.T))
				self.neuron.update(p, float(t), y)
				sum_ += float(t) - y;

			epoch += 1

			if(self.RMSE(len(trainingData), sum_*sum_) < Tou):
				printToFile("stop training for error measure is small enough")
				break

			if epoch > 10000:
				printToFile("training epoch exceed")
				break

		printToFile("Total epoch: {}\n\nFinal w:".format(epoch))
		printToFile("w:\n{}\n".format(self.neuron.w))

	def classify(self, testingData):

		printToFile("result: ")

		cnt = 1

		for data in testingData:

			x1, x2 = data.split(',')
			p = np.array([float(1), float(x1), float(x2)])

			a = self.neuron.hardlims(self.neuron.w.dot(p))

			self.testT.append(float(a))
			printToFile(str(cnt) + " " + str(a))

			cnt += 1


	def draw(self, trainingData, testingData):

		x1 = []
		y1 = []
		x2 = []
		y2 = []
		for data in trainingData:
			x_, y_, t = data.split(',')
			x_ = float(x_)
			y_ = float(y_)
			if float(t) > 0:
				x1.append(x_)
				y1.append(y_)
			else:
				x2.append(x_)
				y2.append(y_)
	
		# plt.figure("result")


		# t1 = []
		# u1 = []
		# t2 = []
		# u2 = []
		# for i, testData in enumerate(testingData):
		# 	t_, u_ = testData.split(',')
		# 	t_ = float(t_)
		# 	u_ = float(u_)
		# 	if self.testT[i] > 0:
		# 		t1.append(t_)
		# 		u1.append(u_)
		# 	else:
		# 		t2.append(t_)
		# 		u2.append(u_)



		a = np.arange(-20, 20, 0.1)
		b = -(a * self.neuron.w[1] + self.neuron.w[0])/self.neuron.w[2]


		plt.plot(x1, y1, "bo", markersize = 3)
		plt.plot(x2, y2, "ro", markersize = 3)
		plt.plot(a, b)
		# plt.plot(t1, u1, "k^", markersize = 3)
		# plt.plot(t2, u2, "g^", markersize = 3)

		plt.savefig("output/output.png") 

		plt.show()
		
