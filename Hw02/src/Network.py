from Neuron import *
from pch import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
Tou = 0.002

class Network:

	def __init__(self):
		self.neuron = None
		self.testT = []

	def createNeuron(self, dimension=3):
		self.neuron = Neuron(dimension)

	def RMSE(self, len_, sum_):
		return math.sqrt(sum_/len_)

	def cross_entropy(self, y, y_hat):
		ret = -(y * math.log(y_hat) + (1-y)*math.log(1-y_hat))
		return ret < Tou

	def train(self, trainingData):
		# hot code 
		# TODO : change to match all dimension data
		p = []
		epoch = 0
		
		printToFile("Using {}-Dimension perceptron\nIntial w and b".format(self.neuron.dimension))
		printToFile("w:\n{}\n".format(self.neuron.w))
		epoch = 0
		while True:

			for data in trainingData:

				x1, x2, t = data.split(',')
				p = np.array([float(1), float(x1), float(x2)])
				print(self.neuron.w.T)
				y = self.neuron.segmoid(self.neuron.w.dot(p.T))
				self.neuron.update(p, float(t), y)

			epoch += 1

			if(self.cross_entropy(float(t), y)):
				printToFile("stop training for error measure is small enough")
				break

			if epoch > 100000:
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


	def draw(self, trainingData, testingData=None):

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
	
		plt.figure("result")

		if testingData:
			t1 = []
			u1 = []
			t2 = []
			u2 = []
			for i, testData in enumerate(testingData):
				t_, u_ = testData.split(',')
				t_ = float(t_)
				u_ = float(u_)
				if self.testT[i] > 0:
					t1.append(t_)
					u1.append(u_)
				else:
					t2.append(t_)
					u2.append(u_)



		a = np.arange(-3, 3, 0.1)
		b = -(a * self.neuron.w[1] + self.neuron.w[0])/self.neuron.w[2]

		init_b = -(a * self.neuron.init_w[1] + self.neuron.init_w[0])/self.neuron.init_w[2]


		plt.plot(x1, y1, "bo", markersize = 3)
		plt.plot(x2, y2, "ro", markersize = 3)
		plt.plot(a, b)
		plt.plot(a, init_b, label = "Init", color = "red", linestyle ="--")
		if testingData:
			plt.plot(t1, u1, "k^", markersize = 3)
			plt.plot(t2, u2, "g^", markersize = 3)

		plt.savefig("output/output.png") 

		plt.show()
		
