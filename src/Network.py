from Neuron import *
from pch import *
import numpy as np

class Network:

	def __init__(self):
		self.neuron = None

	def createNeuron(self, dimension=2):
		self.neuron = Neuron(dimension)

	def train(self, trainingData):
		# hot code 
		# TODO : change to match all dimension data
		p = []
		epoch = 0
		
		printToFile("Using {}-Dimension perceptron\nIntial w and b".format(self.neuron.dimension))
		printToFile("w:\n{}\n".format(self.neuron.w))
		printToFile("b:\n{}\n".format(self.neuron.b))
		epoch = 0
		while True:
			self.neuron.clearCountError()

			for data in trainingData:

				x1, x2, t = data.split(',')
				p = np.array([[float(x1)], [float(x2)]])
				a = self.neuron.hardlims(self.neuron.w.dot(p) + self.neuron.b)
				self.neuron.calError(a, float(t))

				if self.neuron.hasError():
					self.neuron.update(p)

			epoch += 1
			printToFile("In epoch {} detect {} error(s)".format(epoch, self.neuron.getErrorNumber()))

			if self.neuron.getErrorNumber() == 0:
				break

			if epoch > 100000:
				printToFile("training epoch exceed")
				break

		printToFile("Total epoch: {}\n\nFinal w and b:".format(epoch))
		printToFile("w:\n{}\n".format(self.neuron.w))
		printToFile("b:\n{}\n".format(self.neuron.b))

	def classify(self, testingData):

		printToFile("result: ")

		cnt = 1

		for data in testingData:

			x1, x2 = data.split(',')
			p = np.array([[float(x1)], [float(x2)]])

			a = self.neuron.hardlims(self.neuron.w.dot(p) + self.neuron.b)

			printToFile(str(cnt) + " " + str(a))

			cnt += 1


