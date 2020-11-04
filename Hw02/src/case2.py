  
import numpy as np
import random
from Neuron import *
from Network import *
from pch import *

if __name__ == "__main__":

	openFile("case2.txt")

	network = Network()	

	trainingData = open("data/case2.txt", "r")
	trainData = trainingData.readlines()

	network.createNeuron()

	network.train(trainData)

	network.draw("case2", trainData)