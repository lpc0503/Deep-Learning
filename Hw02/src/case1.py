  
import numpy as np
import random
from Neuron import *
from Network import *
from pch import *

if __name__ == "__main__":

	openFile("case1.txt")

	network = Network()	

	trainingData = open("data/case1.txt", "r")
	trainData = trainingData.readlines()

	network.createNeuron()

	network.train(trainData)

	network.draw("case1", trainData)