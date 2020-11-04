  
import numpy as np
import random
from Neuron import *
from Network import *
from pch import *

if __name__ == "__main__":

	openFile("case3.txt")

	network = Network()	

	trainingData = open("data/case3.txt", "r")
	trainData = trainingData.readlines()

	network.createNeuron()

	network.train(trainData)

	network.draw("case3", trainData)