  
import numpy as np
import random
from Neuron import *
from Network import *
from pch import *

if __name__ == "__main__":

	openFile("case4.txt")

	network = Network()	

	trainingData = open("data/case4.txt", "r")
	trainData = trainingData.readlines()

	network.createNeuron()

	network.train(trainData)

	testingData = open("data/case4_test.txt", "r")
	testData = testingData.readlines()

	network.classify(testData)

	network.draw("case4", trainData, testData)