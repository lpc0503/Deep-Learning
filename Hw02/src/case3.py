  
import numpy as np
import random
from Neuron import *
from Network import *
from pch import *

if __name__ == "__main__":

	network = Network()	

	trainingData = open("data/case3.txt", "r")
	trainData = trainingData.readlines()

	network.createNeuron()

	network.train(trainData)

	network.draw(trainData)