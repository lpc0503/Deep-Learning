  
import numpy as np
import random
import sys
import time
import os
import matplotlib.pyplot as plt
from Neuron import *
from Network import *
from pch import *

if __name__ == "__main__":

	network = Network()	

	trainingData = open("data/train.txt", "r")
	trainData = trainingData.readlines()

	network.createNeuron()

	network.train(trainData)

	testingData = open("data/test.txt", "r")
	testData = testingData.readlines()

	# network.classify(testData)

	network.draw(trainData, testData)