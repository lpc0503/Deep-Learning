import math
import random
import time
from Network import *
from Layer import *
from neuron import *

MAXIMUM = 80000#epoch 上限
T = 0.2#RMSE < 0.2 then break

OUTPUTFILE = None

def printToFile(string):
	if OUTPUTFILE == None:
		print("FILE NOT OPEN!!!")
	else:
		print(string)
		print(string, file=OUTPUTFILE)
		

def calAccuracy(data, network):#計算準確率

	correct = 0
	for inputs, outputs in data:
		if network.getOutput(inputs) == outputs:
			correct += 1

	return (correct/(len(data))*100)

def readData(file, label=None):#讀檔

	arr = []
	labelArr = []

	if label:
		for data in label.readlines():
			labelArr.append(int(data))

	for i, data in enumerate(file.readlines()):
		# # print(data)
		# sl, sw, pl, pw, kind = data.split()
		temp = data.split(',')
		temp = [float(x) for x in temp]
		tmp = (temp, convert(labelArr[i]))
		arr.append(tmp)
	# print(arr)
	return arr

def convert(kind):#將種類喘換成矩陣
	if kind == 'setosa':
		return [0.9, 0.1, 0.1]
	elif kind == 'versicolor':
		return [0.1, 0.9, 0.1]
	else:
		return [0.1, 0.1, 0.9]

LEARNINGRATE = [100]#測試的learning rate
HIDDEN       = [35]#測試的hidden個數

if __name__ == '__main__':

	TIME		 = 0.0
	OUTPUTFILE = open("output_for_ass.txt", "w")
	TRAININGDATA = open("aa.txt", "r")
	aaa = open("aaa.txt", "r")
	TESTINGDATA = open("iris_testing_data.txt", "r")
	trainingData = readData(TRAININGDATA, aaa)
	# testingData = readDate(TESTINGDATA)


	printToFile("epoch limitation: {}".format(MAXIMUM))
	printToFile("T: {}".format(T))
	printToFile("--------------------I am the Divider--------------------")
	OUTPUTFILE.close()
	for i in HIDDEN:
		for j in LEARNINGRATE:

			OUTPUTFILE = open("output_for_ass.txt", "a")
			network = Network(j)#產生一個network並給定leaening rate
			hiddenLayer = Layer(784, i)#產生hidden layer 並給定input output 個數
			outputLayer = Layer(i, 3)#產生ouptut layer 並給定input output 個數

			#將此兩個layer放進network中
			network.addLayer(hiddenLayer)
			network.addLayer(outputLayer)

			epoch = 1

			while epoch < MAXIMUM and network.calError(trainingData) > T:#訓練終止條件
				network.train(trainingData)#訓練
				print("Hidden: {}, Rate {},  epoch {}, output {}".format(i, j, epoch, network.calError(trainingData)))
				epoch +=1 
			printToFile("Number of hidden neurons: {}".format(i))
			printToFile("Learning rates: {}".format(j))
			OUTPUTFILE.close()

			OUTPUTFILE = open("output_for_ass.txt", "a")
			trainingAccuracy = calAccuracy(trainingData, network)  # 計算trainging準確率
			printToFile("training accuracies {}%".format(trainingAccuracy))


			# testingAccuracy = calAccuracy(testingData, network)#利用testing data檢測此network準確率
			# printToFile("testing  accuracies = {}%".format(testingAccuracy))
			# printToFile("--------------------I am the Divider--------------------")
			# OUTPUTFILE.close()


