import math
import Layer
import neuron


class Network():

	def __init__(self, learningRate=1):
		self.neuronLayers = []
		self.learningRate = learningRate

	def train(self, data):#根據資料訓練
		for inputs, outputs in data:
			self.feed_forward(inputs)
			self.backward(outputs)
			self.update(self.learningRate)

	def feed_forward(self, inputs):#進算整個network
		tmp = inputs
		for layer in self.neuronLayers:
			tmp = layer.feed_forward(tmp)
		return tmp

	def backward(self, outputs):#回推求所有誤差
		layerNum = len(self.neuronLayers)
		counter = layerNum
		previous = []
		while counter != 0:
			currentLayer = self.neuronLayers[counter - 1]
			if len(previous) == 0:
				for i in range(len(currentLayer.neurons)):
					error = -(outputs[i] - currentLayer.neurons[i].output)
					currentLayer.neurons[i].calDelta(error)
			else:
				previousLayer = self.neuronLayers[counter]
				for i in range(len(currentLayer.neurons)):
					error = 0
					for j in range(len(previous)):
						error += previous[j] * previousLayer.neurons[j].weight[i]
					currentLayer.neurons[i].calDelta(error)
			previous = currentLayer.getDelta()
			counter -= 1

	def update(self, learningRate):#更新整個網路
		for layer in self.neuronLayers:
			layer.update(learningRate)

	def calError(self, data):#計算整個網路誤差(RMSE)
		total = 0
		for inputs, outputs in data:
			actual = self.feed_forward(inputs)
			for i in range(len(outputs)):
				total += (outputs[i] - actual[i]) ** 2
		
		return math.sqrt(total/len(data))


	def convert(self, outputSet):#將輸入轉為陣列
		if(max(outputSet) == outputSet[0]):
			return [0.9, 0.1, 0.1]
		elif (max(outputSet) == outputSet[1]):
			return [0.1, 0.9, 0.1]
		else:
			return [0.1, 0.1, 0.9]


	def getOutput(self, inputs):#獲取輸出
		return self.convert(self.feed_forward(inputs))
	
	def addLayer(self, neuronLayers):#將layer禁入進此網路
		self.neuronLayers.append(neuronLayers)
	
	def printf(self):
		for (i, l) in enumerate(self.neuronLayers):
			print("Layer: {}", i+1)
			l.print_neuron()

