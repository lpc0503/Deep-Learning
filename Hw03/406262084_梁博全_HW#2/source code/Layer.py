from neuron import *

class Layer():

	def __init__(self, inputNum, neuronNum):
		self.neurons = []
		for i in range(neuronNum):#每個layer裡面有n個nueron
			tmp = Neuron(inputNum)
			self.neurons.append(tmp)

	def feed_forward(self, inputs):#計算該layer所有output
		outputs = []
		for neuron in self.neurons:
			outputs.append(neuron.activate(inputs))
		return outputs

	def getDelta(self):#回傳所有誤差值
		return [neuron.delta for neuron in self.neurons]
	
	def update(self, learningRate):#更新所有neuron
		for neuron in self.neurons:
			neuron.update(learningRate)
	
	def print_neuron(self):
		for (i, neuron) in enumerate(self.neurons):
			print("neuron: {}".format(i+1))
			neuron.print_info()
			
	
