import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder

OUTPUTFILE = None

def printToFile(string):
	if OUTPUTFILE == None:
		print("FILE NOT OPEN!!!")
	else:
		print(string, file=OUTPUTFILE)

def readData(file, label=None):
	return pd.read_table(file, header=None, sep=',').to_numpy()

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

slices = 6000
def reArrange(training, training_img_label, training_valid_label, test_img):
	
	data = []
	for i in range(len(training)):
		data.append(training[i].reshape(784, 1))
	data = np.array(data)
	label = []
	for i in range(len(training_img_label)):
		label.append(training_img_label[i].reshape(3, 1))
	label = np.array(label)
	train_img, valid_img = data[:slices, :], data[slices:, :]
	train_label = label[:slices,:]
	training_valid_label = training_valid_label[slices:, :]
	train_data = []
	test_data = []
	for i in range(len(train_img)):
		tmp = (train_img[i], train_label[i])
		train_data.append(tmp)
	for i in range(len(valid_img)):
		tmp = (valid_img[i], training_valid_label[i])
		test_data.append(tmp)

	test = []
	for i in range(len(test_img)):
		test.append(test_img[i].reshape(784, 1))
	test = np.array(test)

	return train_data, test_data, test


class Network():

	def __init__(self, sizes):

		self.layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, a):

		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a)+b)
		return a

	def check(self, data):
		res = []
		for a, b in data:
			tmp = (np.argmax(self.feedforward(a)), np.argmax(b))
			res.append(tmp)
		total = sum(int(a) == int(b) for (a, b) in res)

		return total

	def train(self, training_data, epochs, mini_batch_size, learningRate, test_data):

		epoch = 0

		training_data = list(training_data)
		test_data = list(test_data)

		while epoch < epochs:
			# random.shuffle(training_data)
			mini_batches = [ training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size) ]
			for mini_batch in mini_batches:
				self.update(mini_batch, learningRate)	
			print("Epoch {}:".format(epoch))
			print("Training Accuracy : {}%".format(self.check(training_data)/len(training_data) * 100))
			print("Verification Accuracy : {}%".format((self.evaluate(test_data)/len(test_data)) * 100))

			epoch+=1

	def update(self, mini_batch, learningRate):

		new_b = []
		new_w = []

		for bias in self.biases:
			new_b.append(np.zeros(bias.shape))

		for weight in self.weights:
			new_w.append(np.zeros(weight.shape))

		for x, y in mini_batch:
			delta_b, delta_w = self.backward(x, y)

			new_b = [newb + zipb for newb, zipb in zip(new_b, delta_b)]
			new_w = [neww + zipw for neww, zipw in zip(new_w, delta_w)]

		self.weights = [w - (learningRate/len(mini_batch)) * newb for w, newb in zip(self.weights, new_w)]
		self.biases  = [b - (learningRate/len(mini_batch)) * neww for b, neww in zip(self.biases, new_b)]

	def backward(self, x, y):

		new_b = []
		new_w = []

		for bias in self.biases:
			new_b.append(np.zeros(bias.shape))

		for weight in self.weights:
			new_w.append(np.zeros(weight.shape))

		activation = x
		activations = [x]
		arr = []
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			arr.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		delta = (activations[-1] - y) * sigmoid_prime(arr[-1])
		new_b[-1] = delta
		new_w[-1] = np.dot(delta, activations[-2].transpose())

		for l in range(2, self.layers):

			z = arr[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			new_b[-l] = delta
			new_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (new_b, new_w)

	def evaluate(self, test_data):

		res = []

		for a, b in test_data:
			tmp = (np.argmax(self.feedforward(a)), b)
			res.append(tmp)

		total = sum(int(x) == int(y) for (x, y) in res)
		return total

hidden = 100
ep = 100
lr = 0.05

def main():

	global OUTPUTFILE

	OUTPUTFILE = open("output_for_assistance.txt", "w")

	training = readData('train_img.txt', ',')
	training_img_label = OneHotEncoder().fit_transform(pd.read_table('train_label.txt', header=None, sep=',')).toarray()
	training_valid_label = readData('train_label.txt', ',')
	test_img = readData('test_img.txt', ',')

	train_data, test_data, test = reArrange(training, training_img_label, training_valid_label, test_img)

	print("Training Data {}".format(slices))
	print("Verification Data {}".format(8000-slices))
	print("Hidden {}".format(hidden))
	print("epochs {}".format(ep))
	print("learningRate {}".format(lr))
	
	model = Network([784, hidden, 3])
	model.train(train_data, ep, 5, lr, test_data=test_data)

	print("prediction print to file")
	for i in range(len(test)):
		printToFile(np.argmax(model.feedforward(test[i])))

if __name__ == '__main__':
	main()
