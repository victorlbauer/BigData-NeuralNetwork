# encoding: utf-8
import math
import random
import numpy as np
import csv

class NeuralNetwork(object):
	def __init__(self, train_file, m_examples, n_params, n_classes, learning_rate, decay):
		self.training_file = train_file
		self.m_examples = m_examples
		self.n_params = n_params
		self.learning_rate = learning_rate
		self.decay = decay
		
		self.neurons_IL = self.n_params
		self.neurons_HL = int(math.ceil((2*self.n_params)/3))
		self.neurons_OL = n_classes
		
		self.W1, self.W2, self.W3 = self.initializeWeights()
		self.b1, self.b2, self.b3 = self.initializeBias()

	def crossEntropy_cost(self, batch, target):
		cost_sum = 0.0
		count = 0
		for example in batch:
			a2, a3, a4 = self.feedforward(example)
			cost_sum = sum(target[count]*np.log(a4) + (1 - target[count])*np.log(1 - a4))
			count += 1
		c_W1 = (np.power(self.W1, 2)).sum()
		c_W2 = (np.power(self.W2, 2)).sum()
		c_W3 = (np.power(self.W3, 2)).sum()	
		J = -(cost_sum/float(count)) + ((self.decay/2.0*float(count))*(c_W1 + c_W2 + c_W3))
		return J	
			
	def getMiniBatch(self, desired_batch_size):
		train_file = open(self.training_file, 'r')
		reader = csv.reader(train_file)
		minibatch = np.empty((0, self.n_params + 1), float)
		chance = (float(desired_batch_size)/float(self.m_examples))
		
		for linha in train_file:
			if(random.random() < chance):
				l = linha.split(',')
				l[-1] = l[-1].replace('\r\n', '')
				l = np.array(l, float)
				minibatch = np.vstack((minibatch, l))
				
		train_file.close()
		return minibatch
	
	def getTarget(self, batch):
		target = []
		for line in batch:
			target.append(line[-1])
		return target
	
	def createTargetClass(self, target_list):
		targets = np.empty((0, 2), float)
		for target in target_list:
			if(target == 0.0):
				targets = np.vstack((targets, [1.0, 0.0]))
			else:
				targets = np.vstack((targets, [0.0, 1.0]))
		return targets
	
	def removeTarget(self, batch):
		return batch[0:,0:-1]
		
	def initializeWeights(self):
		W1 = np.random.rand(self.neurons_HL, self.neurons_IL)
		W2 = np.random.rand(self.neurons_HL, self.neurons_HL)
		W3 = np.random.rand(self.neurons_OL, self.neurons_HL)
		return W1, W2, W3	
		
	def initializeBias(self):
		b1 = np.random.rand(1, self.neurons_HL)
		b2 = np.random.rand(1, self.neurons_HL)
		b3 = np.random.rand(1, self.neurons_OL)
		return b1[0], b2[0], b3[0]
		
	def train(self, epoch):
		for i in xrange(0, epoch):
			batch = self.getMiniBatch(512)
			target_list = self.getTarget(batch)
			batch = self.removeTarget(batch)
			target_list = self.createTargetClass(target_list)
		
			J = self.crossEntropy_cost(batch, target_list)
			print "Epoch:", i, "- J: " + "{0:.15f}".format(J)
		
			dW1 = np.zeros([self.neurons_HL, self.neurons_IL])
			dW2 = np.zeros([self.neurons_HL, self.neurons_HL])
			dW3 = np.zeros([self.neurons_OL, self.neurons_HL])
			db1 = np.zeros(self.neurons_HL)
			db2 = np.zeros(self.neurons_HL)
			db3 = np.zeros(self.neurons_OL)
		
			t = 0
			for example in batch:
				a2, a3, a4 = self.feedforward(example)
				delta4, delta3, delta2 = self.backpropagation(a4, a3, a2, target_list[t])
				D_W1, D_W2, D_W3, D_b1, D_b2, D_b3 = self.setDerivatives(delta4, delta3, delta2, example, a2, a3)
				
				dW1 += D_W1
				dW2 += D_W2
				dW3 += D_W3
				d_weights = [dW1, dW2, dW3]
				
				db1 += D_b1
				db2 += D_b2
				db3 += D_b3
				d_bias = [db1, db2, db3]
				
				t += 1
			self.updateWeights(t, d_weights, d_bias)
		self.writeWeights()
	
	def writeWeights(self):
		w1_file = open('./weights/weights1.txt', 'w')
		w2_file = open('./weights/weights2.txt', 'w')
		w3_file = open('./weights/weights3.txt', 'w')
		b1_file = open('./weights/b1.txt', 'w')
		b2_file = open('./weights/b2.txt', 'w')
		b3_file = open('./weights/b3.txt', 'w')
		
		np.savetxt(w1_file, self.W1)
		np.savetxt(w2_file, self.W2)
		np.savetxt(w3_file, self.W3)
		np.savetxt(b1_file, self.b1)
		np.savetxt(b2_file, self.b2)
		np.savetxt(b3_file, self.b3)
			
		w1_file.close()
		w2_file.close()
		w3_file.close()
		b1_file.close()
		b2_file.close()
		b3_file.close()
	
	
	def updateWeights(self, m, d_weights, d_bias):
		m = float(m)
		dW1, dW2, dW3 = d_weights[0], d_weights[1], d_weights[2]
		db1, db2, db3 = d_bias[0], d_bias[1], d_bias[2]
	
		self.W1 = self.W1 - self.learning_rate*((1.0/m)*dW1 + self.decay*self.W1)
		self.W2 = self.W2 - self.learning_rate*((1.0/m)*dW2 + self.decay*self.W2)
		self.W3 = self.W3 - self.learning_rate*((1.0/m)*dW3 + self.decay*self.W3)
		self.b1 = self.b1 - self.learning_rate*(1.0/m)*db1
		self.b2 = self.b2 - self.learning_rate*(1.0/m)*db2
		self.b3 = self.b3 - self.learning_rate*(1.0/m)*db3
	
	def setDerivatives(self, delta4, delta3, delta2, x, a2, a3):
		D_W1 = np.outer(delta2, x)
		D_W2 = np.outer(delta3, a2)
		D_W3 = np.outer(delta4, a3)
	
		D_b1 = delta2
		D_b2 = delta3
		D_b3 = delta4

		return D_W1, D_W2, D_W3, D_b1, D_b2, D_b3
	
	def backpropagation(self, a4, a3, a2, expected):
		delta4 = a4 - expected
		delta3 = (np.dot(self.W3.T, delta4))*self.sigmoid_d(a3)
		delta2 = (np.dot(self.W2.T, delta3))*self.sigmoid_d(a2)
		
		return delta4, delta3, delta2
		
	def feedforward(self, example):
		z1 = np.dot(self.W1, example) + self.b1
		a2 = self.sigmoid(z1)
		
		z2 = np.dot(self.W2, a2) + self.b2
		a3 = self.sigmoid(z2)
			
		z3 = np.dot(self.W3, a3) + self.b3
		a4 = self.sigmoid(z3)
			
		return a2, a3, a4
	
	def sigmoid(self, x):
		return 1.0/(1.0 + np.exp(-x))

	def sigmoid_d(self, x):
		return x*(1.0 - x)
	

if __name__ == '__main__':
	#70020 exemplos, 54 parametros
	rede = NeuralNetwork('pp1_training_file.csv', 70020, 54, 2, 0.1, 0.05)
	rede.train(5000)
