import random
import numpy as np
from sklearn.metrics import confusion_matrix


# sizes is the vector representing how many neurons we have on each layer
# biases and weights are randomly generated with the desired shape
class NeuralNetwork(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    # feedForward is calculating output = activation(w * input + b) for every layer
    def feedforward(self, inp):
        # we decided on having 3 layers so we only feedforward on hidden layer and output layer
        layer1 = self.activation(np.dot(self.weights[0], inp) + self.biases[0])
        output = self.activation(np.dot(self.weights[1], layer1) + self.biases[1])
        return output

    # training method splitting the training data, computing the gradient and calling backpropagation for each sample
    def training(self, training_data, epochs, sample_size, learning_rate,
                 evaluation_data=None,
                 monitor_evaluation_accuracy=False):
        n = len(training_data)
        evaluation_accuracy = []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+sample_size]
                for k in range(0, n, sample_size)]
            for mini_batch in mini_batches:
                self.gradient(
                    mini_batch, learning_rate)
            print("Epoch %s training complete" % j)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(float(int(accuracy / len(evaluation_data) * 10000)) / 100)
                print("Accuracy on evaluation data: {} / {} ({} percent) Hidden layer: {}".format(
                    accuracy, len(evaluation_data), float(int(accuracy / len(evaluation_data) * 10000)) / 100,
                    self.sizes[1]))
        return evaluation_accuracy

    # this method computes the stochastic gradient for updating the weights and biases after backpropagation
    def gradient(self, sample, learning_rate):
        biases = [np.zeros(bias.shape) for bias in self.biases]
        weights = [np.zeros(weight.shape) for weight in self.weights]

        # the backprop is called for each sample in order to learn new biases and weights
        for x, y in sample:
            sample_biases, sample_weights = self.backprop(x, y)
            biases = [old + new for old, new in zip(biases, sample_biases)]
            weights = [old + new for old, new in zip(weights, sample_weights)]

        size = len(sample)
        # here we compute the actual gradient to update weights and biases after learning with backprop
        self.weights = [weight - (learning_rate / size) * new_weight
                        for weight, new_weight in zip(self.weights, weights)]
        self.biases = [bias - (learning_rate / size) * new_bias for bias, new_bias in zip(self.biases, biases)]

    # this method is computing the algorithm of backpropagation
    def backprop(self, x, y):
        # setting up gradient matrix for cost function for the hidden and output layer
        bias = [np.zeros(b.shape) for b in self.biases]
        weights = [np.zeros(w.shape) for w in self.weights]

        # simulates the feedforward method step by step
        active = x
        # list to store all the activation functions for the hidden and output layer
        actives = [x]
        # list to store all the z vectors for the hidden and output layer
        zs = []
        for b, weight in zip(self.biases, self.weights):
            zs.append(np.dot(weight, active) + b)
            active = self.activation(np.dot(weight, active) + b)
            actives.append(active)
        # backward pass to propagate the error
        delta = (actives[-1] - y) * \
                (self.activation(zs[-1]) * (1 - self.activation(zs[-1])))
        bias[-1] = delta
        weights[-1] = np.dot(delta, actives[-2].transpose())

        # starting from hidden layer to the input layer
        for layer in range(2, 3):
            z = zs[-layer]
            # sigmoid prime
            sig_p = self.activation(z) * (1 - self.activation(z))
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sig_p
            bias[-layer] = delta
            weights[-layer] = np.dot(delta, actives[-layer - 1].transpose())
        return bias, weights

    # this method predicts the results on the test set
    def compute_unknown(self, data):
        results = [np.argmax(self.feedforward(x)) for x in data]

        return results

    # this method is computing the number of correct predictions and printing the confusion matrix
    def accuracy(self, data):
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        sm = 0
        prediction = []
        true = []

        for (x, y) in results:
            prediction.append(x)
            true.append(y)

            if x == y:
                sm += 1

        print(confusion_matrix(true, prediction))

        return sm

    # this method computes the activation function using the sigmoid function
    @staticmethod
    def activation(z):
        return 1.0/(1.0+np.exp(-z))

    # this method computes the derivative of the activation function
    def activation_derivative(self, z):
        return self.activation(z)*(1 - self.activation(z))
