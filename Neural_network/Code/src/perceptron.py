import numpy as np
from matplotlib import pyplot as plt

class Perceptron(object):

    #initialization of the weights, learning rate and the number of epochs
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        # add one for bias
        self.epochs = epochs
        self.lr = lr

    #The activation function used is the sigmoid function
    def activation(self, x):
        return 1.0/(1.0+np.exp(-x))

    #Returns the prediction
    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation(z)

        return a

    def learning(self, X, d):
        errors = []
        for a in range(self.epochs):
            print("epoch: ", a)
            errorForEachInput = []
            for i in range(d.shape[0]):
                y = self.predict(X[i])
                e = d[i] - y
                errorForEachInput.append(e)

                print("predicted value: ", y)
                print("actual value: ", d[i])
                print("error: ", e)

                self.W = self.W + self.lr * e * np.insert(X[i], 0, 1)
                print("updated weights: ", self.W)
                print(" ")
            errors.append(errorForEachInput)
        return errors


#Learning the AND function
if __name__ == '__main__':
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    d = np.array([0, 0, 0, 1])

    perceptron = Perceptron(input_size=2)
    errors = perceptron.learning(X, d)
    print(errors)
    plt.title("AND function errors")
    plt.xlabel("# epochs")
    plt.ylabel("error")
    plt.plot(np.arange(100), errors)
    plt.show()
    print(perceptron.W)

#Learning the OR function
if __name__ == '__main__':
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    d = np.array([0, 1, 1, 1])

    perceptron = Perceptron(input_size=2)
    errors = perceptron.learning(X, d)
    plt.title("OR function errors")
    plt.xlabel("# epochs")
    plt.ylabel("error")
    plt.plot(np.arange(100), errors)
    plt.show()
    print(perceptron.W)

#Failing to learn the XOR function
if __name__ == '__main__':
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    d = np.array([0, 1, 1, 0])

    perceptron = Perceptron(input_size=2)
    errors = perceptron.learning(X, d)
    plt.title("XOR function errors")
    plt.xlabel("# epochs")
    plt.ylabel("error")
    plt.plot(np.arange(100), errors)
    plt.show()
    print(perceptron.W)