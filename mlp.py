import numpy as np
import matplotlib.pyplot as plt
from random import random

np.random.seed(7)

class MLP:
    def __init__(self, input_size=3, hidden_layers=[3,3], output_size=2):

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        layers = [self.input_size] + self.hidden_layers + [self.output_size]

        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

        self.accuracies = []
        self.errors = []

    def forward(self, inputs):

        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)

            activations = self.softmax(net_inputs)
            self.activations[i+1] = activations

        return activations
    
    def back_propagate(self, error):

        for i in reversed(range(len(self.derivatives))):

            activations = self.activations[i+1]
            delta = error * self.softmax_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

        return error
    
    def gradient_descent(self, learning_rate):

        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0

            for j, (input, target) in enumerate(zip(inputs, targets)):

                output = self.forward(input)

                idx = target
                lst = [0] * self.output_size
                for k in range(len(lst)):
                    if k == idx:
                        lst[k] = 1
                target = lst

                error = target - output

                self.back_propagate(error)

                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

            preds = self.predict(inputs)
            accuracy = np.mean(preds == targets)

            if (i+1) % 100 == 0:
                print("Error: {} at epoch {}".format(sum_error / len(inputs), i+1))
                print(f"Accuracy: {accuracy}")

            self.accuracies.append(accuracy)
            self.errors.append(sum_error / (len(inputs)))

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Error and accuracy per epoch')
        ax1.set_title("MSE")
        ax1.plot(self.errors)
        ax2.set_title("Accuracy")
        ax2.plot(self.accuracies)

        plt.savefig("MLP.png", format="png")

    def predict(self, X):
        probs = self.forward(X)
        preds = probs.argmax(axis=1)
        return preds
    
    def softmax(self, x):
        t = np.exp(x - np.max(x))
        t = t / t.sum(axis=0, keepdims=True)
        return t
    
    def softmax_derivative(self, x):
        output = self.softmax(x)
        return output * (1 - output)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)
    
    def _mse(self, target, output):
        return np.average((target - output)**2)