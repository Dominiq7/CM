import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.func = self._step # Set to preferred function

        self.errors = []
        self.accuracies = []

    def _step(self, x):
        return np.where(x > 0, 1, 0)

    def _sigm(self, x):
        return 1/(1 + np.exp(-x))
    
    def _relu(self, x):
        return np.maximum(0.0, x)
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn()

        for i in range(self.n_iters): #running through epochs
            # print(f"Epoch: {i+1}")
            sum_error = 0
            predictions = []
            for idx, x_i in enumerate(X):
                
                output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.func(output)
                predictions.append(y_predicted)
                self.update(x_i, y[idx], y_predicted)
                sum_error += abs(y[idx] - output)

            accuracy = np.mean(predictions == y)
            error = (sum_error ** 2) / n_samples
            if i == self.n_iters-1:
                print(f"Accuracy: {accuracy}")
                print(f'Error: {error}')

            self.errors.append(error)
            self.accuracies.append(accuracy)
        
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Error and accuracy per epoch')
        ax1.set_title("MSE")
        ax1.plot(self.errors)
        ax2.set_title("Accuracy")
        ax2.plot(self.accuracies)

        plt.savefig("Perceptron.png", format="png")

    def update(self, x, y_true, y_pred):
        error = y_true - y_pred
        update = self.lr * error
        self.weights = self.weights + update * x
        self.bias = self.bias + update

    def predict(self, X):
        output = np.dot(X, self.weights) + self.bias
        y_predicted = self.func(output)
        return y_predicted
    
    def mse(self, target, output):
        return np.average((target - output)**2)