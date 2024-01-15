from mlp import MLP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(7)

data = pd.read_csv("IRIS.csv")
data['species'].replace({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}, inplace=True)

data_shuffled = data.sample(frac=1)

train_size = int(data.shape[0] * 0.8)

X = data_shuffled.iloc[:,:-1]
y = data_shuffled.iloc[:,-1:]

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

num_classes = len(np.unique(y_train))

model = MLP(input_size=X_train.shape[1], hidden_layers=[8], output_size=num_classes)

model.train(X_train.to_numpy(), y_train.to_numpy(), epochs=5000, learning_rate=0.1)

predictions = model.predict(X_test.to_numpy())

error = model._mse(y_test.to_numpy(), predictions)
accuracy = np.mean(predictions == y_test.to_numpy())

print(f"[TEST] Error: {error}, Accuracy: {accuracy}")