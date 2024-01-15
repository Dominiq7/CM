from perceptron import Perceptron
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(7)

data = pd.read_csv("pistachio.csv")

ratio = 0.8
total_rows = data.shape[0]
train_size = int(total_rows*ratio)

train = data[0:train_size]
test = data[train_size:]

X_train = train.iloc[:,:-1].to_numpy()
X_test = test.iloc[:,:-1].to_numpy()
y_train = np.where(train.iloc[:,-1]=="Kirmizi_Pistachio", 0, 1)
y_test = np.where(test.iloc[:,-1]=="Kirmizi_Pistachio", 0, 1)

p = Perceptron(n_iters=2000)
p.fit(X_train, y_train)

pred = p.predict(X_test)

print(f"[Test] Accuracy: {np.mean(pred == y_test)}")
