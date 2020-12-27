import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


def sigmoid(x):
  return 1/(1+np.exp(-x))


# Load data
data = pd.read_csv("../data/data.csv")
print(data)


# === PREPROCESSING ===

# Remove unnecessary data
data = data.drop(['name'], axis=1)
data = data.drop(['release_date'], axis=1)
data = data.drop(['id'], axis=1)
data = data.drop(['artists'], axis=1)


X = data.drop(['explicit'], axis=1).values
Y = data['explicit'].values

w = np.zeros(( ))

print(X)
print(Y)

# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)

# plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y,
#             s=25, edgecolor='k')

# plt.show()