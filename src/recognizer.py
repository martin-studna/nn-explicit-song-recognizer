import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


def sigmoid(x)
  return 1/(1+np.exp(-x))

def initialize_with_zeros(m):
  """ 
  This function creates a vector of zeros of shape (m, 1) for w and       initializes b to 0.

  Argument:
  dim — size of the w vector we want (or number of parameters in this case)

  Returns:
  w — initialized vector of shape (dim, 1)
  b — initialized scalar (corresponds to the bias)
  """

  w = np.zeros((m, 1))
  b = 0

  return w, b

def propagate(w, b, X, Y):
  """
  Arguments:
  w — weights
  b — bias, a scalar
  X — input data 
  Y — true “label” vector
  Return:
  cost — negative log-likelihood cost for logistic regression
  dw — gradient of the loss with respect to w, thus same shape as w
  db — gradient of the loss with respect to b, thus same shape as b
  """

  m = X.shape[1]

  # FORWARD PROPAGATION (FROM X TO COST)
  A = sigmoid(np.dot(w.T, X)+ b) # compute activation
  cost = -(1/m)*(np.sum((Y*np.log(A)) + (1-Y) *np.log(1-A)))

  # BACKWARD PROPAGATION (TO FIND GRAD)
  dw = (1/m)* np.dot(X, ((A-Y).T))
  db = (1/m) * np.sum(A-Y)
  grads = {“dw”: dw,
  “db”: db}

  return grads, cost

def predict(w, b, X):
  """
  Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
  
  Arguments:
  w — weights, a numpy array of size (num_px * num_px * 3, 1)
  b — bias, a scalar
  X — data of size (num_px * num_px * 3, number of examples)
  
  Returns:
  Y_prediction — a numpy array (vector) containing all predictions (0/1) for the examples in X
  """
  
  m = X.shape[1]
  Y_prediction = np.zeros((1,m))
  w = w.reshape(X.shape[0], 1)
  
  A = sigmoid(np.dot(w.T, X) + b)
  
  for i in range(A.shape[1]):
  # Convert probabilities A[0,i] to actual predictions p[0,i]
  Y_prediction[0,i] = 1 if A[0, i] > 0.5 else 0
  pass

  
  return Y_prediction

data = pd.read_csv("data/data.csv")
print(data)


X = data.drop(['explicit'], axis=1).values
Y = data['explicit'].values

w = np.zeros(( ))

print(X)
print(Y)

# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y,
            s=25, edgecolor='k')

plt.show()