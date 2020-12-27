import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def sigmoid(x):
  return 1/(1+np.exp(-x))

# Load data
dataset = pd.read_csv("../data/data.csv")


# === PREPROCESSING ===

# Remove unnecessary data
data = dataset.drop(['name'], axis=1)
data = data.drop(['release_date'], axis=1)
data = data.drop(['id'], axis=1)
data = data.drop(['artists'], axis=1)

print(data)

batch_size = 100
epochs = 10

X = data.drop(['explicit'], axis=1).values
Y = data['explicit'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

w = np.zeros(( ))

print(X[0])
print(Y[0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(14, activation=tf.nn.relu, input_shape=(14,)),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(
    optimizer="SGD",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

tb_callback = tf.keras.callbacks.TensorBoard('true', update_freq=100, profile_batch=0)
model.fit(
    X_train, Y_train,
    batch_size=batch_size, epochs=epochs,
    callbacks=[tb_callback]
)

result = model.predict(X_test)
print(result)

# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)

# plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y,
#             s=25, edgecolor='k')

# plt.show()