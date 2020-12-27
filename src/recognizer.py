import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

batch_size = 100
epochs = 10
hidden_size = 100

# Load data
data = pd.read_csv("../data/data.csv")

# === PREPROCESSING ===

# Remove unnecessary data
data = data.drop(['name'], axis=1).drop(['release_date'], axis=1).drop(['id'], axis=1).drop(['artists'], axis=1)

# normalize
min_max_scaler = preprocessing.MinMaxScaler()
data = pd.DataFrame(min_max_scaler.fit_transform(data.values))

print(data.head())

explicit_data = data.values[np.where(data.values[:, 6] == 1)]
non_explicit_data = data.values[np.where(data.values[:, 6] == 0)]

non_explicit_data = non_explicit_data[0:len(explicit_data)]

print("explicit, nonexplicit")
print(explicit_data.shape[0])
print(non_explicit_data.shape[0])

explicit_train, explicit_test = train_test_split(explicit_data, test_size=0.2, random_state=1)
non_explicit_train, non_explicit_test = train_test_split(non_explicit_data, test_size=0.2, random_state=1)

print("explicit train test, nonexplicit train test")
print(explicit_train.shape[0])
print(explicit_test.shape[0])
print(non_explicit_train.shape[0])
print(non_explicit_test.shape[0])

train = np.concatenate((explicit_train, non_explicit_train))
test = np.concatenate((explicit_test, non_explicit_test))
np.random.shuffle(train)
np.random.shuffle(test)

print("train test")
print(train.shape[0])
print(test.shape[0])

train = pd.DataFrame(train)
test = pd.DataFrame(test)

train_X = train.drop(6, axis=1).values
train_Y = train[6].values

test_X = test.drop([6], axis=1).values
test_Y = test[6].values

# create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(train.shape[1], activation=tf.nn.sigmoid, input_shape=(14,)),
    tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.05),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# train
tb_callback = tf.keras.callbacks.TensorBoard('true', update_freq=1000, profile_batch=0)
model.fit(
    train_X, train_Y,
    batch_size=batch_size, epochs=epochs,
    callbacks=[tb_callback]
)

result = model.predict(test_X)
result_Y = (result > 0.5).flatten()

print(len(test_Y))
print()

a = 0
b = 0
c = 0
d = 0

for i in range(len(test_Y)):
  if (result_Y[i] == False and test_Y[i] == 0):
    a += 1;
  if (result_Y[i] == False and test_Y[i] == 1):
    b += 1;
  if (result_Y[i] == True and test_Y[i] == 0):
    c += 1;
  if (result_Y[i] == True and test_Y[i] == 1):
    d += 1;

print(a)
print(b)
print(c)
print(d)